"""Runtime switches: re-read per use, adopted in step by every rank."""

import logging
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.unified_cache.components.base import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.utils import runtime_switches
from sglang.srt.utils.runtime_switches import RuntimeSwitchSync, RuntimeSwitchUpdate
from sglang.test.test_utils import published_topology

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_NAMES = [field.name for field in runtime_switches.SWITCHES]


class _SwitchTestCase(unittest.TestCase):
    def setUp(self):
        saved = {name: os.environ.get(name) for name in _NAMES}

        def restore():
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            runtime_switches._reset_for_tests()

        self.addCleanup(restore)
        for name in _NAMES:
            os.environ.pop(name, None)
        runtime_switches._reset_for_tests()


class TestRuntimeSwitchSync(_SwitchTestCase):
    def test_unarmed_reads_the_environment_per_call(self):
        field = envs.SGLANG_HICACHE_SWA_SPARSE_BACKUP
        self.assertFalse(runtime_switches.get(field))
        os.environ[field.name] = "1"
        self.assertTrue(runtime_switches.get(field))
        os.environ[field.name] = "0"
        self.assertFalse(runtime_switches.get(field))

    def test_environment_change_takes_effect_only_when_the_update_is_applied(self):
        field = envs.SGLANG_CHUNKED_PREFILL_SHORT_RESERVE
        os.environ[field.name] = "512"
        sync = RuntimeSwitchSync()
        self.assertEqual(runtime_switches.get(field), 512)
        self.assertIsNone(sync.poll())

        # A reload script ran on this rank: reads keep the agreed value until
        # the update rides the request broadcast back to every rank.
        os.environ[field.name] = "2048"
        self.assertEqual(runtime_switches.get(field), 512)
        update = sync.poll()
        self.assertIsInstance(update, RuntimeSwitchUpdate)
        # Stateless poll: a dropped update is sent again.
        self.assertEqual(sync.poll(), update)
        with self.assertLogs(runtime_switches.logger, logging.WARNING) as logs:
            sync.apply(update)
        self.assertIn("SGLANG_CHUNKED_PREFILL_SHORT_RESERVE='2048'", logs.output[0])
        self.assertEqual(runtime_switches.get(field), 2048)
        self.assertIsNone(sync.poll())

        # Unsetting falls back to the field default.
        del os.environ[field.name]
        sync.apply(sync.poll())
        self.assertEqual(runtime_switches.get(field), 0)

    def test_follower_rank_reads_the_broadcast_values_not_its_environment(self):
        field = envs.SGLANG_HICACHE_SWA_SPARSE_BACKUP
        follower = RuntimeSwitchSync()
        os.environ[field.name] = "true"  # reload landed early on this rank
        self.assertFalse(runtime_switches.get(field))
        # The source rank's (older) values win until it sends its own change.
        follower.apply(RuntimeSwitchUpdate(((field.name, "false"),)))
        self.assertFalse(runtime_switches.get(field))
        follower.apply(RuntimeSwitchUpdate(((field.name, "true"),)))
        self.assertTrue(runtime_switches.get(field))

    def test_unknown_names_in_an_update_are_ignored(self):
        sync = RuntimeSwitchSync()
        sync.apply(RuntimeSwitchUpdate((("SGLANG_SOMETHING_ELSE", "1"),)))
        self.assertNotIn("SGLANG_SOMETHING_ELSE", runtime_switches._agreed)

    def test_prefetch_policy_override(self):
        name = envs.SGLANG_HICACHE_PREFETCH_POLICY_OVERRIDE.name
        sync = RuntimeSwitchSync()
        self.assertEqual(
            runtime_switches.prefetch_policy("wait_complete"), "wait_complete"
        )
        for value, expected in (
            ("best_effort", "best_effort"),
            (" timeout ", "timeout"),
            ("", "wait_complete"),
        ):
            os.environ[name] = value
            sync.apply(sync.poll())
            self.assertEqual(
                runtime_switches.prefetch_policy("wait_complete"), expected
            )
        os.environ[name] = "bogus"
        sync.apply(sync.poll())
        with self.assertLogs(runtime_switches.logger, logging.WARNING):
            self.assertEqual(
                runtime_switches.prefetch_policy("wait_complete"), "wait_complete"
            )

    def test_prefill_decode_interval_override(self):
        name = envs.SGLANG_PREFILL_DECODE_INTERVAL_OVERRIDE.name
        sync = RuntimeSwitchSync()
        self.assertEqual(runtime_switches.prefill_decode_interval(3), 3)
        for value, expected in (("0", 0), ("8", 8), ("", 3), ("-1", 3), ("x", 3)):
            os.environ[name] = value
            sync.apply(sync.poll())
            self.assertEqual(runtime_switches.prefill_decode_interval(3), expected)


class TestSchedulerIntegration(_SwitchTestCase):
    def _scheduler(self, applied):
        sync = RuntimeSwitchSync()

        def process_input_requests(reqs):
            for req in reqs:
                if isinstance(req, RuntimeSwitchUpdate):
                    sync.apply(req)
                    applied.append(req)

        return SimpleNamespace(
            hot_reloader=None,
            runtime_switch_sync=sync,
            _deferred_input_requests=[],
            _poll_timeout_aborts=Mock(return_value=[]),
            request_receiver=SimpleNamespace(
                recv_requests=lambda local_reqs=None: list(local_reqs or [])
            ),
            metrics_reporter=Mock(),
            process_input_requests=process_input_requests,
        )

    def test_update_rides_the_request_broadcast(self):
        applied = []
        scheduler = self._scheduler(applied)
        field = envs.SGLANG_HICACHE_PREFETCH_POLICY_OVERRIDE
        with published_topology():
            Scheduler.ingest_requests(scheduler)
            self.assertEqual(applied, [])
            os.environ[field.name] = "best_effort"
            received = Scheduler.ingest_requests(scheduler)
            self.assertEqual(len(received), 1)
            self.assertEqual(runtime_switches.get(field), "best_effort")
            Scheduler.ingest_requests(scheduler)
        self.assertEqual(len(applied), 1)

    def test_prefill_decode_interval_override_is_read_per_decision(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.prefill_decode_interval = 0
        scheduler._prefill_decode_interval_remaining = 0
        scheduler.prefill_decode_interval_min_bs = 0
        scheduler.require_mlp_sync = False
        running = SimpleNamespace(batch_size=lambda: 1)
        extend = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True))
        sync = RuntimeSwitchSync()
        name = envs.SGLANG_PREFILL_DECODE_INTERVAL_OVERRIDE.name

        scheduler._arm_prefill_decode_interval(extend)
        self.assertFalse(scheduler._should_defer_prefill(running))

        os.environ[name] = "3"
        sync.apply(sync.poll())
        scheduler._arm_prefill_decode_interval(extend)
        self.assertTrue(scheduler._should_defer_prefill(running))

        # Lowering it mid-countdown releases the deferred prefill early.
        os.environ[name] = "0"
        sync.apply(sync.poll())
        self.assertFalse(scheduler._should_defer_prefill(running))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)

    def test_prefetch_termination_rereads_the_policy(self):
        cache = SimpleNamespace(prefetch_stop_policy="wait_complete", pp_rank=0)
        cache._all_reduce = lambda tensor, op: None
        cache._prefetch_timeout_check_linear_func = lambda op: True
        cache._prefetch_deadline_expired = lambda op: True
        can_terminate = UnifiedRadixCache._can_terminate_prefetch
        can_terminate = getattr(can_terminate, "__wrapped__", can_terminate)
        sync = RuntimeSwitchSync()
        name = envs.SGLANG_HICACHE_PREFETCH_POLICY_OVERRIDE.name
        operation = object()

        self.assertFalse(can_terminate(cache, operation))
        for value, expected in (("best_effort", True), ("timeout", True), ("", False)):
            os.environ[name] = value
            sync.apply(sync.poll())
            self.assertEqual(can_terminate(cache, operation), expected)

    def test_sparse_swa_backup_is_decided_per_backup(self):
        component = SimpleNamespace(
            component_type=ComponentType.SWA,
            tree_core=SimpleNamespace(has_swa_host_pool=True, enable_hicache=True),
            sliding_window_size=128,
            _collect_unbacked_swa_nodes=Mock(return_value=[]),
        )
        node = SimpleNamespace(backup_chain_tail_len=100)
        sync = RuntimeSwitchSync()
        name = envs.SGLANG_HICACHE_SWA_SPARSE_BACKUP.name

        def window():
            SWAComponent.build_hicache_transfers(
                component, node, CacheTransferPhase.BACKUP_HOST
            )
            return component._collect_unbacked_swa_nodes.call_args.args[1]

        self.assertIsNone(window())  # default: the full trailing window
        os.environ[name] = "1"
        self.assertIsNone(window())  # not adopted until the update is applied
        sync.apply(sync.poll())
        self.assertEqual(window(), 128 - 100)
        os.environ[name] = "0"
        sync.apply(sync.poll())
        self.assertIsNone(window())


if __name__ == "__main__":
    unittest.main()
