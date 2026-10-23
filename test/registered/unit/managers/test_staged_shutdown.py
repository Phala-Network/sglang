"""Staged shutdown: the order of the release steps, their bounds, idempotence,
the helpers each step uses, and the ranks' optional turns.

A rank releases its driver objects while it is alive (each free is a short hold
of the NVIDIA driver's global lock) instead of leaving them to process exit (one
hold for all of them, which stalls every other process on the host); see
staged_shutdown.py. No GPU here: the CUDA calls are stand-ins.
"""

import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import staged_shutdown
from sglang.srt.managers.staged_shutdown import drop_device_tensors, teardown_turn
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

POOL_MODULE = "sglang.srt.mem_cache.fake_pool"


class _SubPool:
    def __init__(self):
        self.buffers = [torch.zeros(4), torch.zeros(4)]
        self.by_layer = {0: torch.zeros(2)}
        self.mixed = [torch.zeros(1), 3]  # not all tensors: kept


class _Pool:
    def __init__(self):
        self.kv = torch.zeros(8)
        self.view = self.kv[:4]  # a view of the same storage
        self.sub = _SubPool()
        self.subs = [_SubPool()]
        self.size = 8


class _Elsewhere:  # not a memory-pool object: never walked
    def __init__(self):
        self.t = torch.zeros(16)


for cls in (_SubPool, _Pool):
    cls.__module__ = POOL_MODULE


class TestDropDeviceTensors(unittest.TestCase):
    def test_unbinds_pool_tensors_and_counts_distinct_storages(self):
        pool = _Pool()
        pool.other = _Elsewhere()
        n = drop_device_tensors(pool, None, device_type="cpu")
        self.assertIsNone(pool.kv)
        self.assertIsNone(pool.view)
        self.assertIsNone(pool.sub.buffers)
        self.assertIsNone(pool.sub.by_layer)
        self.assertIsNone(pool.subs[0].buffers)
        self.assertEqual(len(pool.sub.mixed), 2)
        self.assertEqual(pool.size, 8)
        self.assertIsNotNone(pool.other.t)
        # kv (+ its view, one storage) 32 B, per sub-pool 2x16 + 8 B.
        self.assertEqual(n, 32 + 2 * (2 * 16 + 8))

    def test_other_devices_untouched(self):
        pool = _Pool()
        self.assertEqual(drop_device_tensors(pool), 0)  # cuda: none here
        self.assertIsNotNone(pool.kv)

    def test_depth_bound(self):
        pool = _Pool()
        drop_device_tensors(pool, max_depth=0, device_type="cpu")
        self.assertIsNone(pool.kv)
        self.assertIsNotNone(pool.sub.buffers)


class TestTeardownTurn(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "turn.lock")

    def tearDown(self):
        self.dir.cleanup()

    def test_turns_do_not_overlap(self):
        inside, overlaps = [0], [0]
        guard = threading.Lock()

        def rank():
            with teardown_turn(self.path, 10, "test"):
                with guard:
                    inside[0] += 1
                    overlaps[0] = max(overlaps[0], inside[0])
                time.sleep(0.05)
                with guard:
                    inside[0] -= 1

        threads = [threading.Thread(target=rank) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(overlaps[0], 1)

    def test_timeout_runs_the_body_anyway(self):
        held = threading.Event()
        release = threading.Event()

        def holder():
            with teardown_turn(self.path, 10, "holder"):
                held.set()
                release.wait(5)

        t = threading.Thread(target=holder)
        t.start()
        held.wait(5)
        start = time.monotonic()
        with teardown_turn(self.path, 0.3, "waiter"):
            waited = time.monotonic() - start
        release.set()
        t.join()
        self.assertGreaterEqual(waited, 0.3)
        self.assertLess(waited, 3)

    def test_no_path_or_unopenable_path_runs_the_body(self):
        ran = []
        with teardown_turn("", 1, "test"):
            ran.append(1)
        with teardown_turn(os.path.join(self.dir.name, "no", "such", "dir"), 1, "test"):
            ran.append(2)
        self.assertEqual(ran, [1, 2])


class _Scheduler:
    def __init__(self, calls):
        self.calls = calls
        self.tree_cache = SimpleNamespace(
            shutdown=lambda: calls.append("storage detach")
        )
        self.token_to_kv_pool_allocator = None
        self.req_to_token_pool = None

    def release_host_resources(self):
        self.calls.append("host resources")


class TestStagedRelease(unittest.TestCase):
    """The sequence with every GPU-touching step replaced by a recorder."""

    def setUp(self):
        self.calls = []
        staged_shutdown._released.clear()
        self.addCleanup(staged_shutdown._released.clear)
        hooks = patch.dict(staged_shutdown._RELEASE_HOOKS, clear=True)
        hooks.start()
        self.addCleanup(hooks.stop)
        rec = self.calls.append
        for name, fn in (
            ("destroy_cuda_graphs", lambda: rec("CUDA graphs")),
            ("close_communicators", lambda: rec("communicators")),
            ("release_tensors", lambda scheduler: rec("tensors")),
            ("reset_cuda_context", lambda device: rec(f"context reset {device}")),
            ("_current_device", lambda: 0),
            ("_exit_now", lambda code: rec(f"exit {code}")),
        ):
            p = patch.object(staged_shutdown, name, fn)
            p.start()
            self.addCleanup(p.stop)
        for target, name in (
            ("torch.cuda.synchronize", "synchronize"),
            ("torch.cuda.set_device", None),
            ("sglang.srt.utils.rank_consensus_checker.shutdown", "rank consensus"),
        ):
            p = patch(target, (lambda n=name: rec(n)) if name else (lambda *a: None))
            p.start()
            self.addCleanup(p.stop)
        for env, value in (
            (envs.SGLANG_STAGED_SHUTDOWN_LOCK, ""),
            (envs.SGLANG_STAGED_SHUTDOWN_CTX_RESET, True),
            (envs.SGLANG_STAGED_SHUTDOWN_TIMEOUT, 30),
            (envs.SGLANG_STAGED_SHUTDOWN_STEP_TIMEOUT, 10),
        ):
            o = env.override(value)
            o.__enter__()
            self.addCleanup(o.__exit__, None, None, None)

    def test_steps_run_in_order_then_the_process_exits(self):
        staged_shutdown.register_release_hook(
            "ext", lambda: self.calls.append("plugin")
        )
        staged_shutdown.staged_release(_Scheduler(self.calls), tp_rank=1)
        self.assertEqual(
            self.calls,
            [
                "synchronize",
                "storage detach",
                "rank consensus",
                "CUDA graphs",
                "plugin",
                "host resources",
                "communicators",
                "tensors",
                "context reset 0",
                "exit 0",
            ],
        )

    def test_a_second_call_does_nothing(self):
        staged_shutdown.staged_release(_Scheduler(self.calls), 1, exit_process=False)
        first = list(self.calls)
        self.assertEqual(staged_shutdown.staged_release(_Scheduler(self.calls), 1), [])
        self.assertEqual(self.calls, first)

    def test_a_failing_step_does_not_stop_the_next(self):
        def boom():
            raise RuntimeError("no peer")

        staged_shutdown.register_release_hook("ext", boom)
        record = staged_shutdown.staged_release(
            _Scheduler(self.calls), 1, exit_process=False
        )
        self.assertIn(("plugin ext", "failed"), [(n, s) for n, s, _ in record])
        self.assertIn("host resources", self.calls)
        self.assertIn("context reset 0", self.calls)

    def test_a_hung_step_is_left_behind_and_the_rest_still_run(self):
        release = threading.Event()
        self.addCleanup(release.set)
        staged_shutdown.register_release_hook("ext", lambda: release.wait(10))
        with envs.SGLANG_STAGED_SHUTDOWN_STEP_TIMEOUT.override(0.2):
            start = time.monotonic()
            record = staged_shutdown.staged_release(
                _Scheduler(self.calls), 1, exit_process=False
            )
        self.assertLess(time.monotonic() - start, 5)
        self.assertIn(("plugin ext", "timeout"), [(n, s) for n, s, _ in record])
        self.assertEqual(
            self.calls[-3:], ["communicators", "tensors", "context reset 0"]
        )

    def test_past_the_budget_steps_are_skipped_but_the_context_reset_runs(self):
        staged_shutdown.register_release_hook("ext", lambda: time.sleep(0.4))
        with envs.SGLANG_STAGED_SHUTDOWN_TIMEOUT.override(0.3):
            record = staged_shutdown.staged_release(
                _Scheduler(self.calls), 1, exit_process=False
            )
        status = {n: s for n, s, _ in record}
        self.assertEqual(status["plugin ext"], "timeout")
        self.assertEqual(status["host resources"], "skipped")
        self.assertEqual(status["tensors"], "skipped")
        self.assertEqual(status["context reset"], "ok")
        self.assertEqual(self.calls[-1], "context reset 0")

    def test_without_the_context_reset_the_process_is_not_ended_here(self):
        with envs.SGLANG_STAGED_SHUTDOWN_CTX_RESET.override(False):
            staged_shutdown.staged_release(_Scheduler(self.calls), 1)
        self.assertNotIn("context reset 0", self.calls)
        self.assertNotIn("exit 0", self.calls)
        self.assertEqual(self.calls[-1], "tensors")

    def test_tp0_logs_its_inventory_first(self):
        staged_shutdown.register_inventory_hook("ext", lambda: {"channels": 1})
        self.addCleanup(staged_shutdown._INVENTORY_HOOKS.pop, "ext", None)
        with self.assertLogs(staged_shutdown.logger, "INFO") as logs:
            staged_shutdown.staged_release(
                _Scheduler(self.calls), 0, exit_process=False
            )
        first = logs.output[0]
        self.assertIn("inventory at shutdown start", first)
        self.assertIn('"ext": {"channels": 1}', first)
        # One line per step on TP0.
        self.assertTrue(any("TP0: tensors " in line for line in logs.output))


class TestExitTimeout(unittest.TestCase):
    def test_defaults_cover_the_staged_release(self):
        with envs.SGLANG_STAGED_SHUTDOWN.override(False):
            self.assertEqual(staged_shutdown.scheduler_exit_timeout(), 15)
        with envs.SGLANG_STAGED_SHUTDOWN.override(True):
            self.assertEqual(
                staged_shutdown.scheduler_exit_timeout(),
                envs.SGLANG_STAGED_SHUTDOWN_TIMEOUT.get()
                + envs.SGLANG_STAGED_SHUTDOWN_STEP_TIMEOUT.get()
                + 20,
            )
            with envs.SGLANG_SCHEDULER_EXIT_TIMEOUT.override(42):
                self.assertEqual(staged_shutdown.scheduler_exit_timeout(), 42)


class _Graph:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class TestHelpers(unittest.TestCase):
    def test_every_graph_is_reset(self):
        graphs = [_Graph(), _Graph()]
        holder = {"decode": graphs[0], "verify": [graphs[1]]}
        with patch.object(staged_shutdown, "_cuda_graph_types", lambda: (_Graph,)):
            self.assertEqual(staged_shutdown.destroy_cuda_graphs(), "reset 2 graph(s)")
        self.assertEqual([g.resets for g in graphs], [1, 1])
        self.assertIs(holder["decode"], graphs[0])

    def test_sweep_unbinds_every_tensor_on_the_device_and_nothing_else(self):
        owner = SimpleNamespace(
            weight=torch.nn.Parameter(torch.empty(8, device="meta")),
            buf=torch.empty(4, device="meta"),
            cpu=torch.zeros(4),
        )
        view = owner.buf[:2]
        with torch.inference_mode():
            owner.inference = torch.empty(3, device="meta")
        swept = staged_shutdown.sweep_tensors(device_type="meta", pinned=False)
        self.assertGreaterEqual(swept["unbound"], 4)
        self.assertEqual(swept["failed"], 0)
        for t in (owner.weight, owner.buf, view, owner.inference):
            self.assertEqual(t.numel(), 0)
        self.assertEqual(owner.cpu.numel(), 4)

    def test_communicators_close_before_nccl_and_process_groups_last(self):
        calls = []

        class Lib:
            def ncclCommAbort(self, comm):
                calls.append(f"abort {comm}")
                return 0

        lib = Lib()
        abort = lib.ncclCommAbort
        lib.ncclCommAbort = lambda comm: abort(comm)
        group = SimpleNamespace(
            unique_name="tp",
            ca_comm=SimpleNamespace(close=lambda: calls.append("ca close")),
            qr_comm=None,
            pcie_ipc_comm=None,
            pymscclpp_comm=None,
            torch_symm_mem_comm=object(),
            pynccl_comm=SimpleNamespace(
                disabled=False, comm=7, nccl=SimpleNamespace(lib=lib)
            ),
        )
        with (
            patch.object(staged_shutdown, "_group_coordinators", lambda: [group]),
            patch(
                "sglang.srt.distributed.parallel_state.abort_distributed_environment",
                lambda: calls.append("process groups"),
            ),
            patch(
                "sglang.srt.layers.flashinfer_comm_fusion.cleanup_flashinfer_workspace",
                lambda: calls.append("flashinfer"),
            ),
        ):
            staged_shutdown.close_communicators()
        self.assertEqual(calls, ["ca close", "flashinfer", "abort 7", "process groups"])
        self.assertIsNone(group.ca_comm)
        self.assertIsNone(group.torch_symm_mem_comm)
        self.assertTrue(group.pynccl_comm.disabled)

    def test_inventory_survives_a_failing_part(self):
        def broken():
            raise RuntimeError("gone")

        with patch.dict(staged_shutdown._INVENTORY_HOOKS, {"ext": broken}):
            inv = staged_shutdown.inventory()
        self.assertIn("error", inv["ext"])
        self.assertIn("cuda_graphs", inv["heap"])
        self.assertIn("nvidia_fds", inv["driver"])


if __name__ == "__main__":
    unittest.main()
