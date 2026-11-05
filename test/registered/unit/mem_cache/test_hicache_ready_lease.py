"""The ready-counts lease: a group decision, with the collective's MIN semantics."""

import threading
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache import hicache_ready_lease
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODULE = "sglang.srt.mem_cache.hicache_ready_lease"


def _ack(ready):
    return SimpleNamespace(finish_event=SimpleNamespace(query=lambda: ready))


class _Rendezvous:
    """A MIN all-reduce between threads: a rank that enters alone times out."""

    def __init__(self, ranks):
        self.barrier = threading.Barrier(ranks, timeout=5)
        self.tensors = {}
        self.lock = threading.Lock()
        self.rounds = 0

    def all_reduce(self, tensor, op, group):
        with self.lock:
            self.tensors[threading.get_ident()] = tensor
        if self.barrier.wait() == 0:
            result = torch.stack(list(self.tensors.values())).min(dim=0).values
            for peer in self.tensors.values():
                peer.copy_(result)
            self.tensors = {}
            self.rounds += 1
        self.barrier.wait()


def _cache(write_acks=(), load_acks=()):
    cache = object.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(write_back_duplicate_reclaim_digest=0)
    cache.pp_rank, cache.pp_size = 0, 1
    cache.host_memory_mode = "cache"
    cache.enable_storage = False
    cache.enable_storage_metrics = False
    cache.storage_metrics_collector = None
    cache.buffer_pipeline = None
    cache.linker = None
    cache.is_write_back = False
    cache._ready_counts_group = object()
    cache._pending_ready_counts = None
    cache.work_list = []
    cache.writing_check = MagicMock()
    cache.loading_check = MagicMock()
    cache.cache_controller = SimpleNamespace(
        write_policy="write_through",
        start_writing=MagicMock(),
        ack_write_queue=list(write_acks),
        ack_load_queue=list(load_acks),
    )
    return cache


class _Eligible(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("get_memory", SimpleNamespace(hicache_write_policy="write_through")),
            ("get_parallel", SimpleNamespace(dp_size=1)),
            ("get_disagg", SimpleNamespace(disaggregation_mode="null")),
            ("get_device", SimpleNamespace(sleep_on_idle=False)),
        ):
            patcher = patch(f"{MODULE}.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)


class TestLeaseLockstep(_Eligible):
    def _run(self, caches, passes, leases=None):
        """Run ``passes`` scheduler passes on one thread per rank; a mismatched
        collective breaks the barrier and fails the run."""
        rendezvous = _Rendezvous(len(caches))
        errors = Queue()

        def rank(cache):
            try:
                for lease in leases or [4] * passes:
                    hicache_ready_lease.check_hicache_events_leased(cache, lease)
            except Exception as e:  # noqa: BLE001 - reported on the main thread
                errors.put(e)

        with patch("torch.distributed.all_reduce", rendezvous.all_reduce):
            threads = [threading.Thread(target=rank, args=(c,)) for c in caches]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertTrue(errors.empty(), errors.queue)
        return rendezvous

    def test_idle_group_runs_one_round_per_lease(self):
        caches = [_cache(), _cache()]
        rendezvous = self._run(caches, 8)
        self.assertEqual(rendezvous.rounds, 2)
        for cache in caches:
            # Stock's per-pass work runs on skipped passes too.
            self.assertEqual(cache.cache_controller.start_writing.call_count, 8)

    def test_one_rank_with_work_keeps_every_rank_in_rounds(self):
        # A rank with a ready ack while its peer has none: the MIN drains nothing,
        # and the group must not lease, or the ready rank's ack would wait for nothing.
        ready, empty = _cache([_ack(True)]), _cache()
        rendezvous = self._run([ready, empty], 8)
        self.assertEqual(rendezvous.rounds, 8)
        for cache in (ready, empty):
            for args in cache.writing_check.call_args_list:
                self.assertEqual(args.kwargs, {"finish_count": 0})

    def test_counts_are_the_group_minimum(self):
        a = _cache([_ack(True), _ack(True)], [_ack(True)])
        b = _cache([_ack(True), _ack(False)], [_ack(True)])
        self._run([a, b], 1)
        for cache in (a, b):
            cache.writing_check.assert_called_once_with(finish_count=1)
            cache.loading_check.assert_called_once_with(finish_count=1)

    def test_a_load_in_flight_holds_the_lease_off(self):
        caches = [_cache(load_acks=[_ack(False)]), _cache()]
        rendezvous = self._run(caches, 6)
        self.assertEqual(rendezvous.rounds, 6)

    def test_an_unfinished_backup_does_not(self):
        caches = [_cache(write_acks=[_ack(False)]), _cache()]
        rendezvous = self._run(caches, 8)
        self.assertEqual(rendezvous.rounds, 2)

    def test_work_during_a_lease_waits_for_the_next_round(self):
        cache = _cache()
        with patch("torch.distributed.all_reduce"):
            hicache_ready_lease.check_hicache_events_leased(cache, 3)
            cache.cache_controller.ack_write_queue.append(_ack(True))
            hicache_ready_lease.check_hicache_events_leased(cache, 3)
            hicache_ready_lease.check_hicache_events_leased(cache, 3)
            self.assertEqual(cache.writing_check.call_count, 1)
            hicache_ready_lease.check_hicache_events_leased(cache, 3)
        cache.writing_check.assert_called_with(finish_count=1)

    def test_a_lease_length_change_keeps_lockstep(self):
        caches = [_cache(), _cache()]
        rendezvous = self._run(caches, 0, leases=[4, 4, 2, 2, 2, 2, 8, 8, 8])
        self.assertGreaterEqual(rendezvous.rounds, 2)


class TestLeaseEligibility(_Eligible):
    def _falls_back(self, cache):
        cache.check_hicache_events = MagicMock()
        hicache_ready_lease.check_hicache_events_leased(cache, 4)
        cache.check_hicache_events.assert_called_once_with()

    def test_ineligible_layouts_take_the_stock_path(self):
        for attr, value in (
            ("pp_size", 2),
            ("_ready_counts_group", None),
            ("is_write_back", True),
            ("buffer_pipeline", object()),
            ("linker", object()),
            ("host_memory_mode", "buffer_only"),
            ("cache_controller", None),
        ):
            with self.subTest(attr=attr):
                cache = _cache()
                setattr(cache, attr, value)
                self._falls_back(cache)

    def test_sleep_on_idle_takes_the_stock_path(self):
        with patch(
            f"{MODULE}.get_device", return_value=SimpleNamespace(sleep_on_idle=True)
        ):
            self._falls_back(_cache())

    def test_a_subclass_with_its_own_events_takes_the_stock_path(self):
        class Other(UnifiedRadixCache):
            def check_hicache_events(self):
                self.called = True

        cache = _cache()
        cache.__class__ = Other
        hicache_ready_lease.check_hicache_events_leased(cache, 4)
        self.assertTrue(cache.called)


class TestSchedulerDispatch(unittest.TestCase):
    def _process(self, value):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hierarchical_cache = True
        scheduler.enable_unified_cache_external_linker = False
        scheduler.enable_lmcache = False
        scheduler.enable_hicache_storage = False
        scheduler.tree_cache = MagicMock()
        env = {} if value is None else {"SGLANG_HICACHE_READY_COUNTS_LEASE": value}
        with (
            patch.dict("os.environ", env),
            patch(f"{MODULE}.check_hicache_events_leased") as leased,
        ):
            scheduler._process_hicache_events()
        return scheduler.tree_cache.check_hicache_events, leased

    def test_off_by_default(self):
        stock, leased = self._process(None)
        stock.assert_called_once_with()
        leased.assert_not_called()

    def test_one_is_off(self):
        stock, leased = self._process("1")
        stock.assert_called_once_with()
        leased.assert_not_called()

    def test_on(self):
        stock, leased = self._process("8")
        stock.assert_not_called()
        self.assertEqual(leased.call_args.args[1], 8)


if __name__ == "__main__":
    unittest.main()
