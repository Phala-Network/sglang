"""CPU regression for failed HiCache I/O, TP ACK shape and staging ownership."""

import tempfile
import threading
import unittest
from array import array
from collections import defaultdict, deque
from contextlib import nullcontext
from datetime import timedelta
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from sglang.srt.managers.cache_controller import STORAGE_BATCH_SIZE
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, _OngoingPrefetch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _operation(rid="failure", *, missing=False, pages=2):
    swa = PoolTransfer(
        name=PoolName.SWA,
        keys=["placeholder"],
        host_indices=None if missing else torch.arange(100, 102),
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    derived = PoolTransfer(
        name=PoolName.DEEPSEEK_V4_C4_STATE,
        keys=["placeholder"],
        indices_from_pool=PoolName.SWA,
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    op = PrefetchOperation(
        CacheRequestHandle(rid, 0),
        list(range(pages * 2)),
        pool_transfers=[swa, derived],
    )
    op.hash_value = [f"h{i}" for i in range(pages)]
    op.storage_hit_count = pages * 2
    op.host_indices = torch.arange(pages * 2)
    return op


def _controller():
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.page_size = 2
    cc.mem_pool_host = SimpleNamespace(
        page_size=2,
        layout_lease=nullcontext,
        free=Mock(),
        entry_map={
            PoolName.SWA: SimpleNamespace(
                host_pool=SimpleNamespace(page_size=2), is_primary_index_anchor=False
            ),
            PoolName.DEEPSEEK_V4_C4_STATE: SimpleNamespace(
                host_pool=SimpleNamespace(page_size=2), is_primary_index_anchor=False
            ),
        },
    )
    cc.storage_stop_event = threading.Event()
    cc.prefetch_buffer = Queue()
    cc.prefetch_sync_queue = Queue()
    cc.ack_prefetch_queue = Queue()
    cc.host_mem_release_queue = Queue()
    cc.extra_host_mem_release_queues = {PoolName.SWA: Queue()}
    cc.prefetch_hit_queue = Queue()
    cc.ack_backup_queue = Queue()
    cc.prefetch_tokens_occupied = 0
    cc.prefetch_rate_limited = lambda: False
    cc.pp_prefetch_command_group = None
    cc.page_get_func = lambda op, hashes, indices, extra: len(hashes)
    # Exercise the actual adapter's pre-I/O validation/postprocessing; stub
    # only pointer metadata and the native network transfer.
    store = MooncakeStore.__new__(MooncakeStore)
    store.registered_pools = {
        name: SimpleNamespace(
            page_size=2,
            get_page_buffer_meta=lambda indices: (
                [1] * (len(indices) // 2),
                [8] * (len(indices) // 2),
            ),
        )
        for name in (PoolName.SWA, PoolName.DEEPSEEK_V4_C4_STATE)
    }
    store._tag_keys = lambda keys: keys
    store._get_hybrid_page_component_keys = lambda keys, transfer: (keys, 1)
    store._get_batch_zero_copy_impl = lambda keys, ptrs, sizes: [8] * len(keys)
    cc.storage_backend = store
    cc.prefetch_completion_sync_groups = []
    cc._all_reduce = lambda tensor, op, groups: None
    return cc


def _acks(cc):
    result = []
    while not cc.prefetch_sync_queue.empty():
        result.append(cc.prefetch_sync_queue.get_nowait())
    return result


def _run_worker(cc, operations):
    class StopQueue(Queue):
        def get(self, *args, **kwargs):
            item = super().get(*args, **kwargs)
            if item is None:
                cc.storage_stop_event.set()
            return item

    cc.prefetch_buffer = StopQueue()
    for op in operations:
        cc.prefetch_buffer.put(op)
    cc.prefetch_buffer.put(None)
    cc.prefetch_io_aux_func()
    return _acks(cc)


def _cache(cc, op):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.cache_controller = cc
    cache.host_memory_mode = "buffer_only"
    cache.tree_core = SimpleNamespace(page_size=2)
    cache.prefetch_threshold = 2
    cache.linker = None
    cache.storage_prefetch_retries = Mock()
    cache.discard_storage_prefetch_accounting = Mock()
    cache._handle_storage_prefetch_anchor_loss = Mock()
    cache.enable_storage_metrics = True
    cache.storage_metrics_collector = Mock()
    cache.storage_existence_cache = Mock()
    cache._finish_storage_prefetch = Mock()
    cache._invalidate_absent_from_hit_query = Mock()
    cache._record_storage_prefetch_hit = Mock()
    cache.revoke_pending_prefetch = Mock()
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.prefetch_loaded_storage_start_by_reqid = {}
    cache._prefetch_outcome_stats = defaultdict(int)
    cache.buffer_pipeline = SimpleNamespace(
        pending_hit_allocs=deque(),
        try_lock_anchor=Mock(return_value=("anchor_lost", 0)),
        release_staged_hold=Mock(return_value=False),
        pop_prefix_ctx=Mock(),
        release_anchor_lock=Mock(),
    )
    cache.ongoing_prefetch = {
        op.handle: _OngoingPrefetch(
            0,
            RadixKey(array("q", op.token_ids)),
            op.host_indices,
            op,
            None,
            {"aux": [t for t in op.pool_transfers if t.indices_from_pool is None]},
        )
    }
    cc.prefetch_tokens_occupied = len(op.host_indices)
    return cache


def _drain(cache, *, hits=0, acks=0):
    cache._drain_storage_control_queues_impl(
        n_storage_hit=hits,
        n_ack_prefetch=acks,
        n_backup=0,
        n_release=None,
        extra_release_counts={
            name: None for name in cache.cache_controller.extra_host_mem_release_queues
        },
        log_metrics=False,
    )


def _two_rank_failure(rank, path):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:
        cc = _controller()
        cc._all_reduce = lambda tensor, op, groups: torch.distributed.all_reduce(
            tensor, op=op
        )
        # Only rank 0 has invalid staging; both must reach the next operation.
        acks = _run_worker(cc, [_operation(missing=rank == 0), _operation("next")])
        for ack in acks:
            cc._reduce_prefetch_ack(ack)
        assert len(acks) == 6
        assert acks[1].pool_hits[PoolName.SWA] == 0
        assert acks[4].pool_hits[PoolName.SWA] == 1
        assert acks[2].completed_req and acks[5].completed_req
    finally:
        torch.distributed.destroy_process_group()


class TestPrefetchIOFailures(unittest.TestCase):
    def test_missing_required_sidecar_fails_then_worker_processes_next(self):
        cc = _controller()
        failed, next_op = _operation(missing=True), _operation("next")
        acks = _run_worker(cc, [failed, next_op])
        self.assertEqual(len(acks), 6)
        self.assertEqual(acks[1].pool_hits, {})
        self.assertEqual(acks[4].pool_hits[PoolName.SWA], 1)
        self.assertTrue(acks[2].completed_req)
        self.assertTrue(acks[5].completed_req)
        self.assertEqual(sum(bool(a.completed_req) for a in acks), 2)

    def test_sidecar_backend_exception_reclaims_once_without_false_hit(self):
        for cancel_after_acks in (None, 0, 1, 2):
            with self.subTest(cancel_after_acks=cancel_after_acks):
                cc = _controller()
                op = _operation()
                cc.storage_backend.batch_get_v2 = Mock(
                    side_effect=RuntimeError("fixture")
                )
                cache = _cache(cc, op)
                acks = _run_worker(cc, [op])
                for ack in acks:
                    cc._reduce_prefetch_ack(ack)
                    cc.ack_prefetch_queue.put(ack)
                if cancel_after_acks is not None:
                    # Cover each ownership boundary: before progress, after
                    # the KV ACK, and after the sidecar ACK but before terminal.
                    _drain(cache, acks=cancel_after_acks)
                    cache.release_aborted_request(op.handle)
                _drain(cache, acks=len(acks) - (cancel_after_acks or 0))
                self.assertFalse(cache.ongoing_prefetch)
                self.assertEqual(cc.prefetch_tokens_occupied, 0)
                freed = [
                    (call.args[0].tolist(), call.kwargs.get("pool", PoolName.KV))
                    for call in cc.mem_pool_host.free.call_args_list
                ]
                self.assertEqual(
                    freed, [([0, 1, 2, 3], PoolName.KV), ([100, 101], PoolName.SWA)]
                )
                self.assertEqual(
                    cache.prefetch_loaded_tokens_by_reqid.get(op.handle, 0), 0
                )
                cache.storage_metrics_collector.log_prefetched_tokens.assert_not_called()

    def test_later_sidecar_group_failure_discards_earlier_success_and_reclaims(self):
        cc = _controller()
        op = _operation()
        op.prefix_keys = ["prefix"]
        op.pool_transfers[0].keys = ["placeholder", "placeholder"]
        op.pool_transfers[0].host_indices = torch.arange(100, 104)
        op.pool_transfers[1].keys = ["placeholder", "placeholder"]
        op.pool_transfers.append(
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=["placeholder"],
                host_indices=torch.tensor([200]),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        )
        cc.mem_pool_host.entry_map[PoolName.MAMBA] = SimpleNamespace(
            host_pool=SimpleNamespace(page_size=1),
            is_primary_index_anchor=False,
        )
        cc.extra_host_mem_release_queues[PoolName.MAMBA] = Queue()
        cc.storage_backend.registered_pools[PoolName.MAMBA] = SimpleNamespace(
            page_size=1,
            get_page_buffer_meta=lambda indices: ([1], [8]),
        )
        cc.storage_backend._get_batch_zero_copy_impl = Mock(
            side_effect=[[8, 8], [8, 8], RuntimeError("last group")]
        )
        cache = _cache(cc, op)
        acks = _run_worker(cc, [op])
        self.assertEqual(cc.storage_backend._get_batch_zero_copy_impl.call_count, 3)
        self.assertEqual(acks[1].pool_hits, {})
        for ack in acks:
            cc._reduce_prefetch_ack(ack)
            cc.ack_prefetch_queue.put(ack)
        _drain(cache, acks=len(acks))
        freed = [
            (call.args[0].tolist(), call.kwargs.get("pool", PoolName.KV))
            for call in cc.mem_pool_host.free.call_args_list
        ]
        self.assertEqual(
            freed,
            [
                ([0, 1, 2, 3], PoolName.KV),
                ([100, 101, 102, 103], PoolName.SWA),
                ([200], PoolName.MAMBA),
            ],
        )
        self.assertEqual(cache.prefetch_loaded_tokens_by_reqid[op.handle], 0)
        self.assertEqual(cc.prefetch_tokens_occupied, 0)

    def test_kv_batch_exception_preserves_all_ack_slots_and_next_operation(self):
        cc = _controller()
        op = _operation(pages=STORAGE_BATCH_SIZE + 1)
        cc.page_get_func = Mock(side_effect=[RuntimeError("fixture"), 2])
        acks = _run_worker(cc, [op, _operation("next")])
        self.assertEqual(len(acks), 7)
        self.assertEqual([a.completed_tokens for a in acks[:2]], [0, 0])
        self.assertEqual(acks[2].pool_hits, {})
        self.assertTrue(acks[3].completed_req)
        self.assertEqual(acks[5].pool_hits[PoolName.SWA], 1)
        self.assertTrue(acks[6].completed_req)

    def test_zero_full_span_keeps_required_sidecar_failure(self):
        cc = _controller()
        op = _operation(missing=True)
        cc.trim_prefetch_full_head(op, 4)
        op.host_indices = torch.empty(0, dtype=torch.int64)
        acks = _run_worker(cc, [op])
        self.assertEqual(len(acks), 2)
        self.assertEqual(acks[0].pool_hits, {})
        self.assertTrue(acks[1].completed_req)
        self.assertEqual(op.pool_transfers[0].keys, ["h1"])

    def test_zero_full_span_with_valid_staging_restores_sidecars(self):
        cc = _controller()
        op = _operation()
        cc.trim_prefetch_full_head(op, 4)
        op.host_indices = torch.empty(0, dtype=torch.int64)
        acks = _run_worker(cc, [op])
        self.assertEqual(len(acks), 2)
        self.assertEqual(acks[0].pool_hits[PoolName.SWA], 1)
        self.assertEqual(acks[0].pool_hits[PoolName.DEEPSEEK_V4_C4_STATE], 1)
        self.assertTrue(acks[1].completed_req)

    def test_stale_attempt_cannot_revoke_or_allocate_replacement(self):
        for parked in (False, True):
            for terminated in (False, True):
                with self.subTest(parked=parked, terminated=terminated):
                    cc = _controller()
                    stale, current = _operation(), _operation()
                    if terminated:
                        stale.mark_terminate()
                    cache = _cache(cc, current)
                    if parked:
                        cache.buffer_pipeline.pending_hit_allocs.append(stale)
                    else:
                        cc.prefetch_hit_queue.put(stale)
                    _drain(cache, hits=int(not parked))
                    self.assertIs(
                        cache.ongoing_prefetch[current.handle].operation, current
                    )
                    self.assertTrue(cc.prefetch_buffer.empty())
                    cache.revoke_pending_prefetch.assert_not_called()
                    cache._handle_storage_prefetch_anchor_loss.assert_not_called()
                    cache._record_storage_prefetch_hit.assert_not_called()

    def test_two_rank_asymmetric_sidecar_failure_keeps_collectives_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(
                _two_rank_failure, args=(tmp + "/gloo",), nprocs=2, join=True
            )


if __name__ == "__main__":
    unittest.main()
