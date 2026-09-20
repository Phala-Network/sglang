"""Execute selected cache source methods without importing the SGLang runtime.

Default mode uses CPU collective/tensor stand-ins. Pass --gloo to use real
PyTorch tensors and run a real two-rank CPU Gloo MIN collective as well.
An optional first positional argument selects a checkout or installed-source
shadow root, matching test_chunked_prefill_physical_budget.py.
"""

import ast
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

SOURCE = (
    Path(sys.argv.pop(1))
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else Path(os.environ.get("HICACHE_ACK_TEST_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
)
RUN_GLOO = "--gloo" in sys.argv or os.environ.get("HICACHE_ACK_TEST_GLOO") == "1"
# Spawn imports this file afresh after the CLI arguments have been consumed.
# Preserve the selected installed-source shadow and real-tensor mode in workers.
os.environ["HICACHE_ACK_TEST_SOURCE_ROOT"] = str(SOURCE.resolve())
os.environ["HICACHE_ACK_TEST_GLOO"] = "1" if RUN_GLOO else "0"
if RUN_GLOO:
    if "--gloo" in sys.argv:
        sys.argv.remove("--gloo")
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
else:
    class Tensor:
        def __init__(self, values):
            self.values = list(values)

        def tolist(self):
            return self.values.copy()

        def clone(self):
            return Tensor(self.values)

        def __getitem__(self, index):
            return SimpleNamespace(item=lambda: self.values[index])

        def __setitem__(self, index, value):
            self.values[index] = value

    dist = SimpleNamespace(
        all_reduce=MagicMock(), get_world_size=MagicMock(),
        ReduceOp=SimpleNamespace(MIN="MIN"),
    )
    torch = SimpleNamespace(
        tensor=lambda values, **kwargs: Tensor(values),
        int64="int64", int="int", distributed=dist,
    )

logger = logging.getLogger(__name__)
get_memory = lambda: SimpleNamespace(hicache_write_policy="write_through")
get_disagg = lambda: SimpleNamespace(disaggregation_mode="null")
get_parallel = lambda: SimpleNamespace(dp_size=1)
UnifiedCacheLinkerWrapper = lambda cache, linker: linker
METHODS = {
    "_single_ready_counts_group", "_ready_counts_tensor", "_parse_ready_counts",
    "_sync_hicache_ready_counts", "_async_ready_counts_eligible",
    "_issue_async_ready_counts", "_consume_async_ready_counts",
    "_drain_pending_ready_counts", "_count_ready_acks", "_apply_ready_counts",
    "check_hicache_events", "reset", "release_host_resources",
    "init_cache_linker", "attach_storage_backend", "detach_storage_backend",
    "enable_storage", "is_write_back",
}
source_path = SOURCE / "python/sglang/srt/mem_cache/unified_radix_cache.py"
tree = ast.parse(source_path.read_text(encoding="utf-8"))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnifiedRadixCache")
cls.bases = []
cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
exec(compile(ast.fix_missing_locations(ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls
], type_ignores=[])), str(source_path), "exec"), globals())


def _gloo_min_worker(rank, world_size, init_method):
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            enable_storage=False,
            write_back_duplicate_reclaim_digest=17,
        )
        cache.cache_controller = SimpleNamespace(
            ack_write_queue=[
                SimpleNamespace(
                    finish_event=SimpleNamespace(query=lambda: True)
                )
                for _ in range(2 if rank == 0 else 1)
            ],
            ack_load_queue=[],
            extra_host_mem_release_queues={},
        )
        cache.pp_rank = 0
        cache._ready_counts_group = dist.group.WORLD
        cache._pending_ready_counts = None

        cache._issue_async_ready_counts()
        ready = cache._consume_async_ready_counts()

        assert ready is not None
        assert ready[0] == 1
        assert ready[1:] == (0, (), ())
        assert cache._pending_ready_counts is None
    finally:
        dist.destroy_process_group()


class _FakeWork:
    def __init__(self, on_wait=None):
        self.waited = False
        self.on_wait = on_wait

    def wait(self):
        self.waited = True
        if self.on_wait is not None:
            self.on_wait()


class TestHiCacheAsyncAckSync(unittest.TestCase):
    def _cache(self, write_ready=(True, True, False), load_ready=()):
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            enable_storage=False,
            is_write_back=False,
            write_back_duplicate_reclaim_digest=0,
        )
        cache.pp_rank = 0
        cache.pp_size = 1
        cache.work_list = []
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache.linker = None
        cache.host_memory_mode = "cache"
        cache._ready_counts_group = object()
        cache._hicache_async_ack_sync_requested = True
        cache._hicache_async_ack_sync_logged = True
        cache._pending_ready_counts = None
        cache._drain_async_work = MagicMock()
        cache._async_ready_counts_eligible = MagicMock(return_value=True)
        cache.cache_controller = SimpleNamespace(
            ack_write_queue=[
                SimpleNamespace(
                    finish_event=SimpleNamespace(query=MagicMock(return_value=ready))
                )
                for ready in write_ready
            ],
            ack_load_queue=[
                SimpleNamespace(
                    finish_event=SimpleNamespace(query=MagicMock(return_value=ready))
                )
                for ready in load_ready
            ],
            extra_host_mem_release_queues={},
            write_policy="write_through",
        )
        cache.write_counts = []
        cache.load_counts = []

        def writing_check(*, finish_count):
            cache.write_counts.append(finish_count)
            for _ in range(finish_count):
                cache.cache_controller.ack_write_queue.pop(0)

        def loading_check(*, finish_count):
            cache.load_counts.append(finish_count)
            for _ in range(finish_count):
                cache.cache_controller.ack_load_queue.pop(0)

        cache.writing_check = writing_check
        cache.loading_check = loading_check
        return cache

    def test_pops_previous_counts_before_reducing_remainder(self):
        cache = self._cache()
        issued = []

        def all_reduce(tensor, *, op, group, async_op):
            self.assertTrue(async_op)
            issued.append(tensor.clone())
            return _FakeWork()

        with patch.object(torch.distributed, "all_reduce", all_reduce):
            cache.check_hicache_events()
            self.assertEqual(cache.write_counts, [])
            self.assertEqual(issued[0][0].item(), 2)

            cache.check_hicache_events()
            self.assertEqual(cache.write_counts, [2])
            self.assertEqual(len(cache.cache_controller.ack_write_queue), 1)
            self.assertEqual(issued[1][0].item(), 0)

    def test_waits_before_parsing_and_applying_reduced_count(self):
        cache = self._cache(write_ready=(True,))
        works = []

        def all_reduce(tensor, *, op, group, async_op):
            # Simulate another rank reporting zero ready acks. Parsing before
            # wait would incorrectly consume the local ready ack.
            work = _FakeWork(lambda: tensor.__setitem__(0, 0))
            works.append(work)
            return work

        with patch.object(torch.distributed, "all_reduce", all_reduce):
            cache.check_hicache_events()
            self.assertFalse(works[0].waited)
            cache.check_hicache_events()

        self.assertTrue(works[0].waited)
        self.assertEqual(cache.write_counts, [0])
        self.assertEqual(len(cache.cache_controller.ack_write_queue), 1)

    def test_reset_drains_pending_work_before_resetting_tree(self):
        cache = self._cache(write_ready=(True,))
        work = _FakeWork()
        cache._pending_ready_counts = (
            work,
            torch.tensor([1, 0, 0, 0], dtype=torch.int64),
            (),
            0,
        )

        def assert_drained_before_reset():
            self.assertTrue(work.waited)
            self.assertIsNone(cache._pending_ready_counts)
            self.assertEqual(cache.write_counts, [1])
            self.assertEqual(cache.cache_controller.ack_write_queue, [])

        cache._reset_full = MagicMock(side_effect=assert_drained_before_reset)

        cache.reset()

        self.assertTrue(work.waited)
        self.assertIsNone(cache._pending_ready_counts)
        cache._reset_full.assert_called_once_with()

    def test_release_host_resources_drains_before_destroy(self):
        cache = self._cache(write_ready=(True,))
        work = _FakeWork()
        cache._pending_ready_counts = (
            work,
            torch.tensor([1, 0, 0, 0], dtype=torch.int64),
            (),
            0,
        )
        cache.host_pool_group = MagicMock()

        def assert_drained_before_destroy():
            self.assertTrue(work.waited)
            self.assertIsNone(cache._pending_ready_counts)
            self.assertEqual(cache.write_counts, [1])
            self.assertEqual(cache.cache_controller.ack_write_queue, [])

        cache.host_pool_group.destroy.side_effect = assert_drained_before_destroy

        cache.release_host_resources()

        self.assertTrue(work.waited)
        cache.host_pool_group.destroy.assert_called_once_with()

    def test_write_back_and_pd_modes_are_not_eligible(self):
        cache = self._cache(write_ready=())
        cache._async_ready_counts_eligible = (
            UnifiedRadixCache._async_ready_counts_eligible.__get__(cache)
        )

        memory = SimpleNamespace(hicache_write_policy="write_through")
        disagg = SimpleNamespace(disaggregation_mode="null")
        with (
            patch(
                __name__ + ".get_memory",
                return_value=memory,
            ),
            patch(
                __name__ + ".get_disagg",
                return_value=disagg,
            ),
        ):
            self.assertTrue(cache._async_ready_counts_eligible())
            cache.tree_core.is_write_back = True
            self.assertFalse(cache._async_ready_counts_eligible())
            cache.tree_core.is_write_back = False
            disagg.disaggregation_mode = "decode"
            self.assertFalse(cache._async_ready_counts_eligible())

    def test_unsupported_cache_modes_and_topologies_are_not_eligible(self):
        cache = self._cache(write_ready=())
        cache._async_ready_counts_eligible = (
            UnifiedRadixCache._async_ready_counts_eligible.__get__(cache)
        )
        memory = SimpleNamespace(hicache_write_policy="write_through")
        disagg = SimpleNamespace(disaggregation_mode="null")

        with (
            patch(
                __name__ + ".get_memory",
                return_value=memory,
            ),
            patch(
                __name__ + ".get_disagg",
                return_value=disagg,
            ),
        ):
            cache._hicache_async_ack_sync_requested = False
            self.assertFalse(cache._async_ready_counts_eligible())
            cache._hicache_async_ack_sync_requested = True

            cache.pp_size = 2
            self.assertFalse(cache._async_ready_counts_eligible())
            cache.pp_size = 1
            cache._ready_counts_group = None
            self.assertFalse(cache._async_ready_counts_eligible())
            cache._ready_counts_group = object()

            cache.tree_core.enable_storage = True
            self.assertFalse(cache._async_ready_counts_eligible())
            cache.tree_core.enable_storage = False
            cache.buffer_pipeline = object()
            self.assertFalse(cache._async_ready_counts_eligible())
            cache.buffer_pipeline = None
            cache.linker = object()
            self.assertFalse(cache._async_ready_counts_eligible())

    def test_dp_buffer_only_and_write_policy_guards(self):
        cache = self._cache(write_ready=())
        eligible = UnifiedRadixCache._async_ready_counts_eligible.__get__(cache)
        with patch(__name__ + ".get_parallel", return_value=SimpleNamespace(dp_size=2)):
            self.assertFalse(eligible())
        cache.host_memory_mode = "buffer_only"
        self.assertFalse(eligible())
        cache.host_memory_mode = "cache"
        cache.cache_controller.write_policy = "write_back"
        self.assertFalse(eligible())
        cache.cache_controller.write_policy = "write_through"
        with patch(__name__ + ".get_memory", return_value=SimpleNamespace(hicache_write_policy="write_back")):
            self.assertFalse(eligible())

    def test_single_process_group_selection(self):
        cache = self._cache()
        cache.attn_cp_group, cache.attn_tp_group, cache.tp_group = object(), object(), object()
        cache.tp_world_size = 8
        with patch.object(dist, "get_world_size", side_effect=lambda group: 1 if group is cache.attn_cp_group else 8):
            self.assertIs(cache._single_ready_counts_group(), cache.attn_tp_group)
        with patch.object(dist, "get_world_size", return_value=2):
            self.assertIsNone(cache._single_ready_counts_group())
        cache.attn_cp_group = cache.attn_tp_group = None
        self.assertIs(cache._single_ready_counts_group(), cache.tp_group)
        cache.tp_world_size = 1
        self.assertIsNone(cache._single_ready_counts_group())

    def test_storage_pp_placeholder_shape_and_synchronous_dispatch(self):
        cache = self._cache(write_ready=(True,))
        cache.tree_core.enable_storage = True
        cc = cache.cache_controller
        for name, count in [("prefetch_hit_queue", 3), ("ack_prefetch_queue", 4), ("ack_backup_queue", 5), ("host_mem_release_queue", 6)]:
            setattr(cc, name, SimpleNamespace(qsize=lambda count=count: count))
        cc.extra_host_mem_release_queues = {"mamba": SimpleNamespace(qsize=lambda: 7)}
        tensor, names, digest = cache._ready_counts_tensor()
        self.assertEqual(tensor.tolist(), [1, 0, 3, 4, 5, 6, 7, 0, 0])
        self.assertEqual(names, ("mamba",))
        cache.pp_rank = 1
        self.assertEqual(cache._ready_counts_tensor()[0].tolist(), [0] * 9)
        cache.pp_rank = 0
        cache._async_ready_counts_eligible.return_value = False
        cache._all_reduce = MagicMock()
        cache._drain_storage_control_queues_impl = MagicMock()
        cache.buffer_pipeline = SimpleNamespace(flush_pending_writes=MagicMock())
        cache.check_hicache_events()
        cache._all_reduce.assert_called_once()
        cache._drain_storage_control_queues_impl.assert_called_once_with(
            n_storage_hit=3, n_ack_prefetch=4, n_backup=5, n_release=6,
            extra_release_counts={"mamba": 7}, log_metrics=True,
        )
        cache.buffer_pipeline.flush_pending_writes.assert_called_once()
        self.assertIsNone(cache._pending_ready_counts)

    def test_fallback_drains_pending_then_syncs_remaining_queues(self):
        cache = self._cache(write_ready=(True, True))
        with patch.object(dist, "all_reduce", return_value=_FakeWork()):
            cache.check_hicache_events()
        cache._async_ready_counts_eligible.return_value = False
        cache._all_reduce = MagicMock()
        cache.check_hicache_events()
        self.assertEqual(cache.write_counts, [2, 0])
        self.assertIsNone(cache._pending_ready_counts)

    def test_storage_and_linker_attach_drain_before_mutation(self):
        for operation in ("attach", "detach", "linker"):
            with self.subTest(operation=operation):
                cache = self._cache(write_ready=(True,))
                work = _FakeWork()
                cache._pending_ready_counts = (work, torch.tensor([1, 0, 0, 0]), (), 0)
                def observe(**kwargs):
                    self.assertTrue(work.waited)
                    self.assertEqual(cache.write_counts, [1])
                    self.assertIsNone(cache._pending_ready_counts)
                    return True, "ok"
                cache._storage_attachment = SimpleNamespace(attach=MagicMock(side_effect=observe), detach=MagicMock(side_effect=observe))
                if operation == "attach":
                    self.assertEqual(cache.attach_storage_backend("file"), (True, "ok"))
                elif operation == "detach":
                    self.assertEqual(cache.detach_storage_backend(), (True, "ok"))
                else:
                    linker = object()
                    cache.init_cache_linker(linker)
                    observe()
                    self.assertIs(cache.linker, linker)

    def test_digest_mismatch_rejected_and_issue_time_digest_retained(self):
        cache = self._cache(write_ready=())
        cache.tree_core.write_back_duplicate_reclaim_digest = 17
        with patch.object(dist, "all_reduce", return_value=_FakeWork()):
            cache._issue_async_ready_counts()
        cache.tree_core.write_back_duplicate_reclaim_digest = 23
        self.assertEqual(cache._consume_async_ready_counts(), (0, 0, (), ()))
        with self.assertRaisesRegex(AssertionError, "diverged"):
            cache._parse_ready_counts(torch.tensor([0, 0, 17, -18]), (), 17)

    def test_load_counts_wait_and_pop_once(self):
        cache = self._cache(write_ready=(), load_ready=(True, False, True))
        with patch.object(dist, "all_reduce", return_value=_FakeWork()):
            cache.check_hicache_events()
            cache.check_hicache_events()
            cache._drain_pending_ready_counts()
        self.assertEqual(cache.load_counts, [1, 0])
        self.assertEqual(len(cache.cache_controller.ack_load_queue), 2)

    @unittest.skipUnless(RUN_GLOO, "use --gloo with PyTorch for a real collective")
    def test_real_two_rank_gloo_min_reduce(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_method = "file://" + os.path.join(tmp, "gloo-init")
            mp.spawn(
                _gloo_min_worker,
                args=(2, init_method),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    print(f"source={SOURCE}; real_gloo={RUN_GLOO}")
    unittest.main(verbosity=2)
