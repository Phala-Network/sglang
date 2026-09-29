"""CPU contract tests; GPU copies and remote Mooncake IO are separate gates."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from weakref import WeakKeyDictionary

import torch
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4LayerItem,
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    deepseek_v4_sidecar_specs,
    deepseek_v4_storage_schema,
)
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.utils import get_hash_str, get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PAGE = 128
COMPONENTS = [
    PoolName.KV,
    PoolName.SWA,
    PoolName.DEEPSEEK_V4_C4,
    PoolName.DEEPSEEK_V4_C4_INDEXER,
    PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
    PoolName.DEEPSEEK_V4_C4_STATE,
    PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
    PoolName.DEEPSEEK_V4_C128,
]


def make_manager(*, unified=False):
    names = (
        COMPONENTS
        if not unified
        else [
            n
            for n in COMPONENTS
            if n
            not in (
                PoolName.SWA,
                PoolName.DEEPSEEK_V4_C4_STATE,
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            )
        ]
    )
    pools = HostPoolGroup(
        [
            PoolEntry(
                name,
                LogicalHostPool(PAGE * 16, PAGE),
                None,
                lambda layer: layer,
                is_primary_index_anchor=name == PoolName.KV,
            )
            for name in names
        ]
    )
    cc = object.__new__(HybridCacheController)
    cc.mem_pool_host = pools
    cc.write_queue = []
    cc.ack_write_queue = []
    cc.backup_queue = Queue()
    cc.ack_backup_queue = Queue()
    cc.backup_skip = False
    cc.storage_backend_type = "mooncake"
    cc.get_hash_str = get_hash_str
    events = []

    def start_writing():
        op = cc.write_queue.pop(0)
        # Exercise real transfer expansion: every component must join the same
        # completion, rather than a primary-only completion releasing the row.
        transfers = cc._l2_transfers(
            op.host_indices, op.device_indices, op.pool_transfers
        )
        event = MagicMock()
        event.components = [t.host_pool for t in transfers]
        events.append(event)
        cc.ack_write_queue.append(HiCacheAck(None, event, op.node_ids))

    cc.start_writing = start_writing
    manager = object.__new__(DecodeKVCacheOffloadManager)
    manager.is_dsv4 = True
    manager.page_size = PAGE
    manager.offload_stride = PAGE
    manager.request_counter = 0
    manager.tp_world_size = 1
    manager.tp_group = None
    manager.decode_host_mem_pool = pools
    manager.cache_controller = cc
    manager.sidecar_specs = deepseek_v4_sidecar_specs(pools)
    manager.offloaded_state = WeakKeyDictionary()
    manager.offload_inflight = WeakKeyDictionary()
    for attr in (
        "ongoing_offload",
        "ongoing_backup",
        "offload_extra_pools",
        "backup_extra_pools",
    ):
        setattr(manager, attr, {})
    manager.req_to_token_pool = MagicMock()
    manager.req_to_token_pool.req_to_token = torch.arange(PAGE, PAGE * 9).repeat(2, 1)
    kv_pool = SimpleNamespace(translate_loc_from_full_to_swa=lambda x: x + PAGE * 20)
    manager.token_to_kv_pool_allocator = SimpleNamespace(get_kvcache=lambda: kv_pool)
    manager.tree_cache = MagicMock(protected_size_=0)
    return manager, events


def make_req(slot=0, rid="same-rid"):
    req = MagicMock()
    req.rid = rid
    req.origin_input_ids = list(range(PAGE + 7))
    req.output_ids = list(range(1000, 1000 + PAGE * 2))
    req.extra_key = "adapter-a"
    req.cache_salt = "tenant-a"
    req.kv.req_pool_idx = slot
    req.kv.kv_allocated_len = PAGE * 4
    req.prefix_indices = []
    req.finished.return_value = False
    return req


def finish_storage(manager, *, missing=None):
    operation = manager.cache_controller.backup_queue.get_nowait()
    operation.completed_tokens = len(operation.token_ids)
    operation.pool_storage_result.extra_pool_hit_pages = {
        t.name: len(operation.hash_value)
        for t in operation.pool_transfers
        if t.name != missing
    }
    manager.cache_controller.ack_backup_queue.put(operation)
    manager._check_backup_progress(1)
    return operation


def logical_store(manager):
    store = object.__new__(MooncakeStore)
    store.mem_pool_host = manager.decode_host_mem_pool.anchor_entry.host_pool
    store.registered_pools = {
        e.name: e.host_pool for e in manager.decode_host_mem_pool.entries
    }
    store.is_mla_backend = True
    store.mla_suffix = ""
    store.config_prefix = "schema"
    return store


class TestDSV4DecodeOffload(unittest.TestCase):
    def test_page_rounded_swa_window_accepts_256_and_rejects_512_stride(self):
        """A 128-token SWA window occupies one 256-token storage page.

        The startup regression used page=256/window=128 on NVIDIA's non-unified
        DSV4 path.  The lifecycle guard may round that window to one page, but
        a stride spanning two pages must still be rejected before the host
        stack is built.
        """
        module = "sglang.srt.disaggregation.decode_kvcache_offload_manager."
        fixture, _ = make_manager()
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool.swa_page_size = 256
        pool.sliding_window = 128
        pool._unified_kv = False
        allocator = SimpleNamespace(get_kvcache=lambda: pool, c128_attn_allocator=None)

        def construct(stride):
            with (
                patch(
                    module + "get_schedule", return_value=SimpleNamespace(page_size=256)
                ),
                patch(
                    module + "get_memory",
                    return_value=SimpleNamespace(
                        hicache_storage_backend_extra_config=None,
                        hicache_storage_backend="mooncake",
                    ),
                ),
                patch(
                    module + "get_serving",
                    return_value=SimpleNamespace(served_model_name="model"),
                ),
                patch(module + "torch.distributed.get_world_size", return_value=1),
                patch(
                    module + "envs",
                    SimpleNamespace(
                        SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE=SimpleNamespace(
                            get=lambda: stride
                        )
                    ),
                ),
                patch(
                    module + "build_deepseek_v4_hicache_stack",
                    return_value=(fixture.decode_host_mem_pool, fixture.cache_controller),
                ) as build,
            ):
                manager = DecodeKVCacheOffloadManager(
                    None, allocator, None, SimpleNamespace()
                )
                return manager, build

        manager, build = construct(None)
        self.assertEqual(manager.offload_stride, 256)
        self.assertTrue(build.called)

        with self.assertRaisesRegex(ValueError, "fit the live SWA window"):
            construct(512)

    def test_constructor_loads_at_file_config_without_enabling_radix(self):
        module = "sglang.srt.disaggregation.decode_kvcache_offload_manager."
        fixture, _ = make_manager()
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool.swa_page_size = PAGE
        pool.sliding_window = PAGE * 4
        pool._unified_kv = False
        allocator = SimpleNamespace(get_kvcache=lambda: pool, c128_attn_allocator=None)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "client-decode.json"
            config_path.write_text(
                json.dumps({"tenant": "fixture", "tag": "decode"}),
                encoding="utf-8",
            )
            with (
                patch(
                    module + "get_schedule", return_value=SimpleNamespace(page_size=PAGE)
                ),
                patch(
                    module + "get_memory",
                    return_value=SimpleNamespace(
                        hicache_storage_backend_extra_config=f"@{config_path}",
                        hicache_storage_backend="mooncake",
                    ),
                ),
                patch(
                    module + "get_serving",
                    return_value=SimpleNamespace(served_model_name="model"),
                ),
                patch(module + "torch.distributed.get_world_size", return_value=1),
                patch(
                    module + "build_deepseek_v4_hicache_stack",
                    return_value=(fixture.decode_host_mem_pool, fixture.cache_controller),
                ) as build,
            ):
                manager = DecodeKVCacheOffloadManager(
                    None, allocator, None, SimpleNamespace()
                )

            self.assertTrue(manager.is_dsv4)
            self.assertTrue(build.call_args.kwargs["params"].disable)
            self.assertEqual(
                build.call_args.kwargs["storage_backend_extra_config"],
                {"tenant": "fixture", "tag": "decode"},
            )

    def test_constructor_uses_hybrid_stack_without_radix_and_rejects_unsupported_geometry(
        self,
    ):
        module = "sglang.srt.disaggregation.decode_kvcache_offload_manager."
        fixture, _ = make_manager()
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool.swa_page_size = PAGE
        pool.sliding_window = PAGE * 4
        pool._unified_kv = False
        allocator = SimpleNamespace(get_kvcache=lambda: pool, c128_attn_allocator=None)
        with (
            patch(
                module + "get_schedule", return_value=SimpleNamespace(page_size=PAGE)
            ),
            patch(
                module + "get_memory",
                return_value=SimpleNamespace(
                    hicache_storage_backend_extra_config=None,
                    hicache_storage_backend="mooncake",
                ),
            ),
            patch(
                module + "get_serving",
                return_value=SimpleNamespace(served_model_name="model"),
            ),
            patch(module + "torch.distributed.get_world_size", return_value=1),
            patch(
                module + "build_deepseek_v4_hicache_stack",
                return_value=(fixture.decode_host_mem_pool, fixture.cache_controller),
            ) as build,
        ):
            manager = DecodeKVCacheOffloadManager(
                None, allocator, None, SimpleNamespace()
            )
            self.assertTrue(manager.is_dsv4)
            self.assertTrue(build.call_args.kwargs["params"].disable)
            self.assertIn(
                PoolName.DEEPSEEK_V4_C128, {s.pool_name for s in manager.sidecar_specs}
            )
            build.reset_mock()
            allocator.c128_attn_allocator = object()
            with self.assertRaisesRegex(ValueError, "NPU independent C128"):
                DecodeKVCacheOffloadManager(None, allocator, None, SimpleNamespace())
            build.assert_not_called()
            allocator.c128_attn_allocator = None
            pool.swa_page_size = PAGE // 2
            with self.assertRaisesRegex(ValueError, "matching full/SWA"):
                DecodeKVCacheOffloadManager(None, allocator, None, SimpleNamespace())
            build.assert_not_called()

    def test_distinct_p_and_d_schema_ignore_mount_path_and_capacity(self):
        p, _ = make_manager()
        d, _ = make_manager()
        d.decode_host_mem_pool.get_pool(PoolName.KV).size *= 2
        pool = SimpleNamespace(
            start_layer=0,
            end_layer=1,
            _unified_kv=False,
            layer_mapping=[DeepSeekV4LayerItem(4, 0, object())],
        )
        module = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
        parallel = SimpleNamespace(
            attn_tp_size=1, attn_cp_size=1, pp_size=1, attn_cp_rank=0
        )
        with patch(module + "get_parallel", return_value=parallel):
            with patch(
                module + "get_model",
                return_value=SimpleNamespace(model_path="/p/model", revision="abc"),
            ):
                p_schema = deepseek_v4_storage_schema(pool, p.decode_host_mem_pool)
            with patch(
                module + "get_model",
                return_value=SimpleNamespace(model_path="/d/model", revision="abc"),
            ):
                d_schema = deepseek_v4_storage_schema(pool, d.decode_host_mem_pool)
            self.assertEqual(p_schema, d_schema)
            method = HybridCacheController._storage_config_with_schema
            p_tag = method({}, p_schema)
            self.assertEqual(
                p_tag, method({}, d_schema)
            )  # Also proves JSON scalar serialization.
            with patch(
                module + "get_model",
                return_value=SimpleNamespace(model_path="/d/model", revision="def"),
            ):
                revision_schema = deepseek_v4_storage_schema(
                    pool, d.decode_host_mem_pool
                )
            self.assertNotEqual(p_tag, method({}, revision_schema))

            objects = set()

            def store_for(manager, schema, model_name="model", backend_tag="artifact"):
                store = object.__new__(MooncakeStore)
                store.mem_pool_host = (
                    manager.decode_host_mem_pool.anchor_entry.host_pool
                )
                store.registered_pools = {
                    e.name: e.host_pool for e in manager.decode_host_mem_pool.entries
                }
                store.is_mla_backend = True
                store.mla_suffix = ""
                config = method({"extra_backend_tag": backend_tag}, schema)
                store.config_prefix = config["extra_backend_tag"] + "_" + model_name
                store._batch_exist = lambda keys: [int(key in objects) for key in keys]
                return store

            d_store = store_for(d, d_schema)
            p_store = store_for(p, p_schema)
            indices = torch.arange(PAGE, PAGE * 3)
            d_transfers = d._dsv4_device_transfers(indices)
            p_transfers = p._dsv4_device_transfers(indices)
            page_keys = ["h1", "h2"]
            for transfer in d_transfers:
                keys, _ = d_store._get_hybrid_page_component_keys(page_keys, transfer)
                objects.update(d_store._tag_keys(keys))
            # Distinct P consumes D-created keys, despite different mount paths/capacities.
            self.assertEqual(
                p_store.batch_exists_v2(page_keys, p_transfers).kv_hit_pages, 2
            )
            wrong_readers = [
                store_for(p, revision_schema),
                store_for(p, p_schema, model_name="other-model"),
                store_for(p, p_schema, backend_tag="other-artifact"),
            ]
            c4 = p.decode_host_mem_pool.get_pool(PoolName.DEEPSEEK_V4_C4)
            for owner, field, value in (
                (c4, "layout", "page_first"),
                (c4, "page_size", PAGE * 2),
                (c4, "item_bytes", 123),
                (parallel, "attn_tp_size", 2),
                (parallel, "attn_cp_rank", 1),
            ):
                with (
                    patch.object(owner, field, value, create=True),
                    patch(
                        module + "get_model",
                        return_value=SimpleNamespace(revision="abc"),
                    ),
                ):
                    changed = deepseek_v4_storage_schema(pool, p.decode_host_mem_pool)
                wrong_readers.append(store_for(p, changed))
            for reader in wrong_readers:
                self.assertEqual(
                    reader.batch_exists_v2(page_keys, p_transfers).kv_hit_pages, 0
                )

    def test_two_generated_pages_share_prefill_hashes_and_complete_descriptors(self):
        manager, events = make_manager()
        req = make_req()
        self.assertTrue(manager.offload_kv_cache(req))
        event = events[0]
        event.synchronize.assert_called_once()
        self.assertEqual(len(event.components), len(COMPONENTS))
        manager._check_offload_progress(1)
        operation = manager.cache_controller.backup_queue.queue[0]
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        expected = get_storage_hash_str(
            RadixKey(
                all_tokens[: PAGE * 3],
                extra_key=req.extra_key,
                cache_salt=req.cache_salt,
            ),
            page_size=PAGE,
        )
        self.assertEqual(operation.hash_value, expected[1:])
        transfers = {t.name: t for t in operation.pool_transfers}
        self.assertEqual(set(transfers), set(COMPONENTS) - {PoolName.KV})
        for transfer in transfers.values():
            self.assertEqual(transfer.keys, expected[1:])
        self.assertIs(
            transfers[PoolName.DEEPSEEK_V4_C4].host_indices, operation.host_indices
        )
        self.assertIs(
            transfers[PoolName.DEEPSEEK_V4_C4_STATE].host_indices,
            transfers[PoolName.SWA].host_indices,
        )
        # Neither live GPU rows nor host pages have been released after D2H.
        manager.tree_cache.free_kv_row.assert_not_called()
        self.assertEqual(manager.decode_host_mem_pool.available_size(), PAGE * 14)
        finish_storage(manager)
        self.assertEqual(manager.decode_host_mem_pool.available_size(), PAGE * 16)
        self.assertEqual(
            manager.decode_host_mem_pool.available_size(PoolName.SWA), PAGE * 16
        )

    def test_unified_layout_has_no_swa_state_but_has_c128_and_fp4_scale(self):
        manager, _ = make_manager(unified=True)
        self.assertTrue(manager.offload_kv_cache(make_req()))
        names = {t.name for t in manager.offload_extra_pools[1]}
        self.assertNotIn(PoolName.SWA, names)
        self.assertIn(PoolName.DEEPSEEK_V4_C128, names)
        self.assertIn(PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE, names)

    def test_host_pressure_rolls_back_anchor_and_does_not_advance(self):
        manager, events = make_manager()
        swa = manager.decode_host_mem_pool.get_pool(PoolName.SWA)
        swa.alloc(swa.available_size())
        before = manager.decode_host_mem_pool.available_size()
        req = make_req()
        self.assertFalse(manager.offload_kv_cache(req))
        self.assertEqual(manager.decode_host_mem_pool.available_size(), before)
        self.assertEqual(manager.offloaded_state[req].inc_len, 0)
        self.assertFalse(events)

    def test_evicted_or_repeated_swa_slots_never_enqueue(self):
        for translate in (lambda x: torch.zeros_like(x), lambda x: torch.ones_like(x)):
            manager, events = make_manager()
            manager.token_to_kv_pool_allocator.get_kvcache().translate_loc_from_full_to_swa = translate
            self.assertFalse(manager.offload_kv_cache(make_req()))
            self.assertFalse(events)
            self.assertFalse(manager.ongoing_offload)

    def test_partial_storage_fails_explicitly_and_releases_once(self):
        manager, _ = make_manager()
        manager.offload_kv_cache(make_req())
        manager._check_offload_progress(1)
        with self.assertLogs(level="WARNING"):
            operation = finish_storage(
                manager, missing=PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE
            )
        self.assertFalse(manager.ongoing_backup)
        self.assertTrue(manager.cache_controller.backup_queue.empty())
        self.assertEqual(manager.decode_host_mem_pool.available_size(), PAGE * 16)
        # Duplicate completion must not double-free either independent allocation.
        manager.cache_controller.ack_backup_queue.put(operation)
        manager._check_backup_progress(1)
        self.assertEqual(manager.decode_host_mem_pool.available_size(), PAGE * 16)
        self.assertEqual(
            manager.decode_host_mem_pool.available_size(PoolName.SWA), PAGE * 16
        )

    def test_storage_exception_acknowledges_failure_and_releases_host(self):
        manager, _ = make_manager()
        manager.offload_kv_cache(make_req())
        manager._check_offload_progress(1)
        cc = manager.cache_controller
        cc.storage_stop_event = threading.Event()

        def fail(operation):
            cc.storage_stop_event.set()
            raise RuntimeError("private backend payload")

        cc._page_backup = fail
        with self.assertLogs(level="WARNING") as logs:
            cc.backup_thread_func()
            manager._check_backup_progress(1)
        self.assertNotIn("private backend payload", str(logs.output))
        self.assertFalse(manager.ongoing_backup)
        self.assertEqual(
            manager.decode_host_mem_pool.available_size(PoolName.SWA), PAGE * 16
        )

    def test_submission_exception_fences_and_rolls_back_all_allocations(self):
        manager, _ = make_manager()
        cc = manager.cache_controller
        cc.l2_transfer_engine = MagicMock()
        cc.start_writing = MagicMock(side_effect=RuntimeError("copy failed"))
        req = make_req()
        self.assertFalse(manager.offload_kv_cache(req))
        cc.l2_transfer_engine.device_to_host_stream.synchronize.assert_called_once()
        self.assertEqual(manager.decode_host_mem_pool.available_size(), PAGE * 16)
        self.assertEqual(
            manager.decode_host_mem_pool.available_size(PoolName.SWA), PAGE * 16
        )
        self.assertEqual(manager.offloaded_state[req].inc_len, 0)
        self.assertFalse(cc.write_queue)
        self.assertFalse(manager.ongoing_offload)

    def test_submission_failure_after_queue_clear_rolls_back_all_components(self):
        manager, _ = make_manager()
        cc = manager.cache_controller
        del cc.start_writing  # Exercise the actual inherited queue merge/clear.
        cc._move_write_operation = lambda op: (
            op.host_indices,
            op.device_indices,
            op.pool_transfers,
        )
        cc.l2_transfer_engine = MagicMock()
        cc.l2_transfer_engine.submit_device_to_host.side_effect = RuntimeError(
            "partial copy"
        )
        self.assertFalse(manager.offload_kv_cache(make_req()))
        cc.l2_transfer_engine.device_to_host_stream.synchronize.assert_called_once()
        self.assertFalse(cc.write_queue)
        self.assertFalse(cc.ack_write_queue)
        self.assertEqual(manager.decode_host_mem_pool.available_size(), PAGE * 16)
        self.assertEqual(
            manager.decode_host_mem_pool.available_size(PoolName.SWA), PAGE * 16
        )

    def test_existing_submission_queue_is_rejected_before_allocating_or_merging(self):
        manager, _ = make_manager()
        pending = object()
        manager.cache_controller.write_queue.append(pending)
        with self.assertRaisesRegex(AssertionError, "empty submission queue"):
            manager.offload_kv_cache(make_req())
        self.assertEqual(manager.cache_controller.write_queue, [pending])
        self.assertEqual(manager.decode_host_mem_pool.available_size(), PAGE * 16)

    def test_storage_enqueue_failure_releases_snapshot_and_finished_request(self):
        manager, _ = make_manager()
        req = make_req()
        req.finished.return_value = True
        manager.offload_kv_cache(req)
        manager.cache_controller.write_storage = MagicMock(
            side_effect=RuntimeError("queue closed")
        )
        with self.assertLogs(level="WARNING"):
            manager._check_offload_progress(1)
        self.assertFalse(manager.ongoing_backup)
        self.assertFalse(manager.ongoing_offload)
        self.assertEqual(
            manager.decode_host_mem_pool.available_size(PoolName.SWA), PAGE * 16
        )
        manager.req_to_token_pool.free.assert_called_once_with(req)

    def test_peer_submission_failure_rolls_back_successful_local_snapshot(self):
        manager, events = make_manager()
        manager.tp_world_size = 2
        calls = []

        def reduce(status, **kwargs):
            calls.append(status.item())
            if len(calls) == 2:
                status.fill_(0)  # Peer failed allocation after valid descriptors.

        req = make_req()
        with patch("torch.distributed.all_reduce", side_effect=reduce):
            self.assertFalse(manager.offload_kv_cache(req))
        self.assertEqual(calls, [1, 1])
        events[0].synchronize.assert_called_once()
        self.assertFalse(manager.cache_controller.ack_write_queue)
        self.assertFalse(manager.ongoing_offload)
        self.assertEqual(manager.offloaded_state[req].inc_len, 0)
        self.assertEqual(manager.decode_host_mem_pool.available_size(), PAGE * 16)
        self.assertEqual(
            manager.decode_host_mem_pool.available_size(PoolName.SWA), PAGE * 16
        )

    def test_peer_missing_swa_mapping_prevents_local_submission(self):
        manager, events = make_manager()
        manager.tp_world_size = 2
        with patch(
            "torch.distributed.all_reduce",
            side_effect=lambda status, **kwargs: status.fill_(0),
        ):
            self.assertFalse(manager.offload_kv_cache(make_req()))
        self.assertFalse(events)

    def test_duplicate_d2h_ack_does_not_create_second_backup(self):
        manager, _ = make_manager()
        manager.offload_kv_cache(make_req())
        ack = manager.cache_controller.ack_write_queue[0]
        manager._check_offload_progress(1)
        manager.cache_controller.ack_write_queue.append(ack)
        manager._check_offload_progress(1)
        self.assertEqual(manager.cache_controller.backup_queue.qsize(), 1)

    def test_shutdown_stops_transfers_before_destroy_and_is_idempotent(self):
        manager, _ = make_manager()
        order = []
        manager.cache_controller.l2_transfer_engine = MagicMock()
        manager.cache_controller.l2_transfer_engine.device_to_host_stream.synchronize.side_effect = (
            lambda: order.append("d2h")
        )
        manager.cache_controller.detach_storage_backend = lambda: order.append(
            "storage"
        )
        manager.decode_host_mem_pool.destroy = lambda: order.append("destroy")
        manager.release_host_resources()
        manager.release_host_resources()
        self.assertEqual(order, ["d2h", "storage", "destroy"])

    def test_cancellation_release_waits_for_combined_d2h_not_storage(self):
        manager, events = make_manager()
        old = make_req()
        manager.offload_kv_cache(old)
        old.finished.return_value = True
        manager.finalize_release_on_finish(old)
        manager.tree_cache.free_kv_row.assert_not_called()
        # Reusing rid must not join the new request to the old completion.
        new = make_req(slot=1)
        manager.offload_kv_cache(new)
        manager._check_offload_progress(1)
        manager.req_to_token_pool.free.assert_called_once_with(old)
        self.assertTrue(manager._has_inflight_offload(new))
        self.assertEqual(len(manager.ongoing_backup), 1)
        self.assertGreaterEqual(events[0].synchronize.call_count, 2)

    def test_released_slot_and_unaligned_tail_are_not_read(self):
        manager, events = make_manager()
        req = make_req()
        req.kv.req_pool_idx = None
        self.assertFalse(manager.offload_kv_cache(req))
        req.kv.req_pool_idx = 0
        req.output_ids = [1, 2]
        self.assertFalse(manager.offload_kv_cache(req))
        self.assertFalse(events)

    def test_schema_tag_is_canonical_and_preserves_input(self):
        method = HybridCacheController._storage_config_with_schema
        config = {"extra_backend_tag": "tenant-v1", "other": 7}
        schema = {"page": 128, "layout": "fp4", "tp": 1, "revision": "abc"}
        baseline = method(config, schema)
        self.assertEqual(config["extra_backend_tag"], "tenant-v1")
        self.assertEqual(baseline, method(config, dict(reversed(list(schema.items())))))
        for key, value in [
            ("page", 256),
            ("layout", "fp8"),
            ("tp", 2),
            ("revision", "def"),
        ]:
            changed = dict(schema, **{key: value})
            self.assertNotEqual(
                baseline["extra_backend_tag"],
                method(config, changed)["extra_backend_tag"],
            )
        self.assertNotEqual(
            baseline["extra_backend_tag"],
            method({"extra_backend_tag": "other"}, schema)["extra_backend_tag"],
        )

    def test_logical_anchor_requires_each_sidecar_and_missing_page_blocks_later_hits(
        self,
    ):
        manager, _ = make_manager()
        store = object.__new__(MooncakeStore)
        store.mem_pool_host = manager.decode_host_mem_pool.anchor_entry.host_pool
        store.registered_pools = {
            e.name: e.host_pool for e in manager.decode_host_mem_pool.entries
        }
        store.is_mla_backend = True
        store.mla_suffix = ""
        store.config_prefix = "schema"
        transfers = manager._dsv4_device_transfers(torch.arange(PAGE, PAGE * 3))
        store._batch_exist = lambda keys: [1] * len(keys)
        self.assertEqual(store.batch_exists_v2(["h1", "h2"], transfers).kv_hit_pages, 2)
        for omitted in transfers:
            subset = [t for t in transfers if t is not omitted]
            self.assertEqual(
                store.batch_exists_v2(["h1", "h2"], subset).kv_hit_pages, 0
            )
        self.assertEqual(store.batch_exists_v2(["h1", "h2"], None).kv_hit_pages, 0)
        store._batch_exist = lambda keys: []
        self.assertEqual(store.batch_exists_v2(["h1", "h2"], transfers).kv_hit_pages, 0)
        # A later successful chunk cannot hide a failed first generated C4 page.
        store._batch_exist = lambda keys: [
            0 if "h1" in key and "deepseek_v4_c4" in key else 1 for key in keys
        ]
        self.assertEqual(store.batch_exists_v2(["h1", "h2"], transfers).kv_hit_pages, 0)

    def test_missing_c128_object_prevents_restoring_across_later_complete_pages(self):
        manager, _ = make_manager()
        store = logical_store(manager)
        transfers = manager._dsv4_device_transfers(torch.arange(PAGE, PAGE * 4))
        c128 = next(t for t in transfers if t.name == PoolName.DEEPSEEK_V4_C128)
        missing_keys, _ = store._get_hybrid_page_component_keys(["h2"], c128)
        missing = set(store._tag_keys(missing_keys))
        store._batch_exist = lambda keys: [int(key not in missing) for key in keys]

        result = store.batch_exists_v2(["h1", "h2", "h3"], transfers)
        self.assertEqual(result.kv_hit_pages, 1)
        self.assertEqual(result.restorable_prefix_pages, [1])
        self.assertEqual(result.extra_pool_hit_pages[PoolName.DEEPSEEK_V4_C128], 1)
        # Every h3 object is present, but it must not bridge the C128 h2 gap.
        self.assertEqual(store.batch_exists_v2(["h2", "h3"], transfers).kv_hit_pages, 0)

    def test_trailing_state_gap_resumes_only_after_complete_required_window(self):
        manager, _ = make_manager()
        store = logical_store(manager)
        transfers = manager._dsv4_device_transfers(torch.arange(PAGE, PAGE * 5))
        for transfer in transfers:
            if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                transfer.keys = ["window-page-1", "window-page-2"]
        state = next(t for t in transfers if t.name == PoolName.DEEPSEEK_V4_C4_STATE)
        missing_keys, _ = store._get_hybrid_page_component_keys(["h2"], state)
        missing = set(store._tag_keys(missing_keys))
        store._batch_exist = lambda keys: [int(key not in missing) for key in keys]

        # At h3, the required two-page state window still includes missing h2.
        before = store.batch_exists_v2(["h1", "h2", "h3"], transfers)
        self.assertEqual(before.kv_hit_pages, 1)
        self.assertEqual(before.restorable_prefix_pages, [1])
        # h4 has a complete h3/h4 window for SWA and both C4 state sidecars.
        after = store.batch_exists_v2(["h1", "h2", "h3", "h4"], transfers)
        self.assertEqual(after.kv_hit_pages, 4)
        self.assertEqual(after.restorable_prefix_pages, [1, 4])


if __name__ == "__main__":
    unittest.main()
