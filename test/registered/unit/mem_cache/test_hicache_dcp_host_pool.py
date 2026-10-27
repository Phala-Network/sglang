"""HiCache under decode context parallelism (DCP): host-pool index math.

Under DCP the radix/controller layer works in a widened logical index space
(page_size * dcp_size wide pages, dcp_size * physical capacity), while each
rank's device and host buffers only materialize the owned 1/dcp_size token
shard (owner rule: index % dcp_size == dcp_rank, physical row = index //
dcp_size — the same rule the device-side KV write and page-table kernels
use). These tests cover the translation helper, the logical/physical host
pool sizing, and that the transfer entry points hand *physical* rows to the
kernels.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.arg_groups.hicache_hook import resolve_hicache_dcp_compatibility
from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig, PoolName
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DCP_SIZE = 8
PHYSICAL_PAGE = 64
WIDENED_PAGE = PHYSICAL_PAGE * DCP_SIZE


def _fake_mla_device_pool(size: int = 1024, dtype=torch.float16) -> SimpleNamespace:
    return SimpleNamespace(
        size=size,
        # Match KVCache's default: this static pool's size already counts tokens.
        host_capacity_tokens=None,
        store_dtype=dtype,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        layer_num=2,
        start_layer=0,
        end_layer=1,
        device="cpu",
        layers_to_capture=None,
        layer_shard_enabled=False,
    )


def _make_host_pool(
    dcp_rank: int,
    device_size: int = 1024,
    dcp_size: int = DCP_SIZE,
    layout: str = "layer_first",
    dtype=torch.float16,
) -> MLATokenToKVPoolHost:
    return MLATokenToKVPoolHost(
        _fake_mla_device_pool(device_size, dtype),
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=PHYSICAL_PAGE * dcp_size,
        layout=layout,
        pin_memory=False,
        device="cpu",
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
    )


class TestDcpKernelIndices(CustomTestCase):
    def test_identity_without_dcp(self):
        indices = torch.arange(37)
        self.assertIs(maybe_dcp_kernel_indices(indices, 1, 0), indices)

    def test_aligned_page_translates_to_full_physical_page(self):
        # One widened page starting at logical 512 covers physical rows
        # 64..127 on every rank.
        indices = torch.arange(WIDENED_PAGE, 2 * WIDENED_PAGE)
        for rank in range(DCP_SIZE):
            out = maybe_dcp_kernel_indices(indices, DCP_SIZE, rank)
            torch.testing.assert_close(
                out, torch.arange(PHYSICAL_PAGE, 2 * PHYSICAL_PAGE)
            )

    def test_matches_owner_rule_on_merged_unordered_pages(self):
        # Concatenation of non-adjacent widened pages in arbitrary order, as
        # produced by merged CacheOperations after allocator churn.
        pages = [3, 0, 5]
        indices = torch.cat(
            [torch.arange(p * WIDENED_PAGE, (p + 1) * WIDENED_PAGE) for p in pages]
        )
        for rank in range(DCP_SIZE):
            out = maybe_dcp_kernel_indices(indices, DCP_SIZE, rank)
            expected = (
                indices[indices % DCP_SIZE == rank] // DCP_SIZE
            )  # owner rule, same as filter_dcp_local_kv_indices
            torch.testing.assert_close(out, expected)
            self.assertEqual(out.numel() * DCP_SIZE, indices.numel())

    def test_positional_residue_pairing_survives_host_sort(self):
        # move_indices (direct/layer_first) sorts host indices and permutes
        # device indices to match. Independent residue filtering of both
        # tensors must keep the same token positions on every rank.
        g = torch.Generator().manual_seed(0)
        host_pages = [7, 2]
        device_pages = [1, 4]
        host = torch.cat(
            [torch.arange(p * WIDENED_PAGE, (p + 1) * WIDENED_PAGE) for p in host_pages]
        )
        device = torch.cat(
            [
                torch.arange(p * WIDENED_PAGE, (p + 1) * WIDENED_PAGE)
                for p in device_pages
            ]
        )
        # token identity: position i pairs host[i] <-> device[i]
        perm = torch.randperm(host.numel(), generator=g)
        # sort host as move_indices does, permuting device alongside
        host_sorted, order = host[perm].sort()
        device_matched = device[perm][order]
        for rank in range(DCP_SIZE):
            host_mask = host_sorted % DCP_SIZE == rank
            device_mask = device_matched % DCP_SIZE == rank
            # same positions selected on both sides -> pairing preserved
            torch.testing.assert_close(host_mask, device_mask)
            self.assertEqual(
                maybe_dcp_kernel_indices(host_sorted, DCP_SIZE, rank).numel(),
                host.numel() // DCP_SIZE,
            )


class TestHostPoolSizingUnderDcp(CustomTestCase):
    def test_logical_and_physical_sizing(self):
        pool = _make_host_pool(dcp_rank=3)
        # kernel-facing page is physical
        self.assertEqual(pool.page_size, PHYSICAL_PAGE)
        self.assertEqual(pool.logical_page_size, WIDENED_PAGE)
        # physical rows = ratio * device physical size, page aligned
        self.assertEqual(pool.size, pool.page_num * PHYSICAL_PAGE)
        self.assertEqual(pool.logical_size, pool.size * DCP_SIZE)
        # buffers materialize physical rows only
        self.assertEqual(pool.kv_buffer.shape[1], pool.size)
        # allocator surface is logical
        self.assertEqual(pool.free_slots.numel(), pool.logical_size)
        self.assertEqual(pool.mem_state.numel(), pool.logical_size)

    def test_alloc_is_widened_page_granular(self):
        pool = _make_host_pool(dcp_rank=0)
        out = pool.alloc(WIDENED_PAGE)
        self.assertEqual(out.numel(), WIDENED_PAGE)
        with self.assertRaises(AssertionError):
            pool.alloc(PHYSICAL_PAGE)  # not a multiple of the widened page

    def test_non_dcp_pool_unchanged(self):
        pool = MLATokenToKVPoolHost(
            _fake_mla_device_pool(),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PHYSICAL_PAGE,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
        )
        self.assertEqual(pool.page_size, PHYSICAL_PAGE)
        self.assertEqual(pool.logical_size, pool.size)
        self.assertEqual(pool.logical_page_size, PHYSICAL_PAGE)


class TestTransferEntryPointsTranslate(CustomTestCase):
    def _run_backup(self, pool, host_indices, device_indices):
        device_pool = SimpleNamespace(
            data_ptrs=torch.zeros(2, dtype=torch.uint64),
            kv_buffer=[torch.zeros(1)] * 2,
        )
        # create=True: mla.py imports the kernel only under `if _is_cuda or
        # _is_hip`, so the name is absent on the CPU runner this test targets.
        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mla.transfer_kv_all_layer_mla",
            create=True,
        ) as kernel:
            pool.can_use_jit = False
            pool.can_use_write_back_jit = False
            with mock.patch.object(
                MLATokenToKVPoolHost, "_is_device_layer_sharded", return_value=False
            ):
                pool.backup_from_device_all_layer(
                    device_pool, host_indices, device_indices, io_backend="kernel"
                )
            return kernel.call_args.kwargs

    def test_backup_receives_physical_rows(self):
        pool = _make_host_pool(dcp_rank=5)
        logical = torch.arange(2 * WIDENED_PAGE)
        kwargs = self._run_backup(pool, logical, logical.clone())
        expected = torch.arange(2 * PHYSICAL_PAGE)
        torch.testing.assert_close(kwargs["src_indices"], expected)
        torch.testing.assert_close(kwargs["dst_indices"], expected)


class TestDcpStoragePages(CustomTestCase):
    def test_logical_page_roundtrip_and_zero_copy_pointers(self):
        for dcp in (1, 2, 8):
            for layout in ("layer_first", "page_first", "page_first_direct"):
                for dtype in (torch.float16, torch.float8_e4m3fn):
                    with self.subTest(dcp=dcp, layout=layout, dtype=dtype):
                        pool = _make_host_pool(
                            0, dcp_size=dcp, layout=layout, dtype=dtype
                        )
                        pool.kv_buffer.zero_()
                        page = pool.logical_page_size
                        payload = pool.get_dummy_flat_data_page().fill_(3)
                        pool.set_from_flat_data_page(page, payload)
                        self.assertTrue(
                            torch.equal(
                                pool.get_data_page(page).view(torch.uint8),
                                payload.view(torch.uint8),
                            )
                        )
                        self.assertEqual(
                            pool.get_data_page(0).view(torch.uint8).count_nonzero(), 0
                        )
                        pointers, sizes = pool.get_page_buffer_meta(
                            torch.arange(page, 2 * page)
                        )
                        self.assertEqual(
                            sum(sizes), payload.numel() * payload.element_size()
                        )
                        stride = pool.kv_cache_dim * dtype.itemsize
                        if layout == "layer_first":
                            expected = [
                                pool.kv_buffer.data_ptr()
                                + (PHYSICAL_PAGE + layer * pool.size) * stride
                                for layer in range(pool.layer_num)
                            ]
                        else:
                            expected = [
                                pool.kv_buffer.data_ptr()
                                + PHYSICAL_PAGE * pool.layer_num * stride
                            ]
                        self.assertEqual(pointers, expected)

    def test_invalid_page_boundaries_and_split_layout(self):
        pool = _make_host_pool(0)
        for index in (-1, 1, PHYSICAL_PAGE):
            with self.subTest(index=index), self.assertRaises(ValueError):
                pool.get_data_page(index)
        with self.assertRaises(IndexError):
            pool.get_data_page(pool.logical_size)
        pool.layout = "page_first_kv_split"
        with self.assertRaises(NotImplementedError):
            pool.get_page_buffer_meta(torch.arange(pool.logical_page_size))


def _storage_config(rank=0, tp=8, dcp=8, **overrides):
    fields = dict(
        tp_rank=rank,
        tp_size=tp,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="kimi-test",
        extra_config={},
        dcp_size=dcp,
        dcp_rank=rank % dcp,
        logical_page_size=PHYSICAL_PAGE * dcp,
    )
    fields.update(overrides)
    return HiCacheStorageConfig(**fields)


class TestDcpMooncakeAttachment(CustomTestCase):
    def attach(
        self,
        pool,
        *,
        mode="cache",
        backend="mooncake",
        entries=None,
        spec="DSPARK",
        rank=0,
        tp=8,
    ):
        controller = HiCacheController.__new__(HiCacheController)
        controller.storage_host_pool = pool
        controller.mem_pool_host = (
            pool
            if entries is None
            else SimpleNamespace(entries=entries, logical_size=pool.logical_size)
        )
        controller.page_size = pool.logical_page_size
        controller.enable_storage = False
        controller.host_memory_mode = mode
        controller.storage_stop_event = threading.Event()
        controller._stop_storage_threads = mock.Mock()
        controller._start_storage_threads = mock.Mock()
        controller._create_sync_groups = mock.Mock(return_value=[])
        controller._generate_storage_config = mock.Mock(
            return_value=_storage_config(rank, tp, pool.dcp_size)
        )
        with (
            mock.patch(
                "sglang.srt.managers.cache_controller.get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=pool.dcp_size),
            ),
            mock.patch(
                "sglang.srt.managers.cache_controller.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=spec),
            ),
            mock.patch(
                "sglang.srt.mem_cache.storage.StorageBackendFactory.create_backend"
            ),
        ):
            controller.attach_storage_backend(backend)
        return controller

    def test_prefetch_budget_and_writer_ownership(self):
        for tp, dcp in ((4, 1), (4, 2), (8, 8)):
            for rank in range(tp):
                with self.subTest(tp=tp, dcp=dcp, rank=rank):
                    pool = _make_host_pool(rank % dcp, dcp_size=dcp)
                    controller = self.attach(pool, rank=rank, tp=tp)
                    self.assertEqual(controller.backup_skip, rank >= dcp)
                    self.assertEqual(
                        controller.prefetch_capacity_limit, pool.logical_size // 2
                    )
                    controller.prefetch_tokens_occupied = pool.logical_size // 2 - 1
                    self.assertFalse(controller.prefetch_rate_limited())
                    controller.prefetch_tokens_occupied += 1
                    self.assertTrue(controller.prefetch_rate_limited())

    def test_rejects_unsupported_runtime_attachment(self):
        pool = _make_host_pool(0)
        for kwargs in ({"mode": "buffer_only"}, {"backend": "file"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(NotImplementedError):
                self.attach(pool, **kwargs)
        with self.assertRaises(NotImplementedError):
            self.attach(
                pool,
                entries=[SimpleNamespace(name=PoolName.INDEXER, host_pool=object())],
            )
        pool.layout = "page_first_kv_split"
        with self.assertRaises(NotImplementedError):
            self.attach(pool)

    def test_existing_dspark_sidecar_can_attach(self):
        pool = _make_host_pool(0)
        draft = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        draft.page_size = pool.logical_page_size
        entries = [
            SimpleNamespace(name=PoolName.KV, host_pool=pool),
            SimpleNamespace(name=PoolName.DRAFT, host_pool=draft),
        ]
        self.assertTrue(self.attach(pool, entries=entries).enable_storage)
        with self.assertRaises(NotImplementedError):
            self.attach(pool, entries=entries, spec="DFLASH")

    def test_invalid_storage_identity(self):
        for fields in (
            {"dcp_size": 0},
            {"dcp_rank": 8},
            {"tp_rank": 1},
            {"tp_size": 7},
            {"is_mla_model": False},
            {"logical_page_size": None},
            {"logical_page_size": 511},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                _storage_config(**fields)

    def test_startup_guard_accepts_only_supported_l3_mode(self):
        base = dict(
            dcp_size=8,
            enable_hierarchical_cache=True,
            hicache_storage_backend="mooncake",
            hicache_host_memory_mode="cache",
            speculative_algorithm="DSPARK",
            enable_lmcache=False,
            enable_hisparse=False,
        )
        for changes, allowed in (
            ({}, True),
            ({"hicache_storage_backend": None}, True),
            ({"hicache_storage_backend": "file"}, False),
            ({"hicache_host_memory_mode": "buffer_only"}, False),
            ({"speculative_algorithm": "DFLASH"}, False),
        ):
            cfg = SimpleNamespace(**(base | changes))
            with (
                self.subTest(changes=changes),
                mock.patch(
                    "sglang.srt.arg_groups.hicache_hook.resolving_view",
                    return_value=cfg,
                ),
                mock.patch(
                    "sglang.srt.arg_groups.hicache_hook.use_mla_backend",
                    return_value=True,
                ),
            ):
                if allowed:
                    resolve_hicache_dcp_compatibility(cfg)
                else:
                    with self.assertRaises(NotImplementedError):
                        resolve_hicache_dcp_compatibility(cfg)


if __name__ == "__main__":
    unittest.main()
