"""Exercise real buffer restore, Mamba preparation and controller allocation."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from test_mamba_live_admission import _pool


def restore_fixture(free_slots, *, lazy=False, extra=True):
    pool = _pool(mamba_size=free_slots, req_rows=4, lazy=lazy)
    pool.enable_mamba_extra_buffer = extra
    full = TokenToKVPoolAllocator(128, torch.bfloat16, "cpu", None, False)
    cache = SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=full,
        supports_swa=lambda: False,
        evict_for_alloc=lambda _: None,
    )
    component = MambaComponent.__new__(MambaComponent)
    component.cache = cache
    cache.components = {ComponentType.MAMBA: component}
    entry = SimpleNamespace(
        device_pool=pool.mamba_allocator,
        device_alloc_fn=None,
        device_free_fn=None,
        device_evict_fn=lambda _: None,
    )
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.device = "cpu"
    controller.mem_pool_device_allocator = full
    controller.mem_pool_host = SimpleNamespace(entry_map={PoolName.MAMBA: entry})
    controller.load_queue = []
    cache.cache_controller = controller
    pipeline = BufferModePipeline.__new__(BufferModePipeline)
    pipeline._cache = cache
    req = SimpleNamespace(rid="fixture", kv=ReqKvInfo())
    node_copy = PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([1]))
    staged = SimpleNamespace(
        aux_xfers=[node_copy],
        matched_len=0,
        host_indices=torch.arange(8),
        request=object(),
        occupied_tokens=8,
        hash_values=[],
    )
    plan = SimpleNamespace(full_tokens=8, device_prefix_len=0)
    return pool, full, controller, pipeline, req, staged, plan


@pytest.mark.parametrize("lazy,extra,slots", [(False, True, 3), (True, True, 2), (False, False, 1)])
def test_buffer_checkpoint_cannot_consume_promised_live_slots(lazy, extra, slots):
    pool, full, controller, pipeline, req, staged, plan = restore_fixture(
        slots, lazy=lazy, extra=extra
    )
    needed = pool.mamba_admission_slots(req.kv)
    assert needed == slots
    state, failure = pipeline._allocate_and_queue_load_back(
        req, staged, plan, -1, list(staged.aux_xfers), mamba_slots_to_reserve=needed
    )
    assert state is None and failure == "full_or_aux"
    assert not controller.load_queue
    assert pool.mamba_allocator.available_size() == slots
    assert full.available_size() == 128
    assert not req.kv.holds_mamba and req.kv.req_pool_idx is None
    assert staged.aux_xfers[0].device_indices is None
    assert req.kv.mamba_ping_pong_track_buffer is None
    assert pool.alloc([req]) is not None
    assert pool.mamba_allocator.available_size() == 0
    pool.free_mamba_cache(req)
    pool.free(req)
    assert sorted(pool.mamba_allocator.free_slots.tolist()) == list(range(1, slots + 1))


@pytest.mark.parametrize("lazy,extra,slots", [(False, True, 4), (True, True, 3), (False, False, 2)])
def test_buffer_restore_keeps_distinct_checkpoint_and_live_slots(lazy, extra, slots):
    pool, full, controller, pipeline, req, staged, plan = restore_fixture(
        slots, lazy=lazy, extra=extra
    )
    needed = pool.mamba_admission_slots(req.kv)
    state, failure = pipeline._allocate_and_queue_load_back(
        req, staged, plan, -1, list(staged.aux_xfers), mamba_slots_to_reserve=needed
    )
    assert state is not None and failure == ""
    assert len(controller.load_queue) == 1
    assert pool.mamba_allocator.available_size() == needed - 1
    assert pool.mamba_admission_slots(req.kv) == needed - 1
    assert pool.alloc([req]) is not None
    checkpoint = state.load_transfers[0].device_indices.tolist()
    live = [req.kv.mamba_pool_idx.item()]
    if extra:
        live += [x for x in req.kv.mamba_ping_pong_track_buffer.tolist() if x >= 0]
    assert set(checkpoint).isdisjoint(live)
    assert len(set(checkpoint + live)) == slots
    torch.testing.assert_close(
        state.load_transfers[1].device_indices, req.kv.mamba_pool_idx.unsqueeze(0)
    )
    assert pool.mamba_allocator.available_size() == 0
    assert full.available_size() == 120
    # No transport thread was started; release each queued fixture owner once.
    controller.load_queue.clear()
    full.free(state.device_indices)
    pool.mamba_allocator.free(state.load_transfers[0].device_indices)
    pool.free_mamba_cache(req)
    pool.free(req)
    assert sorted(pool.mamba_allocator.free_slots.tolist()) == list(range(1, slots + 1))
    assert full.available_size() == 128


def test_buffer_batch_reservation_failure_rolls_back_prepared_main():
    pool, full, controller, pipeline, req, staged, plan = restore_fixture(3)
    state, failure = pipeline._allocate_and_queue_load_back(
        req, staged, plan, -1, list(staged.aux_xfers), mamba_slots_to_reserve=6
    )
    assert state is None and failure == "mamba"
    assert not controller.load_queue and not req.kv.holds_mamba
    assert pool.mamba_allocator.available_size() == 3
    assert sorted(pool.mamba_allocator.free_slots.tolist()) == [1, 2, 3]
    assert pool.available_size() == 4
    assert full.available_size() == 128


def test_buffer_full_allocation_failure_rolls_back_every_temporary_slot():
    pool, full, controller, pipeline, req, staged, plan = restore_fixture(4)
    held_full = full.alloc(128)
    assert held_full is not None
    pipeline._ensure_full_capacity = lambda _: True  # Race after admission.
    state, failure = pipeline._allocate_and_queue_load_back(
        req, staged, plan, -1, list(staged.aux_xfers), mamba_slots_to_reserve=3
    )
    assert state is None and failure == "full_or_aux"
    assert pool.mamba_allocator.available_size() == 4
    assert sorted(pool.mamba_allocator.free_slots.tolist()) == [1, 2, 3, 4]
    assert full.available_size() == 0
    assert staged.aux_xfers[0].device_indices is None
    assert not controller.load_queue and not req.kv.holds_mamba
    full.free(held_full)
    assert full.available_size() == 128
