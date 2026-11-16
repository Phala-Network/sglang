"""Persistent load-back runs the actual cache/component/controller allocation path."""

from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from test_mamba_load_back_admission_cpu import restore_fixture


def persistent_fixture(slots):
    pool, full, controller, pipeline, req, staged, _ = restore_fixture(slots)
    cache = pipeline._cache
    component = next(iter(cache.components.values()))
    checkpoint = staged.aux_xfers[0]
    result = NS(delta=0, to_dec_params=lambda: None)
    cache.inc_host_lock_ref = Mock(return_value=result)
    cache.inc_lock_ref = Mock(return_value=result)
    cache.dec_host_lock_ref = Mock()
    cache.dec_lock_ref = Mock()
    cache._components_tuple = (component,)
    cache.load_back_threshold = 1
    cache.tree_core = NS(
        component_has_host_value_only=lambda *_: True,
        build_load_back_spec=lambda *_args, **_kw: (
            PoolTransfer(name=PoolName.KV, host_indices=torch.arange(8)),
            {component.component_type: [checkpoint]},
        ),
        commit_load_back=Mock(return_value=[]),
    )
    component.tree_core = cache.tree_core
    cache._component_available_size = lambda _: full.available_size()
    cache._build_sidecar_transfers = lambda *_: []
    cache._apply_cache_actions = Mock()
    cache.ongoing_load_back = {}
    cache._load_back_transfers = MethodType(UnifiedRadixCache._load_back_transfers, cache)
    return cache, pool, full, controller, req, checkpoint


def test_persistent_checkpoint_cannot_consume_full_batch_live_reservation():
    cache, pool, full, controller, req, checkpoint = persistent_fixture(6)
    assert not UnifiedRadixCache.load_back(cache, 1, req=req, mamba_slots_to_reserve=6)
    assert not req.kv.holds_mamba
    assert not controller.load_queue
    assert pool.mamba_allocator.available_size() == 6
    assert full.available_size() == 128
    assert checkpoint.device_indices is None
    cache.dec_host_lock_ref.assert_called_once()
    cache.dec_lock_ref.assert_called_once()


def test_persistent_prepared_main_counts_toward_live_reservation():
    cache, pool, full, controller, req, checkpoint = persistent_fixture(4)
    assert UnifiedRadixCache.load_back(cache, 1, req=req, mamba_slots_to_reserve=3)
    assert req.kv.holds_mamba
    assert len(controller.load_queue) == 1
    assert pool.mamba_allocator.available_size() == 2
    assert int(req.kv.mamba_pool_idx) != int(checkpoint.device_indices.item())
    assert pool.alloc([req]) is not None
    assert pool.mamba_allocator.available_size() == 0
    controller.load_queue.clear()
    pool.mamba_allocator.free(checkpoint.device_indices)
    pool.free_mamba_cache(req)
    pool.free(req)
    # Fixture tree commit owns the full destination until this explicit cleanup.
    device_indices = cache.tree_core.commit_load_back.call_args.args[1]
    full.free(device_indices)
    assert pool.mamba_allocator.available_size() == 4
    assert full.available_size() == 128


def test_persistent_reservation_failure_preserves_existing_main_owner():
    cache, pool, full, controller, req, _ = persistent_fixture(4)
    original = pool.mamba_allocator.alloc(1)
    req.kv.mamba_pool_idx = original[0]
    assert not UnifiedRadixCache.load_back(cache, 1, req=req, mamba_slots_to_reserve=4)
    assert req.kv.holds_mamba
    assert int(req.kv.mamba_pool_idx) == int(original[0])
    assert pool.mamba_allocator.available_size() == 3
    assert full.available_size() == 128
    assert not controller.load_queue
    pool.mamba_allocator.free(original)
    assert pool.mamba_allocator.available_size() == 4


def test_persistent_full_capacity_failure_rolls_back_new_main_and_reservation():
    cache, pool, full, controller, req, _ = persistent_fixture(4)
    held = full.alloc(128)
    assert not UnifiedRadixCache.load_back(cache, 1, req=req, mamba_slots_to_reserve=3)
    assert not req.kv.holds_mamba
    assert pool.mamba_allocator.available_size() == 4
    assert not controller.load_queue
    assert full.available_size() == 0
    full.free(held)
    assert full.available_size() == 128
