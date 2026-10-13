"""Actual FULL/MAMBA cache and deferred COW; no model-inference claim."""

from array import array
from types import SimpleNamespace

import pytest
import torch
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.configs.request_static_yarn import (
    RequestStaticYarnConfig,
    admit_request_static_yarn,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams


def state_tensors(pool, logical_indices):
    physical = pool.translate_mamba_indices(logical_indices)
    state = pool.mamba_pool.mamba_cache
    return [state.temporal[:, physical].clone()] + [
        conv[:, physical].clone() for conv in state.conv
    ]


def write_state(pool, logical_indices, marker):
    physical = pool.translate_mamba_indices(logical_indices)
    state = pool.mamba_pool.mamba_cache
    # Advanced indexing returns a copy, so fill_ on the indexed result would
    # leave the actual pool unchanged. Assign into each pool tensor directly.
    state.temporal[:, physical] = marker
    for index, conv in enumerate(state.conv):
        conv[:, physical] = marker + index + 1


def cache_key(req, tokens):
    return RadixKey(
        array("q", tokens), extra_key=req.extra_key, cache_salt=req.cache_salt
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA device required")
def test_real_full_mamba_namespaces_deferred_cow_and_suffix_reuse():
    cache, allocator, pool = build_fixture(
        CacheConfig(
            components=(ComponentType.FULL, ComponentType.MAMBA),
            num_layers=4,
            full_attention_layer_ids=(3,),
            enable_mamba_extra_buffer=True,
            kv_size=512,
            max_num_reqs=16,
            mamba_cache_size=64,
        ),
        tree_core_backend="python",
    )
    config = RequestStaticYarnConfig.from_hf_config(
        SimpleNamespace(
            rope_parameters={
                "rope_type": "yarn",
                "factor": 4.0,
                "request_static_factors": [1, 2, 4],
                "original_max_position_embeddings": 256,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            }
        )
    )
    prefix = list(range(64))
    suffix = list(range(1000, 1064))

    def make_req(tokens, factor, *, salt="tenant", namespace="adapter", alloc=True):
        req = Req(
            rid=f"{factor}-{salt}-{namespace}-{len(tokens)}",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(
                temperature=0,
                max_new_tokens=1,
                custom_params={"yarn_factor": factor},
            ),
            extra_key=namespace,
            cache_salt=salt,
        )
        admit_request_static_yarn(req, config, 1024)
        if alloc:
            assert pool.alloc([req]) is not None
        return req

    full_pool = allocator.get_kvcache().full_kv_pool
    key_buffer = full_pool.get_key_buffer(full_pool.start_layer)
    value_buffer = full_pool.get_value_buffer(full_pool.start_layer)
    entries = []
    for factor in (1, 2, 4):
        req = make_req(prefix, factor)
        # A prefix already inserted under another factor must be a cold miss.
        miss = cache.match_prefix(MatchPrefixParams(key=cache_key(req, prefix)))
        assert len(miss.device_indices) == 0
        indices = allocator.alloc(len(prefix))
        assert indices is not None
        key_buffer[indices] = factor
        value_buffer[indices] = factor + 8
        slot = req.kv.mamba_pool_idx.unsqueeze(0)
        write_state(pool, slot, factor * 10)
        inserted = cache.insert(
            InsertParams(key=cache_key(req, prefix), value=indices, mamba_value=slot)
        )
        entries.append((factor, indices, slot, inserted.last_device_node))

    assert len({entry[3] for entry in entries}) == 3
    continuations = []
    for factor, indices, src_slot, node in entries:
        req = make_req(prefix + suffix, factor)
        match = cache.match_prefix(
            MatchPrefixParams(
                key=cache_key(req, prefix + suffix), cow_mamba=True, req=req
            )
        )
        assert match.best_match_node == node
        assert torch.equal(match.device_indices, indices)
        assert torch.all(key_buffer[match.device_indices] == factor)
        assert torch.all(value_buffer[match.device_indices] == factor + 8)
        assert torch.equal(req.kv.mamba_cow_src_index, src_slot)
        dst_slot = req.kv.mamba_pool_idx.unsqueeze(0)
        assert not torch.equal(src_slot, dst_slot)
        # Make the deferred copy observable even if a fresh slot starts at zero.
        write_state(pool, dst_slot, -100)
        continuations.append((req, match, src_slot, dst_slot))

    cold_req = make_req(prefix, 1, salt="other")
    write_state(pool, cold_req.kv.mamba_pool_idx.unsqueeze(0), 99)
    cold_match = cache.match_prefix(
        MatchPrefixParams(key=cache_key(cold_req, prefix), cow_mamba=True, req=cold_req)
    )
    assert len(cold_match.device_indices) == 0
    assert cold_req.kv.mamba_cow_src_index is None
    other_adapter = make_req(prefix, 1, namespace="other", alloc=False)
    assert (
        len(
            cache.match_prefix(
                MatchPrefixParams(key=cache_key(other_adapter, prefix))
            ).device_indices
        )
        == 0
    )

    batch = SimpleNamespace()
    ScheduleBatch._collect_deferred_mamba_cow_and_clear(
        batch, [entry[0] for entry in continuations] + [cold_req]
    )
    assert batch.mamba_cow_src_indices.numel() == 3
    assert batch.mamba_clear_indices.numel() == 1
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        mamba_cow_src_indices=batch.mamba_cow_src_indices,
        mamba_cow_dst_indices=batch.mamba_cow_dst_indices,
        mamba_clear_indices=batch.mamba_clear_indices,
    )
    runner = SimpleNamespace(req_to_token_pool=pool, is_draft_worker=False)
    ModelRunner._maybe_execute_deferred_mamba_cow_and_clear(runner, forward_batch)
    assert forward_batch.mamba_cow_src_indices is None
    assert forward_batch.mamba_cow_dst_indices is None
    assert forward_batch.mamba_clear_indices is None
    for state in state_tensors(pool, cold_req.kv.mamba_pool_idx.unsqueeze(0)):
        assert torch.count_nonzero(state) == 0

    for req, match, src_slot, dst_slot in continuations:
        assert req.kv.mamba_cow_src_index is None
        assert not req.kv.mamba_needs_clear
        source_state = state_tensors(pool, src_slot)
        for actual, expected in zip(state_tensors(pool, dst_slot), source_state):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        write_state(pool, dst_slot, req.rope_profile.factor * 20)
        for actual, expected in zip(state_tensors(pool, src_slot), source_state):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        suffix_indices = allocator.alloc(len(suffix))
        assert suffix_indices is not None
        key_buffer[suffix_indices] = req.rope_profile.factor + 16
        value_buffer[suffix_indices] = req.rope_profile.factor + 24
        extended_indices = torch.cat((match.device_indices, suffix_indices))
        cache.insert(
            InsertParams(
                key=cache_key(req, prefix + suffix),
                value=extended_indices,
                mamba_value=dst_slot,
            )
        )
        reused = cache.match_prefix(
            MatchPrefixParams(key=cache_key(req, prefix + suffix))
        )
        assert torch.equal(reused.device_indices, extended_indices)
        stored_slot = cache.tree_core.get_component_device_value(
            reused.best_match_node, ComponentType.MAMBA
        )
        assert torch.equal(stored_slot, dst_slot)
        assert torch.all(
            key_buffer[reused.device_indices[:64]] == req.rope_profile.factor
        )
        assert torch.all(
            key_buffer[reused.device_indices[64:]] == req.rope_profile.factor + 16
        )
    cache.sanity_check()
