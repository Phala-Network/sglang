"""Execute native Xing EAGLE public-budget and scratch paths on CPU."""

from array import array
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest
import torch

from test_xing_request_static_yarn_lifecycle import intake
from test_request_static_yarn_admission_reject import handle
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_reserve_per_decode,
    get_req_to_token_extra_context_len,
    page_aligned_decode_alloc_lens,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_utils import draft_kv_indices_used_len


def manager(enabled=True):
    obj = object.__new__(TokenizerManager)
    obj.context_len, obj.allow_auto_truncate, obj.validate_total_tokens = 262144, False, True
    obj.num_reserved_tokens = 4
    obj.model_config = NS(request_static_yarn_config=None, xing_eagle_context_headroom=8 if enabled else 0)
    return obj


def validate(obj, prompt, output):
    req = NS(sampling_params={"max_new_tokens": output})
    execution = NS(features=NS(enable_encoder_swa_bounded_replay=False))
    with patch("sglang.srt.managers.tokenizer_manager.get_exec", return_value=execution):
        obj._validate_token_budget(req, list(range(prompt)))
    return req


def native_intake(prompt, output, capacity=262336):
    scheduler, recv, req = intake(prompt, output, pool_capacity=capacity)
    scheduler.model_config.context_len = 262144
    scheduler.model_config.request_static_yarn_config = None
    scheduler.model_config.xing_eagle_context_headroom = 8
    scheduler.max_req_len = scheduler.max_req_input_len = 262144
    return scheduler, recv, req


def test_exact_frozen_request_budget_does_not_count_four_draft_slots_as_input():
    req = validate(manager(), 261888, 256)
    assert req.sampling_params["max_new_tokens"] == 256
    assert validate(manager(), 262143, 1).sampling_params["max_new_tokens"] == 1
    with pytest.raises(ValueError, match="261889 tokens from the input"):
        validate(manager(), 261889, 256)
    with pytest.raises(ValueError):
        validate(manager(enabled=False), 261888, 256)


@pytest.mark.parametrize("output", [256, None])
def test_real_scheduler_preserves_exact_public_budget_and_independent_reserve(output):
    scheduler, recv, req = native_intake(261888, output)
    handle(scheduler, recv, req)
    assert not req.finished() and scheduler.waiting_queue == [req]
    assert req.sampling_params.max_new_tokens == 256
    assert req.rope_profile is None


def test_public_overflow_and_insufficient_scratch_reject_without_clipping_then_recover():
    for prompt, output, capacity in [(261889, 256, 262336), (261888, 256, 262208)]:
        scheduler, recv, req = native_intake(prompt, output, capacity)
        handle(scheduler, recv, req)
        assert req.finished() and scheduler.waiting_queue == []
        assert req.sampling_params.max_new_tokens == output
        req.init_next_round_input.assert_not_called()
    scheduler, recv, req = native_intake(16, 4)
    handle(scheduler, recv, req)
    assert not req.finished() and scheduler.waiting_queue == [req]
    assert req.sampling_params.max_new_tokens == 4


def test_real_worker_advertises_public_context_without_legacy_minus_one_or_five():
    runner = NS(effective_logical_max_total_num_tokens=262336,
        logical_max_total_num_tokens=262336, token_to_kv_pool_allocator=object(),
        max_running_requests=512, forward_stream=None,
        req_to_token_pool=NS(size=512, max_context_len=262215), token_to_kv_pool=NS(size=262336))
    worker = NS(model_config=NS(context_len=262144, request_static_yarn_config=None,
        xing_eagle_context_headroom=8), model_runner=runner, dllm_algorithm=None, random_seed=1, device="cuda")
    with patch("sglang.srt.managers.tp_worker.page_interleave_shard_size", return_value=1), patch(
        "sglang.srt.managers.tp_worker.get_schedule", return_value=NS(max_prefill_tokens=8192, max_queued_requests=None)):
        assert TpModelWorker.get_worker_info(worker)[4:6] == (262144, 262144)
        worker.model_config.xing_eagle_context_headroom = 0
        assert TpModelWorker.get_worker_info(worker)[4:6] == (262143, 262138)


@pytest.mark.parametrize("draft", [False, True])
def test_real_server_args_model_config_enables_only_fixed_native_recipe(tmp_path, draft):
    fixture = Path(__file__).parents[1] / "models/xing4_configs/bf16.json"
    (tmp_path / "config.json").write_text(fixture.read_text())
    args = ServerArgs(model_path=str(tmp_path), trust_remote_code=False,
        context_length=262144, device="cuda", dtype="bfloat16", kv_cache_dtype="fp8_e4m3",
        tp_size=1, page_size=64, attention_backend="trtllm_mla", speculative_algorithm="EAGLE",
        speculative_num_steps=3, speculative_eagle_topk=1, speculative_num_draft_tokens=4,
        speculative_attention_mode="prefill", chunked_prefill_size=8192, prefill_decode_interval=8)
    model = ModelConfig.from_server_args(args, is_draft_model=draft)
    assert model.xing_eagle_context_headroom == 8
    assert model.context_len == 262144 and model.request_static_yarn_config is None
    assert model.hf_text_config.rope_scaling["_xing_eagle_rope_headroom"] == 8
    assert args.cuda_graph_backend_prefill != "disabled"
    for field in ("prefill_attention_backend", "decode_attention_backend", "speculative_draft_attention_backend"):
        setattr(args, field, "flashmla")
        control = ModelConfig.from_server_args(args)
        assert control.xing_eagle_context_headroom == 0
        assert "_xing_eagle_rope_headroom" not in control.hf_text_config.rope_scaling
        setattr(args, field, None)
    args.page_size = 1
    assert ModelConfig.from_server_args(args).xing_eagle_context_headroom == 0
    args.page_size, args.speculative_eagle_topk = 64, 2
    assert ModelConfig.from_server_args(args).xing_eagle_context_headroom == 0
    args.speculative_eagle_topk, args.speculative_adaptive = 1, True
    assert ModelConfig.from_server_args(args).xing_eagle_context_headroom == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_rope_preserves_native_table_and_covers_last_speculative_step(dtype):
    params = dict(rope_type="deepseek_yarn", factor=64, original_max_position_embeddings=4096,
        beta_fast=32, beta_slow=1, mscale=1, mscale_all_dim=1)
    kwargs = dict(head_size=64, rotary_dim=64, max_position=262144, base=10000,
        is_neox_style=False, dtype=dtype)
    native = get_rope(**kwargs, rope_scaling=params)
    extended = get_rope(**kwargs, rope_scaling={**params, "_xing_eagle_rope_headroom": 8})
    assert extended.cos_sin_cache.shape[0] == 262152
    torch.testing.assert_close(extended.cos_sin_cache[:262144], native.cos_sin_cache, rtol=0, atol=0)
    native._ensure_cos_sin_cache_length(262151)
    positions = torch.tensor([262143, 262144, 262146, 262151])
    q, k = torch.randn(4, 2, 64).to(dtype), torch.randn(4, 1, 64).to(dtype)
    actual, reference = extended.forward_native(positions, q, k), native.forward_native(positions, q, k)
    for result, expected in zip(actual, reference):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_actual_allocator_rounding_and_final_accepted_run_fit_reserved_row_and_emitted_cap():
    # Real page rounding at the final public context step: double-buffer8 is
    # scratch, and row headroom8+page63 can hold the rounded allocation.
    with patch("sglang.srt.mem_cache.allocation_sizing.get_spec", return_value=NS(
        speculative_algorithm="EAGLE", speculative_num_steps=3, speculative_eagle_topk=1,
    )), patch("sglang.srt.mem_cache.allocation_sizing.max_speculative_num_draft_tokens", return_value=4), patch(
        "sglang.srt.mem_cache.allocation_sizing.get_schedule", return_value=NS(page_size=64)
    ), patch("sglang.srt.mem_cache.allocation_sizing.get_parallel", return_value=NS(attn_dcp_size=1)):
        reserve = get_alloc_reserve_per_decode()
        row_headroom = get_req_to_token_extra_context_len()
    assert reserve == 8 and row_headroom == 71
    req = NS(kv=NS(kv_allocated_len=262144, kv_committed_len=262143))
    cur, nxt, needed = page_aligned_decode_alloc_lens([req], reserve=reserve, page_size=64)
    assert cur == [262144] and nxt == [262208] and needed == 64
    assert nxt[0] <= 262144 + row_headroom
    scheduler, recv, actual_req = native_intake(261888, 256)
    handle(scheduler, recv, actual_req)
    actual_req.output_ids = array("q", [19] * 259)
    actual_req.reasoning_tokens = 259
    actual_req.update_finish_state(new_accepted_len=4)
    assert actual_req.finished_len == 256 and len(actual_req.output_ids_through_stop) == 256
    assert actual_req.reasoning_tokens == 256
    assert len(actual_req.output_ids) == 259  # owned scratch is retained for normal release


@pytest.mark.parametrize("headroom,dynamic", [(0, False), (8, False), (8, True)])
def test_real_backend_construction_and_graph_metadata_cover_native_tail(headroom, dynamic):
    from sglang.srt.layers.attention import flashinfer_mla_backend as fi
    from sglang.srt.layers.attention import trtllm_mla_backend as trt
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    context = 524288 if dynamic else 262144
    config = NS(context_len=context, xing_eagle_context_headroom=0 if dynamic else headroom,
        request_static_yarn_config=NS(lookahead=headroom) if dynamic else None,
        num_attention_heads=64, get_num_kv_heads=lambda size: 1, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, scaling=1.0)
    runner = NS(model_config=config, device="cpu", page_size=64,
        req_to_token_pool=NS(size=2, req_to_token=torch.zeros((2, context + 71), dtype=torch.int32)),
        token_to_kv_pool=object(), kv_index_translator=NS(is_translating=False),
        dtype=torch.bfloat16, kv_cache_dtype=torch.float8_e4m3fn,
        max_running_requests=2, is_draft_worker=False)
    parallel = NS(attn_tp_size=1, attn_dcp_size=1, dcp_enabled=False)
    schedule = NS(disable_chunked_prefix_cache=True, page_size=64)
    zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        # Exercise real Python buffer construction with CPU tensors. CUDA-only
        # wrapper/kernel entry points are isolated below; no inference is run.
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        kwargs.pop("pin_memory", None)
        return zeros(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch, "zeros", side_effect=cpu_zeros))
        for module in (fi, trt):
            stack.enter_context(patch.object(module, "get_parallel", return_value=parallel))
            stack.enter_context(patch.object(module, "get_schedule", return_value=schedule))
            stack.enter_context(patch.object(module, "get_buffer", return_value=torch.empty(1)))
        stack.enter_context(patch.object(fi, "get_cuda_graph_max_batch_size", side_effect=lambda size: size))
        stack.enter_context(patch.object(fi, "get_disagg", return_value=NS(disaggregation_mode="null")))
        stack.enter_context(patch.object(fi, "get_platform", return_value=NS(is_sm100=False)))
        stack.enter_context(patch.object(fi, "BatchMLAPagedAttentionWrapper", create=True, return_value=MagicMock()))
        stack.enter_context(patch.object(fi, "BatchPrefillWithRaggedKVCacheWrapper", create=True, return_value=MagicMock()))
        stack.enter_context(patch.object(trt, "get_spec", return_value=NS(speculative_num_draft_tokens=4)))
        stack.enter_context(patch.object(trt, "get_kv_shard_pool", return_value=None))
        stack.enter_context(patch.object(trt, "make_persistent_multi_ctas_kv_counter_buffer", return_value=torch.empty(1)))
        stack.enter_context(patch.object(trt, "can_use_set_mla_kv_concat_q_fp8", return_value=False))
        stack.enter_context(patch.object(trt, "maybe_create_verify_mask", return_value=None))
        backend = trt.TRTLLMMLABackend(runner)
        backend.init_cuda_graph_state(max_bs=2, max_num_tokens=8)
        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
            backend._init_cuda_graph_metadata(2, 8, mode, torch.tensor([context - 1, 16]), "cpu")
            metadata = backend.forward_decode_metadata
            assert metadata.max_seq_len_k == context + headroom
            assert metadata.block_kv_indices.shape[1] * 64 >= context + headroom
        draft = fi.FlashInferMLAMultiStepDraftBackend(runner, topk=1, speculative_num_steps=3)
        draft.init_cuda_graph_state(max_bs=2, max_num_tokens=8)
    assert config.context_len == context
    assert backend.max_context_len == draft.max_context_len == context + headroom
    assert draft.cuda_graph_kv_indices.shape == (3, 2 * (context + headroom))
    if headroom:
        # Two last-step requests can still execute the entire draft/verify run.
        required = draft_kv_indices_used_len(2 * (context - 1), topk=1, bs=2, num_steps=3)
        assert required <= draft.cuda_graph_kv_indices.shape[1]
        assert backend.forward_decode_metadata.block_kv_indices.shape[1] >= context // 64 + 1


def test_auto_truncate_cannot_shrink_native_public_input_or_output():
    obj = manager()
    obj.allow_auto_truncate = True
    req = NS(sampling_params={"max_new_tokens": 256})
    ids = list(range(261889))
    execution = NS(features=NS(enable_encoder_swa_bounded_replay=False))
    with patch("sglang.srt.managers.tokenizer_manager.get_exec", return_value=execution):
        with pytest.raises(ValueError, match="261889 tokens from the input"):
            obj._validate_token_budget(req, ids)
    assert len(ids) == 261889 and req.sampling_params["max_new_tokens"] == 256


@pytest.mark.parametrize("prompt,output,capacity,accepted", [
    (262143, 1, 262272, True),
    (262143, 1, 262208, False),
    (261889, 255, 262272, True),
])
def test_allocator_spare_page_covers_tail_without_duplicate_scratch_reservation(
    prompt, output, capacity, accepted
):
    scheduler, recv, req = native_intake(prompt, output, capacity)
    handle(scheduler, recv, req)
    assert req.sampling_params.max_new_tokens == output
    assert (scheduler.waiting_queue == [req]) is accepted
    assert req.finished() is not accepted
    # Public C + eight lookahead slots fits the existing next physical page.
    tail = NS(kv=NS(kv_allocated_len=262144, kv_committed_len=262143))
    _, lengths, _ = page_aligned_decode_alloc_lens([tail], reserve=8, page_size=64)
    if accepted:
        assert lengths[0] <= capacity
