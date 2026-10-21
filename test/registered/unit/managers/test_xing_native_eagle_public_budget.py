"""Actual native 262144/Q2 admission and scratch checks, without 512K profiles."""
from array import array
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch
from pathlib import Path
import pytest
import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs

def manager(enabled=True):
    obj = object.__new__(TokenizerManager)
    obj.context_len, obj.allow_auto_truncate, obj.validate_total_tokens = 262144, False, True
    obj.num_reserved_tokens = 4
    obj.model_config = NS(request_static_yarn_config=None, xing_eagle_context_headroom=8 if enabled else 0)
    obj._validate_joint_schema_request = lambda *_: None
    return obj

def validate(obj, prompt, output):
    req = NS(sampling_params={"max_new_tokens": output})
    execution = NS(features=NS(enable_encoder_swa_bounded_replay=False))
    with patch("sglang.srt.managers.tokenizer_manager.get_exec", return_value=execution):
        obj._validate_one_request(req, list(range(prompt)))
    return req

def test_exact_frozen_request_budget_does_not_count_four_draft_slots_as_input():
    req = validate(manager(), 261888, 256)
    assert req.sampling_params["max_new_tokens"] == 256
    assert validate(manager(), 262143, 1).sampling_params["max_new_tokens"] == 1
    with pytest.raises(ValueError, match="261889 tokens from the input"):
        validate(manager(), 261889, 256)
    with pytest.raises(ValueError):
        validate(manager(enabled=False), 261888, 256)

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

def test_auto_truncate_cannot_shrink_native_public_input_or_output():
    obj = manager()
    obj.allow_auto_truncate = True
    req = NS(sampling_params={"max_new_tokens": 256})
    ids = list(range(261889))
    execution = NS(features=NS(enable_encoder_swa_bounded_replay=False))
    with patch("sglang.srt.managers.tokenizer_manager.get_exec", return_value=execution):
        with pytest.raises(ValueError, match="261889 tokens from the input"):
            obj._validate_one_request(req, ids)
    assert len(ids) == 261889 and req.sampling_params["max_new_tokens"] == 256


def admission(prompt, output, capacity=262336, hard_limit=None):
    allocator = NS(page_size=64)
    allocator.max_new_tokens_for_memory = MethodType(
        BaseTokenToKVPoolAllocator.max_new_tokens_for_memory, allocator)
    scheduler = NS(model_config=NS(context_len=262144, xing_eagle_context_headroom=8),
        max_req_len=262144, max_total_num_tokens=capacity, kv_shard_widening=1,
        max_new_tokens_limit=hard_limit, page_size=64, sliding_window_size=None,
        chunked_prefill_size=8192, token_to_kv_pool_allocator=allocator)
    req = NS(origin_input_ids=array("q", [1]) * prompt,
        sampling_params=SamplingParams(max_new_tokens=output), to_finish=None, rid="native-cpu")
    def reject(message, **kwargs):
        req.to_finish = (message, kwargs)
    req.set_finish_with_abort = reject
    Scheduler.init_req_max_new_tokens(scheduler, req)
    return req


@pytest.mark.parametrize("prompt,output,capacity,accepted", [
    (261888,256,262336,True), (261888,None,262336,True),
    (261889,256,262336,False), (262143,1,262272,True),
    (262143,1,262208,False), (261889,255,262272,True),
])
def test_native_admission_preserves_budget_or_rejects(prompt, output, capacity, accepted):
    req = admission(prompt, output, capacity)
    assert (req.to_finish is None) is accepted
    assert req.sampling_params.max_new_tokens == (262144-prompt if output is None else output)


def test_native_hard_limit_rejects_without_clipping():
    req = admission(16, 257, hard_limit=256)
    assert req.to_finish is not None
    assert req.sampling_params.max_new_tokens == 257


@pytest.mark.parametrize("steps,tokens", [(1,2), (3,4)])
@pytest.mark.parametrize("draft", [False,True])
def test_real_model_config_native_q2_q4_only(tmp_path, steps, tokens, draft):
    fixture = Path(__file__).parents[1] / "models/xing4_configs/bf16.json"
    (tmp_path/"config.json").write_text(fixture.read_text())
    fields=dict(model_path=str(tmp_path), trust_remote_code=False, context_length=262144,
        device="cuda", dtype="bfloat16", kv_cache_dtype="fp8_e4m3", tp_size=1,
        page_size=64, attention_backend="trtllm_mla", speculative_algorithm="EAGLE",
        speculative_num_steps=steps, speculative_eagle_topk=1, speculative_num_draft_tokens=tokens,
        speculative_attention_mode="prefill", chunked_prefill_size=8192)
    model=ModelConfig.from_server_args(ServerArgs(**fields), is_draft_model=draft)
    assert model.context_len == 262144 and model.xing_eagle_context_headroom == 8
    assert model.hf_text_config.rope_scaling["_xing_eagle_rope_headroom"] == 8
    for override in [{"page_size":1}, {"speculative_eagle_topk":2}, {"prefill_attention_backend":"flashmla"}]:
        control=ModelConfig.from_server_args(ServerArgs(**{**fields,**override}))
        assert control.xing_eagle_context_headroom == 0
        assert "_xing_eagle_rope_headroom" not in control.hf_text_config.rope_scaling


def test_native_rope_headroom_preserves_all_public_rows(monkeypatch):
    monkeypatch.setattr("sglang.srt.layers.rotary_embedding.rope_variant.get_device", lambda: "cpu")
    monkeypatch.setattr("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
    params=dict(rope_type="deepseek_yarn",factor=64,original_max_position_embeddings=4096,
        beta_fast=32,beta_slow=1,mscale=1,mscale_all_dim=1)
    kwargs=dict(head_size=64,rotary_dim=64,max_position=262144,base=10000,
        is_neox_style=False,dtype=torch.float32)
    native=get_rope(**kwargs,rope_scaling=params)
    extended=get_rope(**kwargs,rope_scaling={**params,"_xing_eagle_rope_headroom":8})
    assert extended.cos_sin_cache.shape[0] == 262152
    torch.testing.assert_close(extended.cos_sin_cache[:262144],native.cos_sin_cache,rtol=0,atol=0)


def test_speculative_finish_caps_reasoning_to_emitted_tokens():
    req=Req("reasoning-cap", "", array("q",[1]), SamplingParams(max_new_tokens=2))
    req.sampling_params.normalize(None)
    req.output_ids=array("q",[19]*4)
    req.reasoning_tokens=4
    req.update_finish_state(new_accepted_len=4)
    assert req.finished_len == 2 and len(req.output_ids_through_stop) == 2
    assert req.reasoning_tokens == 2 and len(req.output_ids) == 4
