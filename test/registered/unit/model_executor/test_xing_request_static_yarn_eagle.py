"""CPU execution of Xing EAGLE position/config/replay paths; no CUDA claims."""

from array import array
from contextlib import nullcontext
from dataclasses import fields
import json
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.request_static_yarn import RequestStaticYarnConfig, admit_request_static_yarn
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.request_static_yarn import scale_request_static_yarn_query
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleDraftExtendInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

PARAMETERS = dict(rope_type="deepseek_yarn", rope_theta=10000, factor=128,
    request_static_factors=[64, 128], original_max_position_embeddings=4096,
    beta_fast=32, beta_slow=1, mscale=1, mscale_all_dim=1)


def config():
    return RequestStaticYarnConfig.from_hf_config(NS(model_type="xing4_0",
        rope_theta=10000, rope_scaling={**PARAMETERS, "_request_static_yarn_lookahead": 8}))


def runtime_args(**updates):
    return NS(attention_backend="trtllm_mla", device="cuda", dtype="bfloat16",
        kv_cache_dtype="fp8_e4m3", tp_size=1, page_size=64, speculative_algorithm="EAGLE",
        speculative_num_steps=3, speculative_eagle_topk=1,
        speculative_num_draft_tokens=4, speculative_attention_mode="prefill", **updates)


@pytest.mark.parametrize("draft", [False, True])
def test_real_model_config_target_and_embedded_mtp_own_same_profile_layout(tmp_path, draft):
    fixture = Path(__file__).parents[1] / "models/xing4_configs/bf16.json"
    (tmp_path / "config.json").write_text(fixture.read_text())
    model = ModelConfig(str(tmp_path), trust_remote_code=False, context_length=524288,
        model_override_args=json.dumps(dict(max_position_embeddings=524288,
            rope_scaling=PARAMETERS)), is_draft_model=draft, speculative_algorithm="EAGLE")
    cfg = model.request_static_yarn_config
    assert cfg.lookahead == 8
    assert [p[1] for p in cfg.profiles] == [0, 262152]
    assert cfg.select(262144).factor == 64 and cfg.select(262145).factor == 128
    assert model.context_len == 524288 and model.dtype == torch.bfloat16
    assert model.hf_text_config.rope_scaling["_request_static_yarn_lookahead"] == 8
    assert model.hf_config.architectures[0] == ("DeepseekV3ForCausalLMNextN" if draft else "Xing4_0ForCausalLM")


def test_fixed_eagle_runtime_is_supported_and_unknown_layouts_fail_closed():
    config().validate_runtime_support(runtime_args())
    invalid = [dict(tp_size=2), dict(speculative_algorithm="EAGLE3"),
        dict(speculative_eagle_topk=2), dict(speculative_num_steps=4),
        dict(speculative_num_draft_tokens=5), dict(speculative_adaptive=True),
        dict(enable_multi_layer_eagle=True), dict(speculative_use_rejection_sampling=True),
        dict(speculative_attention_mode="decode"), dict(speculative_draft_window_size=32),
        dict(page_size=1), dict(kv_cache_dtype="bfloat16"), dict(quantization="fp8"),
        dict(speculative_draft_attention_backend="flashmla"),
        dict(prefill_attention_backend="flashmla"), dict(decode_attention_backend="flashmla")]
    for changes in invalid:
        args = runtime_args()
        vars(args).update(changes)
        with pytest.raises(ValueError):
            config().validate_runtime_support(args)
    no_headroom = RequestStaticYarnConfig.from_hf_config(NS(model_type="xing4_0", rope_scaling=PARAMETERS))
    with pytest.raises(ValueError, match="lookahead"):
        no_headroom.validate_runtime_support(runtime_args())


@pytest.mark.parametrize("mode,width", [(ForwardMode.DECODE, 1),
    (ForwardMode.TARGET_VERIFY, 4), (ForwardMode.DRAFT_EXTEND_V2, 4), (ForwardMode.EXTEND, 2)])
def test_real_forward_batch_mixed_profiles_all_eagle_phases(mode, width):
    cfg = config()
    profiles = [cfg.select(32, f) for f in (64, 128, 64)]
    fb = object.__new__(ForwardBatch)
    fb.forward_mode, fb.positions = mode, torch.arange(3 * width, dtype=torch.int64) + 19
    fb.out_cache_loc = torch.arange(3 * width) + 100
    fb.spec_info = NS(num_tokens_per_req=width, num_front_tokens=0)
    # Verify intentionally retains unrelated prefill lengths; it must use the
    # uniform speculative width instead. GPU-only extend lengths are tensors.
    lengths = [27, 31, 7] if mode.is_target_verify() else torch.tensor([width] * 3)
    batch = NS(reqs=[NS(rope_profile=p) for p in profiles], extend_lens=lengths)
    runner = NS(device="cpu", model_config=NS(request_static_yarn_config=cfg))
    physical, cache_locs = fb.positions.clone(), fb.out_cache_loc.clone()
    fb._init_request_static_yarn_mla(runner, batch)
    offsets = torch.tensor([p.offset for p in profiles]).repeat_interleave(width)
    assert torch.equal(fb.rope_positions, physical + offsets)
    assert torch.equal(fb.positions, physical) and torch.equal(fb.out_cache_loc, cache_locs)
    fb.positions.add_(7)
    fb.refresh_request_static_yarn_positions()
    assert torch.equal(fb.rope_positions, physical + offsets + 7)


def test_real_rope_lookup_and_q_scale_keep_boundary_lookahead_in_own_profile():
    cfg = config()
    params = {**PARAMETERS, "_request_static_yarn_lookahead": 8}
    combined = get_rope(64, 64, 524288, 10000, False, rope_scaling=params, dtype=torch.float32)
    assert combined.cos_sin_cache.shape[0] == 786448
    assert combined.scaling_factor_to_offset[128] == cfg.profiles[1][1]
    gen = torch.Generator().manual_seed(519)
    q, k = torch.randn(6, 2, 64, generator=gen), torch.randn(6, 1, 64, generator=gen)
    physical = torch.tensor([262143, 262144, 262151, 524287, 524288, 524295])
    offsets = torch.tensor([0] * 3 + [262152] * 3)
    actual = combined.forward_native(physical + offsets, q, k)
    for i, factor in enumerate((64, 128)):
        static_params = {k: v for k, v in PARAMETERS.items() if k != "request_static_factors"}
        static_params["factor"] = factor
        static = get_rope(64, 64, 524288, 10000, False, rope_scaling=static_params, dtype=torch.float32)
        static._ensure_cos_sin_cache_length(int(physical[i * 3 + 2]))
        reference = static.forward_native(physical[i * 3:(i+1)*3], q[i*3:(i+1)*3], k[i*3:(i+1)*3])
        for a, b in zip(actual, reference):
            torch.testing.assert_close(a[i*3:(i+1)*3], b, rtol=0, atol=0)
    module = NS(request_static_yarn_scale_ratio=1.1, request_static_yarn_long_offset=262152)
    scaled = scale_request_static_yarn_query(q, NS(rope_positions=physical + offsets), module)
    assert torch.equal(scaled[:3], q[:3])
    torch.testing.assert_close(scaled[3:], q[3:] * 1.1, rtol=0, atol=0)
    with pytest.raises(ValueError, match="table capacity"):
        combined._ensure_cos_sin_cache_length(786448)


def graph_fixture(phase, monkeypatch):
    if phase == "draft":
        import sglang.srt.speculative.eagle_draft_cuda_graph_runner as module
        cls, width = module.EAGLEDraftCudaGraphRunner, 1
        buffer_cls = module.EagleDraftInputBuffers
    else:
        import sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner as module
        cls, width = module.EAGLEDraftExtendCudaGraphRunner, 4
        buffer_cls = module.EagleDraftExtendInputBuffers
    assert "rope_positions" in {f.name for f in fields(buffer_cls)}
    n, bs = 3 * width, 3
    z = lambda size, dtype=torch.int64: torch.zeros(size, dtype=dtype)
    buffers = NS(input_ids=z(n), req_pool_indices=z(bs), out_cache_loc=z(n * (3 if phase == "draft" else 1)),
        positions=z(n), rope_positions=z(n), mrope_positions=z((3, n)), seq_lens=z(bs), seq_lens_cpu=z(bs),
        extend_seq_lens=z(bs, torch.int32), topk_p=z((bs, 1), torch.float32), topk_index=z((bs, 1)),
        draft_probs=None, hidden_states=z((n if phase == "extend" else bs, 2), torch.float32),
        rids_int=None, bootstrap_room_ids_int=None, dsa_seed_topk=None, dsa_seed_topk_capture=None,
        num_correct_drafts=z(bs, torch.int32), num_accept_tokens=z(bs, torch.int32),
        select_index=z(bs), next_token_logits_buffer=z((bs, 16), torch.float32))
    graph = object.__new__(cls)
    graph.buffers, graph.captured_req_width, graph.speculative_num_steps = buffers, width, 3
    graph.require_mlp_tp_gather = graph.require_attn_tp_gather = graph.require_gathered_buffer = False
    graph.capture_bs, graph.seq_len_fill_value = [3], 1
    graph.extend_seq_lens_cpu = [width] * 3
    graph.deepep_adapter = NS(replay=lambda: None)
    graph.model_runner = NS(model_config=NS(model_is_mrope=False, vocab_size=16),
        spec_algorithm=SpeculativeAlgorithm.from_string("EAGLE"), device_timer=None)
    graph.draft_attn_backend = NS(init_forward_metadata_out_graph=lambda fb: None)
    graph.draft_extend_attn_backend = NS(init_forward_metadata_out_graph=lambda fb: None)
    graph.forward_mode = ForwardMode.DRAFT_EXTEND_V2
    graph.device_module = NS(Event=lambda: NS(record=lambda: None))
    graph._make_graph_key = lambda size: size
    graph._pad_to_bucket = lambda size, buckets: 3
    monkeypatch.setattr(module, "device_timer_ctx", lambda *a: nullcontext())
    if phase == "draft":
        monkeypatch.setattr(module, "get_spec", lambda: NS(speculative_use_rejection_sampling=False))
    def replay(key, fb):
        assert key == 3
        if phase == "draft":
            return tuple(torch.zeros((3, 1), dtype=torch.int64) for _ in range(3)) + (None,)
        return NS(next_token_logits=torch.zeros(3, 16), hidden_states=torch.zeros(3, 2))
    graph._replay_graph = replay
    return graph, width


@pytest.mark.parametrize("phase", ["draft", "extend"])
def test_real_eagle_graph_execute_refreshes_profiles_and_clears_padded_rows(phase, monkeypatch):
    graph, width = graph_fixture(phase, monkeypatch)
    ptr = graph.buffers.rope_positions.data_ptr()
    for factors in ((64, 128), (128,), (128, 64)):
        bs, n = len(factors), len(factors) * width
        physical = torch.arange(n) + 11
        offsets = torch.tensor([config().select(32, f).offset for f in factors]).repeat_interleave(width)
        fb = object.__new__(ForwardBatch)
        fb.batch_size, fb.input_ids, fb.positions, fb.rope_positions = bs, torch.arange(n), physical, physical + offsets
        fb.req_pool_indices, fb.seq_lens, fb.seq_lens_cpu = torch.arange(bs), torch.arange(bs) + 40, torch.arange(bs) + 40
        fb.seq_lens_sum, fb.out_cache_loc = int(fb.seq_lens.sum()), torch.arange(n * (3 if phase == "draft" else 1))
        fb.extend_seq_lens, fb.extend_seq_lens_cpu = torch.full((bs,), width), [width] * bs
        fb.spec_info = (EagleDraftInput(topk_p=torch.ones(bs, 1), topk_index=torch.ones(bs, 1, dtype=torch.int64), hidden_states=torch.ones(bs, 2))
            if phase == "draft" else EagleDraftExtendInput(hidden_states=torch.ones(n, 2),
                num_correct_drafts=torch.zeros(bs, dtype=torch.int32), num_accept_tokens=torch.ones(bs, dtype=torch.int32),
                num_tokens_per_req=width))
        expected = fb.rope_positions.clone()
        if phase == "draft": graph.execute(fb)
        else: graph.execute(fb, torch.arange(bs) * width)
        assert graph.buffers.rope_positions.data_ptr() == ptr
        assert torch.equal(graph.buffers.rope_positions[:n], expected)
        assert torch.count_nonzero(graph.buffers.rope_positions[n:]) == 0
        assert torch.equal(graph.buffers.positions[:n], physical)


def test_real_target_verify_late_load_refreshes_model_positions():
    graph = object.__new__(DecodeCudaGraphRunner)
    graph.dllm_uses_input_embeds = graph.ragged_verify_mode = graph.enable_pdmux = False
    graph.deepep_adapter = NS(replay=lambda: None)
    graph.bs, graph.raw_num_token, graph.captured_req_width = 3, 8, 4
    graph._capture_graph_size = lambda **kw: 3
    graph._resolve_lora_variant = graph._resolve_attention_variant = lambda fb: None
    graph._make_graph_key = lambda *args: args
    graph.model_runner = NS(spec_algorithm=SpeculativeAlgorithm.from_string("EAGLE"), is_draft_worker=False)
    graph.buffers = NS(input_ids=torch.zeros(12, dtype=torch.int64), positions=torch.zeros(12, dtype=torch.int64),
        rope_positions=torch.zeros(12, dtype=torch.int64), pp_proxy_tensors=None)
    fb = NS(needs_forward_metadata_init=lambda: False, input_ids=torch.arange(8), positions=torch.arange(8) + 100,
        rope_positions=torch.arange(8) + torch.tensor([0]*4 + [262152]*4) + 100)
    graph.load_batch(fb)
    assert torch.equal(graph.buffers.positions[:8], fb.positions)
    assert torch.equal(graph.buffers.rope_positions[:8], fb.rope_positions)


def test_real_draft_loop_advances_physical_and_model_positions_together(monkeypatch):
    import sglang.srt.speculative.eagle_worker_v2 as module
    worker = object.__new__(module.EagleDraftWorker)
    worker.topk, worker.speculative_num_steps, worker.speculative_num_draft_tokens = 1, 3, 4
    worker.hot_token_id, worker._topk1_parents_prealloc = None, torch.zeros(3, 3)
    worker._topk1_score_indices_prealloc = torch.zeros(3, 3, dtype=torch.int64)
    worker.index_share_for_mtp_iteration = worker.seed_dsa_topk_from_draft_extend = False
    worker.draft_attn_backend = NS(attn_backends=[None, None])
    seen = []
    def forward(fb):
        seen.append((fb.positions.clone(), fb.rope_positions.clone()))
        return NS(logits_output=NS(next_token_logits=torch.tensor([[1., 2.], [2., 1.]]), hidden_states=torch.ones(2, 2)))
    worker.draft_runner = NS(forward=forward, canary_manager=None,
        model_config=NS(hf_config=NS(architectures=["DeepseekV3ForCausalLMNextN"]), model_is_mrope=False))
    monkeypatch.setattr(module, "_is_cuda", False)
    monkeypatch.setattr(module, "_is_hip", False)
    monkeypatch.setattr(module, "get_spec", lambda: NS(speculative_use_rejection_sampling=False))
    monkeypatch.setattr(module, "forward_context", lambda *a: nullcontext())
    monkeypatch.setattr(module, "IndexTopKShareState", NS(mtp_iteration=lambda *a, **kw: nullcontext()))
    monkeypatch.setattr(module, "per_step_draft_out_cache_loc", lambda loc, *a: loc.reshape(3, 2))
    monkeypatch.setattr(module, "select_top_k_tokens", lambda i, p, ids, h, score, k: (ids.flatten(), h, p, (p, ids, ids)))
    monkeypatch.setattr(module, "organize_draft_results", lambda *a: (None, None, None))
    fb = NS(forward_mode=ForwardMode.DECODE, out_cache_loc=torch.arange(6), batch_size=2,
        positions=torch.tensor([262143, 524287]), rope_positions=torch.tensor([262143, 524287+262152]),
        spec_info=EagleDraftInput(topk_p=torch.ones(2, 1), topk_index=torch.ones(2, 1, dtype=torch.int64), hidden_states=torch.ones(2, 2)))
    worker.draft_forward(fb)
    assert len(seen) == 2
    for step, (physical, model) in enumerate(seen):
        assert torch.equal(model - physical, torch.tensor([0, 262152]))
        assert torch.equal(physical, torch.tensor([262143, 524287]) + step)
    assert torch.equal(fb.rope_positions - fb.positions, torch.tensor([0, 262152]))


def test_real_cache_namespace_separates_profile_and_caller_salt():
    cache, tokens = RadixCache.create_simulated(), array("q", range(16))
    def key(factor, salt="a"):
        req = NS(origin_input_ids=tokens, sampling_params=NS(max_new_tokens=4, custom_params={"yarn_factor": factor}),
            rope_profile=None, extra_key="caller", multimodal_inputs=None)
        admit_request_static_yarn(req, config(), 524288)
        return RadixKey(tokens, req.extra_key, cache_salt=salt)
    values = torch.arange(16)
    cache.insert(InsertParams(key=key(64), value=values))
    assert torch.equal(cache.match_prefix(MatchPrefixParams(key=key(64))).device_indices, values)
    assert len(cache.match_prefix(MatchPrefixParams(key=key(128))).device_indices) == 0
    assert len(cache.match_prefix(MatchPrefixParams(key=key(64, "b"))).device_indices) == 0


def test_real_dynamic_frontend_keeps_exact_public_budget_and_rejects_without_truncation():
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    manager = object.__new__(TokenizerManager)
    manager.context_len, manager.allow_auto_truncate, manager.validate_total_tokens = 524288, True, True
    manager.num_reserved_tokens = 4
    manager.model_config = NS(request_static_yarn_config=config(), xing_eagle_context_headroom=0)
    execution = NS(features=NS(enable_encoder_swa_bounded_replay=False))
    with patch("sglang.srt.managers.tokenizer_manager.get_exec", return_value=execution):
        ids = array("q", [35]) * 524032
        req = NS(sampling_params={"max_new_tokens": 256})
        manager._validate_token_budget(req, ids)
        assert len(ids) == 524032 and req.sampling_params["max_new_tokens"] == 256
        ids.append(35)
        with pytest.raises(ValueError):
            manager._validate_token_budget(req, ids)
        assert len(ids) == 524033 and req.sampling_params["max_new_tokens"] == 256


@pytest.mark.parametrize("nextn", [False, True])
def test_real_attention_constructor_uses_long_profile_start_after_short_scratch(nextn, monkeypatch):
    import sglang.srt.models.deepseek_v2 as module
    monkeypatch.setattr(module, "get_parallel", lambda: NS(attn_tp_rank=0, attn_tp_size=1, dcp_enabled=False))
    monkeypatch.setattr(module, "get_model", lambda: NS(kv_cache_dtype="fp8_e4m3"))
    monkeypatch.setattr(module, "get_device", lambda: NS(device="cpu"))
    monkeypatch.setattr(module, "is_deepseek_dsa", lambda cfg: False)
    for name in ("ColumnParallelLinear", "ReplicatedLinear", "RowParallelLinear", "RadixAttention", "RMSNorm"):
        monkeypatch.setattr(module, name, lambda *a, **kw: torch.nn.Identity())
    for name in ("init_mha_forward", "init_mla_forward", "init_mla_fused_rope_rocm_forward", "init_mla_fused_rope_cpu_forward"):
        monkeypatch.setattr(module.DeepseekV2AttentionMLA, name, lambda self: None)
    def cpu_rope(*args, **kwargs):
        kwargs.pop("device", None)
        return get_rope(*args, **kwargs, dtype=torch.float32)
    monkeypatch.setattr(module, "get_rope_wrapper", cpu_rope)
    attention = module.DeepseekV2AttentionMLA(NS(rms_norm_eps=1e-6), 8, 2, 128, 64, 128,
        None, 512, rope_scaling={**PARAMETERS, "_request_static_yarn_lookahead": 8},
        max_position_embeddings=524288, is_nextn=nextn)
    assert attention.request_static_yarn_long_offset == attention.rotary_emb.scaling_factor_to_offset[128] == 262152
    physical = torch.tensor([262143, 262144, 262151, 0])
    fb = NS(rope_positions=physical + torch.tensor([0, 0, 0, 262152]))
    query = torch.ones(4, 2, 64)
    scaled = scale_request_static_yarn_query(query, fb, attention)
    assert torch.equal(scaled[:3], query[:3])
    torch.testing.assert_close(scaled[3:], query[3:] * attention.request_static_yarn_scale_ratio, rtol=0, atol=0)


def test_real_draft_capture_restores_both_position_buffers_between_warmups(monkeypatch):
    import sglang.srt.speculative.eagle_draft_cuda_graph_runner as module
    graph, _ = graph_fixture("draft", monkeypatch)
    graph.buffers.positions.copy_(torch.tensor([19, 23, 31]))
    graph.buffers.rope_positions.copy_(graph.buffers.positions + torch.tensor([0, 262152, 0]))
    original = graph.buffers.rope_positions.clone()
    graph.temperatures, graph.top_ks = torch.ones(3, 1), torch.ones(3, dtype=torch.int32)
    seen = []
    def draft(fb):
        seen.append((fb.positions.clone(), fb.rope_positions.clone()))
        fb.positions.add_(2)
        fb.rope_positions.add_(2)
        return ()
    graph.eagle_worker = NS(speculative_num_steps=3, draft_forward=draft)
    graph.draft_attn_backend = NS(init_forward_metadata_in_graph=lambda fb: None,
        init_forward_metadata_out_graph=lambda fb, **kw: None)
    graph.deepep_adapter.capture = lambda **kw: None
    graph.backend = NS(capture_one=lambda key, run, **kw: (run(), run()))
    monkeypatch.setattr(module, "forward_context", lambda *a: nullcontext())
    monkeypatch.setattr(module, "set_dp_buffer_len", lambda *a: None)
    monkeypatch.setattr(module, "set_is_extend_in_batch", lambda *a: None)
    monkeypatch.setattr(module, "maybe_flashinfer_autotune_speculative_draft", lambda *a, **kw: None)
    monkeypatch.setattr(module.DpPaddingMode, "get_default_mode_in_cuda_graph", lambda: module.DpPaddingMode.MAX_LEN)
    graph.capture_one_shape(3, None)
    assert len(seen) == 2
    assert torch.equal(seen[0][0], seen[1][0]) and torch.equal(seen[0][1], seen[1][1])
    assert torch.equal(graph.buffers.rope_positions, original)
