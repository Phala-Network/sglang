"""Actual CPU RoPE/ForwardBatch/registry checks; CUDA gates are explicit."""

from array import array
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.model_config import ModelConfig, compute_mla_mscale_scaling
from sglang.srt.configs.request_static_yarn import RequestStaticYarnConfig, admit_request_static_yarn
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.request_static_yarn import scale_request_static_yarn_query
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.model_executor.cuda_graph_buffer_registry import build_decode_registry, build_prefill_registry
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

PARAMETERS = {
    "rope_type": "deepseek_yarn", "factor": 128,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32, "beta_slow": 1, "mscale": 1, "mscale_all_dim": 1,
}


def config():
    return RequestStaticYarnConfig.from_hf_config(SimpleNamespace(
        model_type="xing4_0", rope_scaling={**PARAMETERS, "request_static_factors": [64, 128]},
        rope_theta=10000, qk_nope_head_dim=128, qk_rope_head_dim=64,
    ))


def rope(factor=None, dtype=torch.float32):
    params = dict(PARAMETERS)
    if factor is None:
        params["request_static_factors"] = [64, 128]
    else:
        params["factor"] = factor
    return get_rope(
        head_size=64, rotary_dim=64, max_position=524288, base=10000,
        is_neox_style=False, rope_scaling=params, dtype=dtype,
    )


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
def test_real_native_model_config_accepts_dynamic_override(kind, tmp_path):
    fixture = Path(__file__).parents[1] / "models/xing4_configs" / f"{kind}.json"
    (tmp_path / "config.json").write_text(fixture.read_text())
    override = {
        "max_position_embeddings": 524288,
        "rope_scaling": {
            **PARAMETERS, "rope_type": "yarn", "request_static_factors": [64, 128]
        },
    }
    model = ModelConfig(str(tmp_path), trust_remote_code=False,
        context_length=524288, model_override_args=json.dumps(override))
    assert model.dtype == torch.bfloat16
    assert model.context_len == 524288
    assert model.request_static_yarn_config.position_kind == "mla"
    assert model.request_static_yarn_config.select(524288).factor == 128
    assert model.request_static_yarn_config.select(262144).factor == 64


def test_real_runner_rejects_resolved_backend_and_nonbf16_before_weight_allocation():
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    model = SimpleNamespace(request_static_yarn_config=config(), dtype=torch.bfloat16)
    args = ServerArgs(model_path="unused", device="cuda", attention_backend="trtllm_mla")
    args._resolved_overrides = [("test", {"decode_attention_backend": "fa3"})]
    with pytest.raises(ValueError, match="both prefill and decode"):
        ModelRunner(model, 0.8, 0, 1, args)
    args._resolved_overrides = []
    model.dtype = torch.float16
    with pytest.raises(ValueError, match="BF16 activations"):
        ModelRunner(model, 0.8, 0, 1, args)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_concatenated_rope_tables_match_isolated_static_tables(dtype):
    combined = rope(dtype=dtype)
    gen = torch.Generator().manual_seed(5014)
    positions = torch.tensor([0, 7, 262143, 0, 262144, 524287])
    q = torch.randn(6, 2, 64, generator=gen).to(dtype)
    k = torch.randn(6, 1, 64, generator=gen).to(dtype)
    offsets = torch.tensor([0, 0, 0, 262144, 262144, 262144])
    actual = combined.forward_native(positions + offsets, q, k)
    for index, factor in enumerate((64, 128)):
        static = rope(factor, dtype)
        start = combined.scaling_factor_to_offset[factor]
        torch.testing.assert_close(
            combined.cos_sin_cache[start:start + int(4096 * factor)],
            static.cos_sin_cache, rtol=0, atol=0,
        )
        sl = slice(index * 3, (index + 1) * 3)
        expected = static.forward_native(positions[sl], q[sl], k[sl])
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result[sl], reference, rtol=0, atol=0)
    with pytest.raises(ValueError, match="table capacity"):
        combined._ensure_cos_sin_cache_length(786432)


@pytest.mark.parametrize("mode,lengths", [
    (ForwardMode.EXTEND, [3, 1, 2]), (ForwardMode.DECODE, [1, 1, 1]),
])
def test_actual_forward_batch_mixed_offsets_preserve_physical_positions_and_kv(mode, lengths):
    profiles = [config().select(32, f) for f in (128, 64, 128)]
    fb = object.__new__(ForwardBatch)
    fb.forward_mode = mode
    fb.positions = torch.arange(sum(lengths)) + 19
    fb.out_cache_loc = torch.arange(sum(lengths)) + 72
    physical, kv = fb.positions.clone(), fb.out_cache_loc.clone()
    batch = SimpleNamespace(reqs=[SimpleNamespace(rope_profile=p) for p in profiles], extend_lens=lengths)
    runner = SimpleNamespace(device="cpu", model_config=SimpleNamespace(request_static_yarn_config=config()))
    fb._init_request_static_yarn_mla(runner, batch)
    assert torch.equal(fb.positions, physical) and torch.equal(fb.out_cache_loc, kv)
    expected = physical + torch.tensor([p.offset for p in profiles]).repeat_interleave(torch.tensor(lengths))
    assert torch.equal(fb.rope_positions, expected)
    # Chunk/retraction metadata rebuild selects the existing immutable request
    # profile, even though physical position and input length changed.
    fb.positions = physical + 101
    fb._init_request_static_yarn_mla(runner, batch)
    assert torch.equal(fb.rope_positions, expected + 101)
    batch.reqs[0].rope_profile = None
    with pytest.raises(RuntimeError, match="profile missing"):
        fb._init_request_static_yarn_mla(runner, batch)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_scalar_and_real_q_scaling_match_static_attention_math(dtype):
    base = 192 ** -0.5
    scale64 = compute_mla_mscale_scaling({**PARAMETERS, "factor": 64}, base)
    scale128 = compute_mla_mscale_scaling(PARAMETERS, base)
    dynamic_scale = compute_mla_mscale_scaling({**PARAMETERS, "request_static_factors": [64, 128]}, base)
    assert dynamic_scale == scale64
    assert scale64 == pytest.approx(0.144679625919)
    assert scale128 == pytest.approx(0.159191925976)
    gen = torch.Generator().manual_seed(15014)
    q = torch.randn(6, 2, 576, generator=gen).to(dtype)
    k = torch.randn(6, 2, 576, generator=gen).to(dtype)
    fb = SimpleNamespace(rope_positions=torch.tensor([0, 1, 262143, 262144, 262145, 786431]))
    module = SimpleNamespace(request_static_yarn_scale_ratio=scale128 / scale64, request_static_yarn_long_offset=262144)
    actual = scale_request_static_yarn_query(q, fb, module)
    assert torch.equal(actual[:3], q[:3])
    actual_logits = (actual.float() * k.float()).sum(-1) * scale64
    expected = (q.float() * k.float()).sum(-1) * torch.tensor([scale64] * 3 + [scale128] * 3)[:, None]
    torch.testing.assert_close(actual_logits, expected, rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
                               atol=0.03 if dtype == torch.bfloat16 else 2e-6)
    # Assert the FP8 cast observes scaled long Q; re-scaling already quantized
    # Q is not equivalent and the helper explicitly rejects that input.
    expected_q8 = (q[3:].float() * (scale128 / scale64)).to(dtype).to(torch.float8_e4m3fn)
    assert torch.equal(actual[3:].to(torch.float8_e4m3fn).view(torch.uint8), expected_q8.view(torch.uint8))
    with pytest.raises(RuntimeError, match="before quantization"):
        scale_request_static_yarn_query(q.to(torch.float8_e4m3fn), fb, module)


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_actual_graph_registry_copies_profile_changes_and_clears_stale_tail(phase):
    common = dict(device=torch.device("cpu"), max_bs=3, max_num_token=6,
                  cache_loc_dtype=torch.int64, enable_request_static_yarn=True, share_pool=False)
    registry = (build_decode_registry(seq_len_fill_value=1, **common)
                if phase == "decode" else build_prefill_registry(**common))
    ptr = registry.get_slot("rope_positions").buffer.data_ptr()
    for positions in ([262144, 3, 262150], [4, 262149], [5]):
        n = len(positions)
        fb = SimpleNamespace(rope_positions=torch.tensor(positions), positions=torch.arange(n),
                             input_ids=torch.arange(n), out_cache_loc=torch.arange(n))
        registry.fill_from(fb, raw_bs=n, padded_bs=3, raw_num_tokens=n, padded_num_tokens=6)
        buffer = registry.get_slot("rope_positions").buffer
        assert buffer.data_ptr() == ptr
        assert torch.equal(buffer[:n], fb.rope_positions)
        assert torch.count_nonzero(buffer[n:]) == 0
        assert torch.equal(registry.get_slot("positions").buffer[:n], fb.positions)


def test_real_radix_namespace_isolates_profiles_caller_identity_and_salt():
    cache = RadixCache.create_simulated()
    tokens = array("q", range(16))
    def key(factor, caller="tenant:[x]", salt="salt:one"):
        req = SimpleNamespace(origin_input_ids=tokens,
            sampling_params=SimpleNamespace(max_new_tokens=1, custom_params={"yarn_factor": factor}),
            rope_profile=None, extra_key=caller, multimodal_inputs=None)
        admit_request_static_yarn(req, config(), 524288)
        return RadixKey(tokens, extra_key=req.extra_key, cache_salt=salt)
    for factor in (64, 128):
        profile_key = key(factor)
        assert len(cache.match_prefix(MatchPrefixParams(key=profile_key)).device_indices) == 0
        values = torch.arange(16) + factor
        cache.insert(InsertParams(key=profile_key, value=values))
        assert torch.equal(cache.match_prefix(MatchPrefixParams(key=key(factor))).device_indices, values)
    assert len(cache.match_prefix(MatchPrefixParams(key=key(64, salt="salt:two"))).device_indices) == 0
    assert len(cache.match_prefix(MatchPrefixParams(key=key(64, caller="other"))).device_indices) == 0


class TupleLinear(torch.nn.Linear):
    def forward(self, x):
        return super().forward(x), None


@pytest.mark.parametrize("kv_dtype", [torch.float32, torch.float8_e4m3fn])
def test_actual_mha_prepare_scales_q_before_backend_rope_quant_hook(kv_dtype):
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import DeepseekMHAForwardMixin
    torch.manual_seed(7014)
    module = SimpleNamespace(
        q_lora_rank=None, num_local_heads=2, qk_head_dim=192,
        qk_nope_head_dim=128, qk_rope_head_dim=64, kv_lora_rank=512,
        q_proj=TupleLinear(8, 384), kv_a_proj_with_mqa=TupleLinear(8, 576),
        kv_a_layernorm=torch.nn.RMSNorm(512), kv_b_proj=TupleLinear(512, 512),
        rotary_emb=rope(), request_static_yarn_scale_ratio=None,
    )
    scale64 = compute_mla_mscale_scaling({**PARAMETERS, "factor": 64}, 192**-0.5)
    scale128 = compute_mla_mscale_scaling(PARAMETERS, 192**-0.5)
    hidden = torch.randn(4, 8)
    physical = torch.tensor([5, 13, 17, 29])
    # CPU reference backend at the real prepare_prefill_qkv seam. CUDA
    # TRTLLM quantization itself remains a separate GPU gate.
    class CpuBackend:
        def prepare_prefill_qkv(self, *, q, q_pe, kv_a, k_pe, positions, layer, forward_batch):
            self.q_before_quant = q.clone()
            self.positions = positions.clone()
            q_pe, k_pe = layer.rotary_emb.forward_native(positions, q_pe, k_pe)
            q = q.clone()
            q[..., 128:] = q_pe
            kv = layer.kv_b_proj(kv_a)[0].view(-1, 2, 256)
            k = torch.cat((kv[..., :128], k_pe.expand(-1, 2, -1)), dim=-1)
            return q.to(kv_dtype), k.to(kv_dtype), kv[..., 128:].to(kv_dtype)
    backend = CpuBackend()
    method = DeepseekMHAForwardMixin.forward_normal_prepare
    with patch("sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha.resolve_attn_backend", return_value=backend):
        module.request_static_yarn_scale_ratio = scale128 / scale64
        module.request_static_yarn_long_offset = 262144
        fb = SimpleNamespace(rope_positions=physical + torch.tensor([0, 0, 262144, 262144]))
        actual = method(module, physical, hidden, fb, None)[:3]
        projected = module.q_proj(hidden)[0].view(4, 2, 192)
        assert torch.equal(backend.q_before_quant[:2], projected[:2])
        torch.testing.assert_close(backend.q_before_quant[2:], projected[2:] * (scale128 / scale64), rtol=0, atol=0)
        assert torch.equal(backend.positions, fb.rope_positions)
        for i, factor in enumerate((64, 128)):
            sl = slice(i*2, (i+1)*2)
            module.rotary_emb = rope(factor)
            module.request_static_yarn_scale_ratio = None
            # Keep the same GEMM shape: slicing the batch before projection
            # can change FP32 reduction rounding independently of YaRN.
            static = method(module, physical, hidden, SimpleNamespace(rope_positions=None), None)[:3]
            # K/V must be independent of attention scale. Q uses the correct
            # table and ratio before the backend's FP8 cast.
            for observed, expected in zip(actual[1:], static[1:]):
                assert torch.equal(observed[sl].view(torch.uint8), expected[sl].view(torch.uint8))
            if factor == 64:
                assert torch.equal(actual[0][sl].view(torch.uint8), static[0][sl].view(torch.uint8))


def test_actual_mla_core_scales_post_absorb_q_before_backend_and_keeps_kv():
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import DeepseekMLAForwardMixin
    torch.manual_seed(8014)
    q_nope = torch.randn(4, 2, 128)
    w_kc = torch.randn(2, 128, 512) / 16
    absorbed = torch.bmm(q_nope.transpose(0, 1), w_kc).transpose(0, 1)
    q_rope, k_rope = rope().forward_native(torch.tensor([7, 13, 262151, 262157]),
        torch.randn(4, 2, 64), torch.randn(4, 1, 64))
    k_latent = torch.randn(4, 1, 512)
    scale64 = compute_mla_mscale_scaling({**PARAMETERS, "factor": 64}, 192**-0.5)
    scale128 = compute_mla_mscale_scaling(PARAMETERS, 192**-0.5)
    seen = {}
    def cpu_attention(q, k, v, fb, *, q_rope, k_rope, **kwargs):
        seen.update(q=q.clone(), q_rope=q_rope.clone(), k=k.clone(), k_rope=k_rope.clone())
        query = torch.cat((q, q_rope), dim=-1)
        key = torch.cat((k, k_rope), dim=-1).expand(-1, 2, -1)
        logits = torch.einsum("thd,shd->hts", query, key) * scale64
        return torch.einsum("hts,shd->thd", logits.softmax(-1), v.expand(-1, 2, -1))
    module = SimpleNamespace(
        num_local_heads=2, kv_lora_rank=512, v_head_dim=128,
        current_attention_backend="trtllm_mla", attn_mqa=cpu_attention,
        _fuse_rope_for_trtllm_mla=lambda fb: False, use_deep_gemm_bmm=False,
        w_vc=torch.randn(2, 512, 128)/16, o_proj=TupleLinear(256, 8),
        next_skip_topk=None, request_static_yarn_scale_ratio=scale128/scale64,
        request_static_yarn_long_offset=262144,
    )
    fb = SimpleNamespace(rope_positions=torch.tensor([7, 13, 262151, 262157]), forward_mode=ForwardMode.DECODE)
    with patch("sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla.get_parallel", return_value=SimpleNamespace(dcp_enabled=False)):
        output = DeepseekMLAForwardMixin.forward_absorb_core(
            module, q_rope, k_rope, absorbed, k_latent, fb, None, fb.rope_positions, None, None)
    assert torch.isfinite(output).all()
    assert torch.equal(seen["q"][:2], absorbed[:2])
    torch.testing.assert_close(seen["q"][2:], absorbed[2:] * (scale128/scale64), rtol=0, atol=0)
    assert torch.equal(seen["k"], k_latent) and torch.equal(seen["k_rope"], k_rope)
    torch.testing.assert_close(seen["q_rope"][2:], q_rope[2:] * (scale128/scale64), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA device required")
def test_actual_cuda_mixed_rope_and_graph_replay_match_static_profiles():
    combined = rope(dtype=torch.bfloat16).to("cuda")
    positions = torch.tensor([7, 262151], device="cuda")
    q = torch.randn(2, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 1, 64, device="cuda", dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    for _ in range(3):
        combined(positions, q.clone(), k.clone())
    with torch.cuda.graph(graph):
        actual = combined(positions, q.clone(), k.clone())
    for factors in ((64, 128), (128, 64), (64, 64)):
        positions.copy_(torch.tensor([7 + (262144 if f == 128 else 0) for f in factors], device="cuda"))
        graph.replay()
        for i, factor in enumerate(factors):
            expected = rope(factor, torch.bfloat16).to("cuda")(
                torch.tensor([7], device="cuda"), q[i:i+1].clone(), k[i:i+1].clone())
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result[i:i+1], reference, rtol=0, atol=0)
