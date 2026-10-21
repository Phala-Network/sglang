"""CPU coverage of real Xing configuration, operand loading and protocol code.

CUDA attention and TileLang execution require the separate B300 qualification.
"""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.configs.model_config import ModelConfig, is_deepseek_dsa
from sglang.srt.configs.xing4_0 import Xing4_0Config
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.function_call.xing4_0_detector import Xing4_0Detector
from sglang.srt.models.xing4_0 import (
    Xing4_0ForCausalLM,
    input_expand,
    mHCModule,
    output_contract,
)
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils.hf_transformers.config import get_config

FIXTURES = Path(__file__).with_name("xing4_configs")


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
def test_official_config_loading_and_mla(kind, tmp_path):
    source = json.loads((FIXTURES / f"{kind}.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(source))
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert isinstance(config, Xing4_0Config)
    assert config.hc_mult == 4
    assert config.rope_scaling["factor"] == 64
    assert config.rope_scaling["original_max_position_embeddings"] == 4096
    assert config.qk_nope_head_dim + config.qk_rope_head_dim == 192
    assert config.kv_lora_rank + config.qk_rope_head_dim == 576
    assert not is_deepseek_dsa(config)
    if kind == "fp8":
        assert config.quantization_config["weight_block_size"] == [128, 128]
        assert config.quantization_config["quant_method"] == "fp8"


@pytest.mark.parametrize("tokens", [0, 1, 7])
def test_mhc_stream_expand_contract(tokens):
    x = torch.randn(tokens, 1, 3584)
    streams = input_expand(x, 4)
    assert streams.shape == (tokens, 1, 14336)
    assert torch.equal(output_contract(streams, 4), x)


class OperandHarness(nn.Module):
    load_weights = Xing4_0ForCausalLM.load_weights

    def __init__(self):
        super().__init__()
        self.attn_hc = mHCModule(Xing4_0Config(architectures=["Xing4_0ForCausalLM"]), 0)
        self.forwarded = None

    def do_load_weights(self, weights):
        self.forwarded = weights


@pytest.mark.parametrize("prefix", ["", "model."])
def test_mhc_actual_operand_loader(prefix):
    harness = OperandHarness()
    weights = [
        (prefix + "attn_hc." + name, torch.full_like(param, 0.25))
        for name, param in harness.attn_hc.named_parameters()
    ]
    weights.append(("model.fake.weight", torch.ones(2)))
    harness.load_weights(weights)
    assert len(harness.forwarded) == 1
    for name in ("fn_fp32", "base_fp32", "scale_fp32"):
        buffer = getattr(harness.attn_hc, name)
        assert buffer.dtype == torch.float32
        assert torch.all(buffer == 0.25)
    assert harness.attn_hc._finalized


def test_mhc_operand_shape_failure_is_not_swallowed():
    with pytest.raises((AssertionError, RuntimeError)):
        OperandHarness().load_weights([("attn_hc.hc_fn", torch.zeros(2, 3))])


def test_unknown_mhc_operand_is_not_swallowed():
    with pytest.raises(ValueError, match="no parameter"):
        OperandHarness().load_weights([("attn_hc.unknown", torch.zeros(2))])


def test_missing_mhc_operands_are_rejected():
    harness = OperandHarness()
    with pytest.raises(ValueError, match="operands missing"):
        harness.load_weights([("attn_hc.hc_scale", torch.zeros(3))])
    assert not harness.attn_hc._finalized


def tools():
    return [Tool(type="function", function={"name": "lookup", "parameters": {
        "type": "object", "properties": {"count": {"type": "integer"},
        "city": {"type": "string"}}, "required": ["city"]}})]


@pytest.mark.parametrize("payload", [
    '{"name":"lookup","arguments":{"city":"Paris","count":2}}',
    'lookup<param_key>city</param_key><param_value>Paris</param_value>'
    '<param_key>count</param_key><param_value>2</param_value>',
    'lookup{"city":"Paris","count":2}',
])
def test_tools_complete_and_character_stream(payload):
    text = "prefix<tool_call>" + payload + "</tool_call>"
    parsed = Xing4_0Detector().detect_and_parse(text, tools())
    assert parsed.normal_text == "prefix"
    assert len(parsed.calls) == 1
    assert json.loads(parsed.calls[0].parameters) == {"city": "Paris", "count": 2}
    detector = Xing4_0Detector()
    calls = []
    content = ""
    for char in text:
        part = detector.parse_streaming_increment(char, tools())
        calls += part.calls
        content += part.normal_text
    assert content == "prefix"
    assert len(calls) == 1
    assert calls[0].parameters == parsed.calls[0].parameters


def test_reasoning_marker_survives_default_detokenization():
    owner = SimpleNamespace(reasoning_parser="xing4_0", chat_encoding_spec=None)
    request = SimpleNamespace(skip_special_tokens=True, reasoning_effort=None)
    OpenAIServingChat._patch_reasoning_skip_special_tokens(owner, request)
    assert request.skip_special_tokens is False
    parser = ReasoningParser("xing4_0", force_reasoning=True)
    reasoning, content = parser.parse_non_stream("analysis</think>answer")
    assert reasoning == "analysis"
    assert content == "answer"


def test_native_model_api_import():
    assert "config" in inspect.signature(Xing4_0ForCausalLM).parameters
    assert "Xing4_0ForCausalLM" in inspect.getsource(ModelConfig._derive_model_shapes)


def _construct_cpu_xing_model(kind, dense_tp_size, monkeypatch, *, config_changes=None):
    from sglang.srt.layers import logits_processor
    from sglang.srt import runtime_context
    from sglang.srt.layers.layer_boundary import layout
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.layers.layer_boundary.adapters import attention
    from sglang.srt.models import deepseek_v2, xing4_0

    source = json.loads((FIXTURES / f"{kind}.json").read_text())
    source.update(config_changes or {})
    config = Xing4_0Config(**source)
    parallel = runtime_context.ParallelContext()
    context_owner = runtime_context.RuntimeContext(parallel=parallel)
    monkeypatch.setattr(runtime_context, "_PARALLEL", parallel)
    monkeypatch.setattr(runtime_context, "_CONTEXT", context_owner)
    runtime_context.publish(ServerArgs(model_path="dummy", boundary_reduction="ar",
        moe_dense_tp_size=dense_tp_size), role="scheduler")
    pp = SimpleNamespace(
        is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
    )
    context = attention.AttnTpContext()
    stage_calls = []

    def storage_stub(original, *, attention_module=False, moe_module=False):
        signature = inspect.signature(original.__init__)

        class StorageStub(nn.Module):
            def __init__(self, *args, **kwargs):
                nn.Module.__init__(self)
                signature.bind(self, *args, **kwargs)
                self.constructor_kwargs = kwargs
                if attention_module:
                    self.prepare_qkv_latent = lambda *_: None
                if moe_module:
                    self.experts = SimpleNamespace(quant_method=object())
                    self.num_fused_shared_experts = 0

        return StorageStub

    original_append = xing4_0.append_stages
    def append_stages(*args, **kwargs):
        stage_calls.append((args, kwargs))
        return original_append(*args, **kwargs)

    monkeypatch.setattr(deepseek_v2, "DeepseekV2AttentionMLA", storage_stub(
        deepseek_v2.DeepseekV2AttentionMLA, attention_module=True
    ))
    monkeypatch.setattr(deepseek_v2, "DeepseekV2MLP", storage_stub(deepseek_v2.DeepseekV2MLP))
    monkeypatch.setattr(deepseek_v2, "DeepseekV2MoE", storage_stub(
        deepseek_v2.DeepseekV2MoE, moe_module=True
    ))
    for name in ("VocabParallelEmbedding", "ParallelLMHead"):
        monkeypatch.setattr(xing4_0, name, storage_stub(getattr(xing4_0, name)))
    monkeypatch.setattr(logits_processor, "LogitsProcessor", storage_stub(
        logits_processor.LogitsProcessor
    ))
    monkeypatch.setattr(xing4_0, "append_stages", append_stages)
    monkeypatch.setattr(xing4_0, "get_pp_group", lambda: pp)
    for module in (xing4_0, layout, attention):
        monkeypatch.setattr(module, "get_parallel", lambda: parallel)
    monkeypatch.setattr(xing4_0, "get_attn_tp_context", lambda: context)
    quant_config = SimpleNamespace(update_packed_modules_mapping=lambda _: None)
    model = Xing4_0ForCausalLM(
        config, quant_config=quant_config if kind == "fp8" else None
    )
    return model, stage_calls, context


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
@pytest.mark.parametrize("dense_tp_size", [1])
def test_actual_full_constructor_uses_stable_layer_boundary_api(
    kind, dense_tp_size, monkeypatch
):
    """Run every official constructor with signature-constrained GPU storage."""

    model, stage_calls, context = _construct_cpu_xing_model(
        kind, dense_tp_size, monkeypatch
    )
    config = model.config

    assert len(model.model.layers) == config.num_hidden_layers
    assert len(stage_calls) == config.num_hidden_layers
    for idx, layer in enumerate(model.model.layers):
        sparse = idx >= config.first_k_dense_replace and idx % config.moe_layer_freq == 0
        assert layer.is_layer_sparse is sparse
        assert layer.enable_mhc
        assert layer.attn_hc.hc_fn.shape[-1] == config.hidden_size * config.hc_mult
        if not sparse:
            assert layer.mlp.constructor_kwargs["parallel_group"] == "replicated"
        stages, kwargs = stage_calls[idx]
        assert stages[1][0].sparse is sparse
        assert stages[1][0].output_complete is False
        assert kwargs == {}
        assert layer.ffn_boundary is not None
    assert context.allow_input_scattered is False


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
@pytest.mark.parametrize("profile", ["native64"])
@pytest.mark.parametrize("tokens", [0, 3])
def test_actual_full_forward_uses_stable_compute_and_terminal_api(
    kind, profile, tokens, monkeypatch
):
    """Run the full official stack and real attention/MLP/MoE forward methods.

    GPU compute is replaced at its boundaries and bound to the actual production
    signatures. Real mHC shape handling, RMSNorm, residual lifecycle and terminal
    normalization execute on CPU; this does not qualify CUDA kernels or weights.
    """
    from contextlib import nullcontext

    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.layers.moe import mega_moe
    from sglang.srt.models import deepseek_v2, xing4_0

    attention_cls = deepseek_v2.DeepseekV2AttentionMLA
    dense_cls = deepseek_v2.DeepseekV2MLP
    moe_cls = deepseek_v2.DeepseekV2MoE
    pre_signature = inspect.signature(xing4_0._mhc_pre_orig)
    post_signature = inspect.signature(xing4_0._mhc_post_orig)
    prepare_signature = inspect.signature(attention_cls.forward_prepare)
    moe_compute_signature = inspect.signature(moe_cls.forward_normal)
    mega_signature = inspect.signature(mega_moe.should_use_mega_moe)
    logits_signature = inspect.signature(LogitsProcessor.forward)
    decode_scope_signature = inspect.signature(attention_cls.maybe_use_decode_attn_tp)
    calls = {"attention": [], "dense": [], "moe": [], "mhc_pre": [], "mhc_post": []}

    owner, _, _ = _construct_cpu_xing_model(kind, 1, monkeypatch)
    config = owner.config

    def mhc_pre(*args, **kwargs):
        bound = pre_signature.bind(*args, **kwargs).arguments
        residual = bound["residual"]
        rows, streams, width = residual.shape
        assert (streams, width) == (config.hc_mult, config.hidden_size)
        assert bound["fn"].shape == (streams * streams + 2 * streams, streams * width)
        assert bound["n_splits_pre"] == 8
        calls["mhc_pre"].append(tuple(residual.shape))
        return (
            torch.zeros(rows, streams, 1, dtype=torch.float32),
            torch.eye(streams, dtype=torch.float32).expand(rows, streams, streams),
            residual.mean(dim=1),
        )

    def mhc_post(*args, **kwargs):
        bound = post_signature.bind(*args, **kwargs).arguments
        residual = bound["residual"]
        assert bound["x"].shape == (tokens, config.hidden_size)
        assert bound["post_layer_mix"].shape == (tokens, config.hc_mult, 1)
        assert bound["comb_res_mix"].shape == (tokens, config.hc_mult, config.hc_mult)
        calls["mhc_post"].append(tuple(residual.shape))
        return residual.clone()

    def forward_prepare(self, *args, **kwargs):
        bound = prepare_signature.bind(self, *args, **kwargs).arguments
        assert bound["input_on_attn_tp_slices"] is False
        assert bound["hidden_states"].shape == (tokens, config.hidden_size)
        calls["attention"].append(self.layer_id)
        return bound["hidden_states"], None, bound["forward_batch"], None

    def forward_normal(self, *args, **kwargs):
        bound = moe_compute_signature.bind(self, *args, **kwargs).arguments
        calls["moe"].append(self.layer_id)
        return bound["hidden_states"]

    def no_mega(*args, **kwargs):
        mega_signature.bind(*args, **kwargs)
        return False

    class Projection(nn.Module):
        def forward(self, value):
            return value, None

    def logits_forward(self, *args, **kwargs):
        bound = logits_signature.bind(self, *args, **kwargs).arguments
        return bound["hidden_states"]

    def decode_scope(self, *args, **kwargs):
        decode_scope_signature.bind(self, *args, **kwargs)
        return nullcontext()

    owner.logits_processor.forward = logits_forward.__get__(owner.logits_processor)
    monkeypatch.setattr(xing4_0, "mhc_pre", mhc_pre)
    monkeypatch.setattr(xing4_0, "mhc_post", mhc_post)
    monkeypatch.setattr(xing4_0, "attention_tensor_model_parallel_all_reduce", lambda x: x)
    monkeypatch.setattr(mega_moe, "should_use_mega_moe", no_mega)
    for layer in owner.model.layers:
        attention = layer.self_attn
        attention.layer_id = layer.layer_id
        attention.forward = attention_cls.forward.__get__(attention)
        attention.forward_prepare = forward_prepare.__get__(attention)
        attention.forward_core = attention_cls.forward_core.__get__(attention)
        attention.maybe_use_decode_attn_tp = decode_scope.__get__(attention)
        mlp = layer.mlp
        if layer.is_layer_sparse:
            mlp.layer_id = layer.layer_id
            mlp.gate = SimpleNamespace(e_score_correction_bias_vl=None)
            mlp._enable_a2a_moe = False
            mlp.alt_stream = None
            mlp._can_dual_stream_graph = lambda _: False
            mlp.forward = moe_cls.forward.__get__(mlp)
            mlp.forward_normal = forward_normal.__get__(mlp)
        else:
            mlp.tp_size = 1
            mlp.is_replicated = True
            mlp.swiglu_limit = None
            mlp.use_fused_clamp_act_mul = False
            mlp.gate_up_proj = Projection()
            mlp.down_proj = Projection()
            mlp.act_fn = nn.Identity()
            original_forward = dense_cls.forward

            def dense_forward(self, *args, _layer_id=layer.layer_id, **kwargs):
                calls["dense"].append(_layer_id)
                return original_forward(self, *args, **kwargs)

            mlp.forward = dense_forward.__get__(mlp)

    forward_batch = SimpleNamespace(
        can_run_tbo=False, residual_stream=None,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        moe_num_token_non_padded=lambda: None,
    )
    inputs = torch.ones(tokens, config.hidden_size, dtype=torch.bfloat16)
    result = owner(
        input_ids=None, positions=torch.arange(tokens),
        forward_batch=forward_batch, input_embeds=inputs,
    )
    assert result.shape == inputs.shape and result.dtype == torch.bfloat16
    assert torch.isfinite(result).all()
    assert calls["attention"] == list(range(config.num_hidden_layers))
    assert calls["dense"] == list(range(config.first_k_dense_replace))
    assert calls["moe"] == list(range(config.first_k_dense_replace, config.num_hidden_layers))
    assert len(calls["mhc_pre"]) == (2 * config.num_hidden_layers if tokens else 0)
    assert len(calls["mhc_post"]) == (2 * config.num_hidden_layers if tokens else 0)
    assert forward_batch.residual_stream is None
    assert xing4_0.get_forward().attn_inputs is None
