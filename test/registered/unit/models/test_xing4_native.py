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


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
@pytest.mark.parametrize("dense_tp_size", [1, 2])
def test_actual_full_constructor_uses_stable_layer_boundary_api(
    kind, dense_tp_size, monkeypatch
):
    """Execute every official decoder constructor without GPU weight storage.

    Module storage and distributed stage binding are stubbed with their real
    signatures. The Xing constructors, local imports, reduction enums, FFN
    declarations, mHC operands and attention-context initialization run for real.
    """
    from sglang.srt.layers import logits_processor
    from sglang.srt.layers.layer_boundary import HandoffRows, ProducerReduction
    from sglang.srt.layers.layer_boundary import layout
    from sglang.srt.layers.layer_boundary.adapters import attention
    from sglang.srt.models import deepseek_v2, xing4_0

    source = json.loads((FIXTURES / f"{kind}.json").read_text())
    config = Xing4_0Config(**source)
    parallel = SimpleNamespace(
        tp_size=1, moe_dense_tp_size=dense_tp_size,
        enable_dp_lm_head=False, enable_attn_tp_input_scattered=False,
    )
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

    layers_signature = inspect.signature(xing4_0.make_layers)
    stages_signature = inspect.signature(xing4_0.make_stages)

    def make_layers(num_layers, layer_fn, **kwargs):
        layers_signature.bind(num_layers, layer_fn, **kwargs)
        return nn.ModuleList([
            layer_fn(idx, f"{kwargs['prefix']}.{idx}") for idx in range(num_layers)
        ]), 0, num_layers

    def make_stages(*args, **kwargs):
        stages_signature.bind(*args, **kwargs)
        stage_calls.append((args, kwargs))
        return object(), object()

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
    monkeypatch.setattr(xing4_0, "make_layers", make_layers)
    monkeypatch.setattr(xing4_0, "make_stages", make_stages)
    monkeypatch.setattr(xing4_0, "get_pp_group", lambda: pp)
    for module in (xing4_0, layout, attention):
        monkeypatch.setattr(module, "get_parallel", lambda: parallel)
    monkeypatch.setattr(xing4_0, "get_attn_tp_context", lambda: context)
    quant_config = SimpleNamespace(update_packed_modules_mapping=lambda _: None)
    model = Xing4_0ForCausalLM(
        config, quant_config=quant_config if kind == "fp8" else None
    )

    assert len(model.model.layers) == config.num_hidden_layers
    assert len(stage_calls) == config.num_hidden_layers
    for idx, layer in enumerate(model.model.layers):
        sparse = idx >= config.first_k_dense_replace and idx % config.moe_layer_freq == 0
        assert layer.is_layer_sparse is sparse
        assert layer.enable_mhc
        assert layer.attn_hc.hc_fn.shape[-1] == config.hidden_size * config.hc_mult
        if not sparse:
            expected = (0, 1) if dense_tp_size == 1 else (None, None)
            assert (layer.mlp.constructor_kwargs["tp_rank"],
                    layer.mlp.constructor_kwargs["tp_size"]) == expected
        stages, kwargs = stage_calls[idx]
        assert stages[1][0].sparse is sparse
        assert stages[1][0].reduction is ProducerReduction.SCOPED
        if idx == config.first_k_dense_replace - 1:
            assert stages[1][0].handoff_rows is HandoffRows.TBO_SPLIT
        assert kwargs["terminal"] is (idx == config.num_hidden_layers - 1)
    assert context.allow_input_scattered is False
