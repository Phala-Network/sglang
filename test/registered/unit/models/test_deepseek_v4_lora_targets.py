"""DeepSeek-V4 LoRA module-name gating and target dims, without a model instance."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM as Model
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# DeepSeek-V4.1-Flash text config
CONFIG = SimpleNamespace(
    hidden_size=5120,
    q_lora_rank=1280,
    o_lora_rank=1024,
    o_groups=8,
    num_attention_heads=64,
    head_dim=512,
    moe_intermediate_size=2304,
    n_shared_experts=1,
)


class TestDeepseekV4LoRATargets(CustomTestCase):
    def test_only_attention_and_shared_expert_projections_are_wrapped(self):
        wrapped = [
            f"model.layers.3.self_attn.{leaf}"
            for leaf in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
        ] + [
            "model.layers.3.mlp.shared_experts.gate_up_proj",
            "model.layers.3.mlp.shared_experts.down_proj",
        ]
        # The indexer, compressor and Engram reuse the leaves wq_b / wkv.
        skipped = [
            "model.layers.3.self_attn.indexer.wq_b",
            "model.layers.3.self_attn.compressor.wkv",
            "model.layers.3.self_attn.indexer.compressor.wkv",
            "model.layers.3.engram.wkv",
            "model.layers.3.mlp.experts",
            "model.layers.3.mlp.gate",
            "model.embed_tokens",
            "lm_head",
        ]
        for name in wrapped:
            self.assertTrue(Model.should_apply_lora(Model, name), name)
        for name in skipped:
            self.assertFalse(Model.should_apply_lora(Model, name), name)

    def test_shared_expert_checkpoint_names_map_to_lora_targets(self):
        name = Model.get_lora_target_module_name
        prefix = "model.layers.3.mlp.shared_experts"
        self.assertEqual(name(f"{prefix}.w1"), "gate_proj")
        self.assertEqual(name(f"{prefix}.w2"), "down_proj")
        self.assertEqual(name(f"{prefix}.w3"), "up_proj")
        # Routed experts and attention keep their names.
        for kept in ("model.layers.3.mlp.experts.w2", "model.layers.3.self_attn.wq_b"):
            self.assertEqual(name(kept), kept)

    def test_hidden_dims_follow_the_config(self):
        model = SimpleNamespace(config=CONFIG)
        expected = {
            "wq_a": (5120, 1280),
            "wq_b": (1280, 64 * 512),
            "wkv": (5120, 512),
            "wo_a": (64 * 512 // 8, 8 * 1024),
            "wo_b": (8 * 1024, 5120),
            "gate_up_proj": (5120, 2 * 2304),
            "down_proj": (2304, 5120),
        }
        for module, dims in expected.items():
            self.assertEqual(Model.get_hidden_dim(model, module, 0), dims, module)


class TestDecoderReplayUnderLoRA(CustomTestCase):
    """Decoder SWA bounded replay runs the late layers over a per-request tail,
    which the LoRA batch info does not describe, so LoRA turns it off."""

    def _validate(self, **server_args):
        args = ServerArgs(
            model_path="local-deepseek-v41",
            attention_backend="dsv4",
            # Bounded replay needs eager or breakable prefill; use eager.
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.DISABLED)
            ),
            **server_args,
        )
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41")
        )
        with patch(
            "sglang.srt.layers.attention.dsv4.unified_kv_gate.is_unified_kv_triton",
            return_value=False,
        ):
            deepseek_v4_hook.validate_deepseek_v41_features(args)
        return resolving_view(args).enable_decoder_swa_bounded_replay

    @override_platform(is_cuda=True, is_hip=False)
    def test_lora_resolves_replay_off(self):
        self.assertTrue(self._validate())
        self.assertFalse(self._validate(enable_lora=True))
        self.assertFalse(self._validate(lora_paths=["adapter=/path/to/adapter"]))

    @override_platform(is_cuda=True, is_hip=False)
    def test_explicit_replay_with_lora_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "LoRA"):
            self._validate(enable_lora=True, enable_decoder_swa_bounded_replay=True)


if __name__ == "__main__":
    unittest.main()
