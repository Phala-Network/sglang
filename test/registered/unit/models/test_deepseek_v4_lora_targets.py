"""DeepSeek-V4 LoRA module-name gating and target dims, without a model instance."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.lora.layers import ColumnParallelLinearWithLoRA
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM as Model
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ADAPTER = ["adapter=/path/to/adapter"]
# Ways to request LoRA: --lora-paths alone becomes --enable-lora only later,
# in check_lora_server_args, so every guard must also catch it.
LORA_REQUESTS = {
    "enable_lora": {"enable_lora": True},
    "lora_paths": {"lora_paths": ADAPTER},
    "both": {"enable_lora": True, "lora_paths": ADAPTER},
}
# --enable-lora false with paths keeps LoRA off (the paths are ignored).
LORA_DISABLED = {"enable_lora": False, "lora_paths": ADAPTER}


def _server_args(**server_args):
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
    return args


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
        args = _server_args(**server_args)
        with patch(
            "sglang.srt.layers.attention.dsv4.unified_kv_gate.is_unified_kv_triton",
            return_value=False,
        ):
            deepseek_v4_hook.validate_deepseek_v41_features(args)
        return resolving_view(args).enable_decoder_swa_bounded_replay

    @override_platform(is_cuda=True, is_hip=False)
    def test_lora_resolves_replay_off(self):
        self.assertTrue(self._validate())
        self.assertTrue(self._validate(**LORA_DISABLED))
        for name, lora in LORA_REQUESTS.items():
            with self.subTest(name):
                self.assertFalse(self._validate(**lora))

    @override_platform(is_cuda=True, is_hip=False)
    def test_explicit_replay_with_lora_is_rejected(self):
        for name, lora in LORA_REQUESTS.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, "LoRA"):
                self._validate(enable_decoder_swa_bounded_replay=True, **lora)


class TestMegaAttentionUnderLoRA(CustomTestCase):
    """MegaAttention reads wq_b/wo_a directly, bypassing LoRA wrappers."""

    def _default_backend(self, **server_args):
        args = _server_args(
            device="cuda", tp_size=1, kv_cache_dtype="fp8_e4m3", **server_args
        )
        with patch("sglang.srt.utils.common.is_sm100_supported", return_value=True):
            return deepseek_v4_hook._default_dsv4_attention_backend(args)

    def test_lora_never_defaults_to_mega(self):
        self.assertEqual(self._default_backend(), "flashmla_mega")
        self.assertEqual(self._default_backend(**LORA_DISABLED), "flashmla_mega")
        for name, lora in LORA_REQUESTS.items():
            with self.subTest(name):
                self.assertEqual(self._default_backend(**lora), "flashmla")

    def test_explicit_mega_with_lora_raises_value_error(self):
        for name, lora in LORA_REQUESTS.items():
            args = _server_args(
                device="cuda", tp_size=1, dsv4_attn_backend="flashmla_mega", **lora
            )
            with (
                self.subTest(name),
                patch("sglang.srt.utils.common.is_sm100_supported", return_value=True),
                # only the backend checks are under test, not the KV dtype pass
                patch.object(deepseek_v4_hook, "run_post_process_pass"),
                self.assertRaisesRegex(
                    ValueError, "flashmla_mega does not support LoRA"
                ),
            ):
                deepseek_v4_hook.apply_deepseek_v4_defaults(
                    args, "DeepseekV4ForCausalLM"
                )


class TestCpDecodeAttnTpUnderLoRA(CustomTestCase):
    """Decode attention TP slices the raw linears, not a LoRA wrapper."""

    def test_rejected_for_every_lora_request(self):
        deepseek_v4_hook.validate_deepseek_v4_cp(
            _server_args(enable_cp_decode_attn_tp=True, **LORA_DISABLED)
        )
        for name, lora in LORA_REQUESTS.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(ValueError, "does not support LoRA"),
            ):
                deepseek_v4_hook.validate_deepseek_v4_cp(
                    _server_args(enable_cp_decode_attn_tp=True, **lora)
                )


class TestWoAFusedQuantUnderLoRA(CustomTestCase):
    def test_wrapped_wo_a_never_takes_the_fused_quant(self):
        layer = SimpleNamespace(wo_a=MagicMock(spec=ColumnParallelLinearWithLoRA))
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: True, is_target_verify=lambda: False
            )
        )
        self.assertIsNone(MQALayer._wo_a_quant_freqs(layer, batch, MagicMock()))


if __name__ == "__main__":
    unittest.main()
