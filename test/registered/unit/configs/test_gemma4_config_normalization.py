"""Gemma4 config translation from Transformers' per-layer representation."""

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from transformers import Gemma4Config, Gemma4TextConfig

from sglang.srt.utils.hf_transformers.config import (
    HfModelConfigParser,
    _normalize_gemma4_text_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGemma4ConfigNormalization(CustomTestCase):
    def make_config(self, **kwargs):
        return Gemma4TextConfig(
            num_hidden_layers=6,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            head_dim=256,
            num_key_value_heads=8,
            attention_k_eq_v=True,
            global_head_dim=512,
            num_global_key_value_heads=2,
            **kwargs,
        )

    def test_converts_heterogeneous_full_and_sliding_dimensions(self):
        config = self.make_config()
        with self.assertRaises(RuntimeError):
            _ = config.head_dim

        _normalize_gemma4_text_config(config)

        self.assertEqual((config.head_dim, config.num_key_value_heads), (512, 2))
        self.assertEqual(
            (config.swa_head_dim, config.swa_num_key_value_heads), (256, 8)
        )
        self.assertEqual(config.per_layer_attributes, set())
        restored = Gemma4TextConfig.from_dict(config.to_dict())
        self.assertEqual((restored.head_dim, restored.num_key_value_heads), (512, 2))
        self.assertEqual(
            (restored.swa_head_dim, restored.swa_num_key_value_heads), (256, 8)
        )

    def test_rejects_inconsistent_full_attention_layers(self):
        config = self.make_config(
            per_layer_config={
                2: {"head_dim": 512, "num_key_value_heads": 2},
                5: {"head_dim": 768, "num_key_value_heads": 2},
            }
        )
        with self.assertRaisesRegex(ValueError, "inconsistent per-layer"):
            _normalize_gemma4_text_config(config)

    def test_rejects_unrepresented_per_layer_attributes(self):
        config = self.make_config(per_layer_config={2: {"intermediate_size": 10240}})
        with self.assertRaisesRegex(ValueError, "unsupported per-layer"):
            _normalize_gemma4_text_config(config)

    def test_legacy_global_fields(self):
        config = SimpleNamespace(
            head_dim=256,
            num_key_value_heads=8,
            global_head_dim=512,
            num_global_key_value_heads=2,
        )
        _normalize_gemma4_text_config(config)
        self.assertEqual((config.head_dim, config.num_key_value_heads), (512, 2))
        self.assertEqual(
            (config.swa_head_dim, config.swa_num_key_value_heads), (256, 8)
        )

    def test_parser_loads_gemma4_checkpoint_config(self):
        root = Gemma4Config(text_config=self.make_config())
        root.architectures = ["Gemma4ForConditionalGeneration"]
        with TemporaryDirectory() as model_dir:
            root.save_pretrained(model_dir)
            parsed = HfModelConfigParser().parse(model_dir, trust_remote_code=False)

        self.assertEqual(
            (parsed.text_config.head_dim, parsed.text_config.num_key_value_heads),
            (512, 2),
        )
        self.assertEqual(
            (
                parsed.text_config.swa_head_dim,
                parsed.text_config.swa_num_key_value_heads,
            ),
            (256, 8),
        )


if __name__ == "__main__":
    unittest.main()
