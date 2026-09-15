"""CPU regression tests for llama.cpp Qwen3.5 checkpoint conventions."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.model_loader.gguf_name_maps import (
    Qwen3_5GGUFWeightTransform,
    apply_gguf_weight_transform,
    get_gguf_weight_transform,
)


class TestQwen35GGUFTransform(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            model_type="qwen3_5_text",
            linear_num_key_heads=2,
            linear_num_value_heads=6,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
        )
        self.transform = Qwen3_5GGUFWeightTransform(self.config)
        self.head_order = [0, 2, 4, 1, 3, 5]

    def test_state_conversion_and_gqa_order(self):
        hf_log = torch.arange(6, dtype=torch.float32) / 4
        gguf_log = hf_log.reshape(2, 3).T.reshape(-1)
        actual = self.transform("model.layers.0.linear_attn.A_log", -gguf_log.exp(), None)
        torch.testing.assert_close(actual, hf_log)
        actual_bias = self.transform("model.layers.0.linear_attn.dt_bias", gguf_log, None)
        torch.testing.assert_close(actual_bias, hf_log)

    def test_folded_norm_and_gated_norm_are_distinct(self):
        weight = torch.tensor([1.25, 0.75])
        for name in ("model.norm.weight", "model.layers.3.self_attn.q_norm.weight",
                     "model.layers.3.self_attn.k_norm.weight", "model.layers.0.input_layernorm.weight",
                     "model.layers.0.post_attention_layernorm.weight"):
            torch.testing.assert_close(self.transform(name, weight, None), weight - 1)
        self.assertIs(self.transform("model.layers.0.linear_attn.norm.weight", weight, None), weight)

    def test_qkv_and_conv_preserve_key_rows(self):
        weight = torch.arange((128 + 192) * 4).reshape(320, 4)
        expected = torch.cat((weight[:128], weight[128:].reshape(6, 32, 4)[self.head_order].reshape(192, 4)))
        torch.testing.assert_close(self.transform("model.layers.0.linear_attn.in_proj_qkv.qweight", weight, 8), expected)
        torch.testing.assert_close(self.transform("model.layers.0.linear_attn.conv1d.weight", weight, None), expected.unsqueeze(1))

    def test_q8_output_projection_is_exact_byte_permutation(self):
        # Q8_0 uses 34 stored bytes for each 32-element block.
        weight = torch.arange(2 * 6 * 34).remainder(256).to(torch.uint8).reshape(2, 204)
        expected = weight.reshape(2, 6, 34)[:, self.head_order].reshape(2, 204)
        torch.testing.assert_close(self.transform("model.layers.0.linear_attn.out_proj.qweight", weight, 8), expected)

    def test_bf16_embedding_reinterprets_bytes(self):
        original = torch.tensor([[1.5, -2.0, 0.25]], dtype=torch.bfloat16)
        weights = [("model.embed_tokens.qweight_type", torch.tensor(30)),
                   ("model.embed_tokens.qweight", original.view(torch.uint8))]
        converted = list(apply_gguf_weight_transform(weights, self.transform))
        torch.testing.assert_close(converted[1][1], original)

    def test_reject_invalid_decay_and_geometry(self):
        for invalid in (torch.zeros(6), torch.full((6,), -float("inf")), torch.full((6,), float("nan"))):
            with self.assertRaises(ValueError):
                self.transform("model.layers.0.linear_attn.A_log", invalid, None)
        self.config.linear_num_key_heads = 0
        with self.assertRaises(ValueError):
            Qwen3_5GGUFWeightTransform(self.config)

    def test_unaligned_superblock_is_rejected(self):
        # Q6_K spans 256 elements and cannot be permuted as 32-element heads.
        with self.assertRaises(NotImplementedError):
            self.transform("model.layers.0.linear_attn.out_proj.qweight", torch.zeros(2, 210, dtype=torch.uint8), 14)

    def test_other_architectures_unchanged(self):
        self.assertIsNone(get_gguf_weight_transform(SimpleNamespace(model_type="gemma4")))
        self.assertIsNone(get_gguf_weight_transform(SimpleNamespace(model_type="qwen3_next")))

    def test_snapshot_requires_a_single_gguf(self):
        from sglang.srt.model_loader.loader import GGUFModelLoader

        loader = object.__new__(GGUFModelLoader)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                loader._prepare_weights(directory)
            weight = Path(directory) / "target.gguf"
            weight.touch()
            self.assertEqual(loader._prepare_weights(directory), str(weight))
            (Path(directory) / "draft.gguf").touch()
            with self.assertRaises(ValueError):
                loader._prepare_weights(directory)

    def test_packed_type_metadata_broadcasts_to_each_shard(self):
        from sglang.srt.layers.linear import MergedColumnParallelLinear
        from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

        module = SimpleNamespace(output_sizes=[4, 4, 12, 12])
        param = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
        param.is_gguf_weight_type = True
        param.shard_weight_type = {}
        original = lambda p, value, shard: MergedColumnParallelLinear.weight_loader(module, p, value, shard)
        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(module, original)
        loader(param, torch.tensor(8), (0, 1, 2))
        torch.testing.assert_close(param.data, torch.tensor([8., 8., 8., 0.]))
        self.assertEqual(param.shard_weight_type, {0: 8, 1: 8, 2: 8})


if __name__ == "__main__":
    unittest.main()
