"""Verify exact BF16 bytes, temporal convolution parity and fail-closed loading."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gguf
import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.model_loader.gguf_vision import (
    _read_unquantized_tensor,
    qwen35_vision_tensor_map,
    qwen35_vision_weights_iterator,
)


class TestQwen35Vision(unittest.TestCase):
    def setUp(self):
        self.vision = SimpleNamespace(
            temporal_patch_size=2, deepstack_visual_indexes=[], hidden_size=4,
            intermediate_size=8, spatial_merge_size=2, in_channels=3, patch_size=2,
            num_position_embeddings=16, out_hidden_size=12, depth=1, num_heads=2,
        )
        self.config = SimpleNamespace(vision_config=self.vision)
        self.mapping = qwen35_vision_tensor_map(self.config)
        metadata = {
            "general.architecture": "clip", "clip.projector_type": "qwen3vl_merger",
            "clip.vision.block_count": 1, "clip.vision.embedding_length": 4,
            "clip.vision.attention.head_count": 2, "clip.vision.feed_forward_length": 8,
            "clip.vision.patch_size": 2, "clip.vision.spatial_merge_size": 2,
            "clip.vision.projection_dim": 12,
        }
        self.metadata = metadata
        self.tensors = [self.tensor(name, torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape) / 64)
                        for name, (_, shape) in self.mapping.items()]
        self.reader = SimpleNamespace(tensors=self.tensors, get_field=lambda key:
                                      SimpleNamespace(contents=lambda: metadata[key]) if key in metadata else None)

    @staticmethod
    def tensor(name, value):
        bf16 = value.dtype == torch.bfloat16
        return SimpleNamespace(name=name, shape=tuple(reversed(value.shape)),
                               tensor_type=gguf.GGMLQuantizationType.BF16 if bf16 else gguf.GGMLQuantizationType.F32,
                               data=value.view(torch.uint8).numpy() if bf16 else value.numpy())

    def load(self):
        with patch.object(gguf, "GGUFReader", return_value=self.reader):
            return dict(qwen35_vision_weights_iterator("test-mmproj.gguf", self.config))

    def test_temporal_patch_convolution_matches_converter_slices(self):
        torch.manual_seed(63)
        weight = torch.randn(4, 3, 2, 2, 2)
        self.tensors[0] = self.tensor("v.patch_embd.weight", weight[:, :, 0].contiguous())
        self.tensors[1] = self.tensor("v.patch_embd.weight.1", weight[:, :, 1].contiguous())
        loaded = self.load()
        self.assertEqual(len(loaded), 21)
        actual_weight = loaded["visual.patch_embed.proj.weight"]
        torch.testing.assert_close(actual_weight, weight, rtol=0, atol=0)
        inputs = torch.randn(2, 3, 2, 4, 4)
        expected = F.conv2d(inputs[:, :, 0], weight[:, :, 0], stride=2) + F.conv2d(inputs[:, :, 1], weight[:, :, 1], stride=2)
        actual = F.conv3d(inputs, actual_weight, stride=(2, 2, 2)).squeeze(2)
        torch.testing.assert_close(actual, expected)

    def test_bf16_values_preserved(self):
        original = torch.tensor([[1.5, -2, 0.03125], [3.5, -0.5, 64]], dtype=torch.bfloat16)
        actual = _read_unquantized_tensor(self.tensor("bf16", original), (2, 3))
        torch.testing.assert_close(actual, original, rtol=0, atol=0)

    def test_missing_and_extra_tensors_fail(self):
        missing = self.tensors.pop()
        with self.assertRaisesRegex(ValueError, "missing="):
            self.load()
        self.tensors.append(missing)
        self.tensors.append(self.tensor("unexpected", torch.ones(2)))
        with self.assertRaisesRegex(ValueError, "extra="):
            self.load()

    def test_duplicate_and_bad_shape_fail(self):
        self.tensors.append(self.tensors[-1])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.load()
        self.tensors.pop()
        self.tensors[-1].shape = (123,)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.load()

    def test_incompatible_metadata_and_geometry_fail(self):
        self.metadata["clip.projector_type"] = "qwen2vl_merger"
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            self.load()
        self.vision.deepstack_visual_indexes = [0]
        with self.assertRaisesRegex(ValueError, "no deepstack"):
            qwen35_vision_tensor_map(self.config)

    def test_quantized_vision_is_explicitly_rejected(self):
        tensor = self.tensor("quantized", torch.ones(2, 32))
        tensor.tensor_type = gguf.GGMLQuantizationType.Q8_0
        with self.assertRaisesRegex(ValueError, "only F32/F16/BF16"):
            _read_unquantized_tensor(tensor, (2, 32))


if __name__ == "__main__":
    unittest.main()
