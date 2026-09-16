"""Keep the BF16 Q8_0 prefill path bounded and preserve decode dispatch."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from gguf import GGMLQuantizationType as WeightType

from sglang.srt.layers.quantization import gguf as impl


class TestGGUFPrefillDispatch(unittest.TestCase):
    def test_supported_geometry_boundary(self):
        for rows, expected in ((0, False), (1, False), (20, False), (127, False), (128, True), (16384, True)):
            x = SimpleNamespace(shape=(rows, 5120), dtype=torch.bfloat16, device=SimpleNamespace(type="cuda"))
            self.assertEqual(impl._use_bf16_gguf_prefill(x, WeightType.Q8_0), expected)
        for dtype, device, quant in ((torch.float16, "cuda", WeightType.Q8_0),
                                     (torch.bfloat16, "cpu", WeightType.Q8_0),
                                     (torch.bfloat16, "cuda", WeightType.Q4_K)):
            x = SimpleNamespace(shape=(128, 5120), dtype=dtype, device=SimpleNamespace(type=device))
            self.assertFalse(impl._use_bf16_gguf_prefill(x, quant))

    def test_prefill_multiplies_original_dequantized_checkpoint(self):
        x = torch.tensor([[1., 2.], [-3., 4.]])
        weight = torch.tensor([[2., 3.], [-1., 5.], [7., -2.]])
        packed = torch.zeros(3, 34, dtype=torch.uint8)
        with patch.object(impl, "_use_bf16_gguf_prefill", return_value=True), \
                patch.object(impl, "dequantize_gguf_weight", return_value=weight) as dequant, \
                patch.object(impl, "ggml_mul_mat_a8") as mmq:
            output = impl.fused_mul_mat_gguf(x, packed, WeightType.Q8_0)
            torch.testing.assert_close(output, torch.tensor([[8., 9., 3.], [6., 23., -29.]]))
            dequant.assert_called_once_with(packed, WeightType.Q8_0, x.dtype)
            mmq.assert_not_called()

    def test_decode_and_small_batches_retain_original_kernels(self):
        packed = torch.zeros(8192, 34, dtype=torch.uint8)
        for rows, expected_name in ((1, "ggml_mul_mat_vec_a8"), (20, "ggml_mul_mat_a8")):
            x = torch.zeros(rows, 32)
            expected = torch.ones(rows, 8192)
            with patch.object(impl, "_use_bf16_gguf_prefill", return_value=False), \
                    patch.object(impl, expected_name, return_value=expected) as kernel, \
                    patch.object(impl, "dequantize_gguf_weight") as dequant:
                self.assertIs(impl.fused_mul_mat_gguf(x, packed, WeightType.Q8_0), expected)
                kernel.assert_called_once()
                dequant.assert_not_called()

    def test_empty_and_unquantized_do_not_dispatch_quantized_kernels(self):
        with patch.object(impl, "dequantize_gguf_weight") as dequant:
            self.assertEqual(tuple(impl.fused_mul_mat_gguf(torch.zeros(0, 32), torch.zeros(8, 34), WeightType.Q8_0).shape), (0, 8))
            torch.testing.assert_close(impl.fused_mul_mat_gguf(torch.ones(2, 3), torch.ones(4, 3), WeightType.BF16), torch.full((2, 4), 3.))
            dequant.assert_not_called()


if __name__ == "__main__":
    unittest.main()
