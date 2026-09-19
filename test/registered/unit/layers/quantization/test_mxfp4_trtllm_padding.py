"""A TP-sharded MXFP4 trtllm-gen MoE whose per-rank intermediate size needs
padding must sum to the unsharded experts' output."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe as mxfp4
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def make_layer(weights):
    layer = torch.nn.Module()
    names = (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        "w2_weight_scale_inv",
    )
    for name, tensor in zip(names, weights):
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    layer.num_experts = weights[0].shape[0]
    layer.num_local_experts = layer.num_experts
    layer.moe_ep_rank = 0
    return layer


def make_weights(intermediate, hidden=256, device="cpu"):
    experts = 8

    def fp4_packed(*shape):
        return torch.randint(-128, 128, shape, dtype=torch.int8, device=device)

    def e8m0_scales(*shape):
        return torch.randint(-6, -3, shape, device=device).float().exp2()

    return (
        fp4_packed(experts, 2 * intermediate, hidden // 2),
        fp4_packed(experts, hidden, intermediate // 2),
        e8m0_scales(experts, 2 * intermediate, hidden // 32),
        e8m0_scales(experts, hidden, intermediate // 32),
    )


class TestMxfp4TrtllmPadding(CustomTestCase):
    def test_cpu_weight_load_pads_before_reorder(self):
        """Exercise the real caller and padding helper, stopping before GPU shuffle."""
        weights = make_weights(576)
        layer = make_layer([tensor.clone() for tensor in weights])
        layer.intermediate_size_per_partition = 576
        method = object.__new__(mxfp4.Mxfp4FlashinferTrtllmMoEMethod)
        method._fp8 = Mock()

        class ReorderReached(Exception):
            pass

        def check_reorder(weight, scales):
            self.assertEqual(layer.intermediate_size_per_partition, 640)
            self.assertEqual(weight.shape, (8, 1280, 128))
            self.assertEqual(scales.shape, (8, 1280, 8))
            self.assertEqual(layer.w2_weight.shape, (8, 256, 320))
            self.assertEqual(layer.w2_weight_scale_inv.shape, (8, 256, 20))
            torch.testing.assert_close(weight[:, :576], weights[0][:, :576])
            torch.testing.assert_close(weight[:, 640:1216], weights[0][:, 576:])
            torch.testing.assert_close(scales[:, :576], weights[2][:, :576])
            torch.testing.assert_close(scales[:, 640:1216], weights[2][:, 576:])
            self.assertEqual(torch.count_nonzero(weight[:, 576:640]).item(), 0)
            self.assertEqual(torch.count_nonzero(weight[:, 1216:]).item(), 0)
            self.assertTrue(torch.all(scales[:, 576:640] == 1).item())
            self.assertTrue(torch.all(scales[:, 1216:] == 1).item())
            torch.testing.assert_close(layer.w2_weight[..., :288], weights[1])
            torch.testing.assert_close(layer.w2_weight_scale_inv[..., :18], weights[3])
            self.assertEqual(torch.count_nonzero(layer.w2_weight[..., 288:]).item(), 0)
            self.assertTrue(torch.all(layer.w2_weight_scale_inv[..., 18:] == 1).item())
            raise ReorderReached

        with (
            patch(
                "sglang.srt.layers.quantization.utils.reorder_w1w3_to_w3w1",
                side_effect=check_reorder,
            ),
            patch.object(mxfp4, "print_warning_once"),
            self.assertRaises(ReorderReached),
        ):
            method.process_weights_after_loading(layer)
        method._fp8.process_weights_after_loading.assert_called_once_with(layer)

    def test_cpu_aligned_padding_is_identity(self):
        layer = make_layer(make_weights(640))
        layer.intermediate_size_per_partition = 640
        before = {name: value.data_ptr() for name, value in layer.named_parameters()}
        mxfp4._pad_intermediate_size(layer)
        self.assertEqual(
            before, {name: value.data_ptr() for name, value in layer.named_parameters()}
        )
        self.assertEqual(layer.intermediate_size_per_partition, 640)

    def test_cpu_mega_moe_does_not_enter_padding(self):
        method = object.__new__(mxfp4.Mxfp4FlashinferTrtllmMoEMethod)
        method._fp8 = Mock()
        layer = SimpleNamespace(_mega_moe_weights_built=True)
        with patch.object(mxfp4, "_pad_intermediate_size") as pad:
            method.process_weights_after_loading(layer)
        method._fp8.process_weights_after_loading.assert_called_once_with(layer)
        pad.assert_not_called()

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
        "Requires Blackwell",
    )
    def test_tp4_matches_unsharded_experts(self):
        torch.manual_seed(42)
        weights = make_weights(2304, hidden=5120, device="cuda")

        def prepare(tensors):
            layer = make_layer(tensors)
            method = object.__new__(mxfp4.Mxfp4FlashinferTrtllmMoEMethod)
            method._fp8 = Mock()
            method.prefix = "test.experts"
            method.flashinfer_mxfp4_moe_precision = "default"
            method.process_weights_after_loading(layer)
            method.create_moe_runner(layer, SimpleNamespace(swiglu_limit=10.0))
            return method, layer

        full = prepare([tensor.clone() for tensor in weights])
        shards = []
        for rank in range(4):
            start = rank * 576
            end = start + 576
            w13, w2, s13, s2 = weights
            shard = (
                torch.cat(
                    (w13[:, start:end], w13[:, 2304 + start : 2304 + end]), dim=1
                ),
                w2[..., start // 2 : end // 2].contiguous(),
                torch.cat(
                    (s13[:, start:end], s13[:, 2304 + start : 2304 + end]), dim=1
                ),
                s2[..., start // 32 : end // 32].contiguous(),
            )
            shards.append(prepare(shard))

        with (
            patch.object(mxfp4, "get_tp_group", return_value=None),
            patch.object(mxfp4, "is_allocation_symmetric", return_value=False),
            patch.object(mxfp4, "use_symmetric_memory", return_value=nullcontext()),
        ):
            for tokens in (1, 64):
                with self.subTest(tokens=tokens):
                    x = torch.randn(tokens, 5120, dtype=torch.bfloat16, device="cuda")
                    logits = torch.randn(tokens, 8, device="cuda")
                    scores, ids = logits.softmax(-1).topk(6, dim=-1)
                    topk = StandardTopKOutput(scores, ids.to(torch.int32), logits)
                    dispatch = StandardDispatchOutput(x, None, topk)
                    reference = full[0].apply(full[1], dispatch).hidden_states.float()
                    actual = sum(
                        method.apply(layer, dispatch).hidden_states.float()
                        for method, layer in shards
                    )
                    rmse = torch.linalg.norm(actual - reference) / torch.linalg.norm(
                        reference
                    )
                    self.assertLess(rmse.item(), 0.01)


if __name__ == "__main__":
    unittest.main()
