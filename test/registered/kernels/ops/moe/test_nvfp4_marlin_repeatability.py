"""NVFP4 MoE numerical and CUDA-graph regression at Nemotron expert geometry."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_nvfp4_layer_for_marlin,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import make_nvfp4_weight_and_ref

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.fixture(scope="module")
def nvfp4_layer():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper NVFP4 atomic-reduction regression")
    torch.manual_seed(97)
    layer = torch.nn.Module()
    layer.quant_config = SimpleNamespace(group_size=16)
    layer.moe_runner_config = SimpleNamespace(is_gated=False)
    layer.params_dtype = torch.bfloat16
    layer.intermediate_size_per_partition = 1856
    references = {}
    for prefix, n, k in [("w13", 1856, 2688), ("w2", 2688, 1856)]:
        parts = [make_nvfp4_weight_and_ref(n, k, torch.bfloat16) for _ in range(128)]
        for index, suffix in [
            (0, "weight"),
            (1, "weight_scale"),
            (2, "weight_scale_2"),
        ]:
            setattr(
                layer,
                prefix + "_" + suffix,
                torch.nn.Parameter(
                    torch.stack([p[index] for p in parts]), requires_grad=False
                ),
            )
        references[prefix] = torch.stack([p[3] for p in parts])
    prepare_moe_nvfp4_layer_for_marlin(layer)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield layer, references
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.mark.parametrize("tokens", [1, 6, 81, 337])
def test_nvfp4_moe_repeatability_and_reference(nvfp4_layer, tokens):
    layer, refs = nvfp4_layer
    torch.manual_seed(97 + tokens)
    hidden = torch.randn(tokens, 2688, device="cuda", dtype=torch.bfloat16) / 20
    router = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    weights, ids = torch.topk(torch.softmax(router, dim=-1, dtype=torch.float32), 6)
    ids = ids.to(torch.int32)

    def run():
        return fused_marlin_moe(
            hidden_states=hidden,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            gating_output=router,
            topk_weights=weights,
            topk_ids=ids,
            w1_global_scale=layer.w13_weight_scale_2,
            w2_global_scale=layer.w2_weight_scale_2,
            workspace=layer.workspace,
            num_bits=4,
            routed_scaling_factor=1.0,
            activation="relu2",
            is_gated=False,
        )

    first = run().clone()
    for _ in range(12):
        assert torch.equal(run(), first)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for _ in range(10):
        graph.replay()
        assert torch.equal(result, first)
    assert torch.isfinite(first).all()

    # Compare with the quantized weights' dequantized reference, preserving
    # BF16 stage boundaries. This checks numerical correctness as well as
    # repeatability; a deterministic wrong result must not pass.
    reference = torch.zeros(tokens, 2688, device="cuda", dtype=torch.float32)
    for expert in range(128):
        locations = torch.nonzero(ids == expert)
        if locations.numel() == 0:
            continue
        rows, ranks = locations[:, 0], locations[:, 1]
        stage1 = (hidden[rows].float() @ refs["w13"][expert].float().T).to(hidden.dtype)
        activated = torch.relu(stage1).square()
        stage2 = (activated.float() @ refs["w2"][expert].float().T).to(hidden.dtype)
        reference.index_add_(0, rows, stage2.float() * weights[rows, ranks, None])
    relative_error = (first.float() - reference).norm() / reference.norm()
    assert relative_error < 0.02
