import pytest
import torch

from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    _make_deferred_finalize_output,
)


def make_result(expert_weights: torch.Tensor):
    rows, top_k = expert_weights.shape
    gemm2_out = torch.arange(rows * 4, dtype=torch.bfloat16).reshape(rows, 4)
    expanded_idx = torch.arange(rows * top_k, dtype=torch.int32)
    return gemm2_out, expert_weights, expanded_idx


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_deferred_finalize_preserves_declared_expert_weight_values(dtype):
    weights = torch.tensor(
        [[0.25657165, 0.24222815, 0.14130422], [0.5, 0.25, 0.125]],
        dtype=dtype,
    )

    deferred = _make_deferred_finalize_output(
        make_result(weights),
        top_k=weights.shape[1],
        expected_expert_weights_dtype=dtype,
    )

    assert deferred.expert_weights is weights
    assert deferred.expert_weights.dtype == dtype
    assert torch.isfinite(deferred.expert_weights).all()
    torch.testing.assert_close(deferred.expert_weights, weights, rtol=0, atol=0)


def test_deferred_finalize_does_not_reinterpret_genuine_fp32_weights():
    weights = torch.tensor([[0.25657165, 0.24222815, 0.14130422]], dtype=torch.float32)
    reinterpreted = weights.view(torch.bfloat16).view(-1, weights.shape[1])[:1]
    assert reinterpreted.dtype == torch.bfloat16
    assert not torch.equal(reinterpreted.float(), weights)

    deferred = _make_deferred_finalize_output(
        make_result(weights),
        top_k=weights.shape[1],
        expected_expert_weights_dtype=torch.float32,
    )

    assert deferred.expert_weights is weights
    torch.testing.assert_close(deferred.expert_weights, weights, rtol=0, atol=0)


def test_deferred_finalize_rejects_route_abi_dtype_mismatch():
    weights = torch.tensor([[0.5, 0.25]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="expert weights dtype mismatch"):
        _make_deferred_finalize_output(
            make_result(weights),
            top_k=weights.shape[1],
            expected_expert_weights_dtype=torch.bfloat16,
        )


def test_deferred_finalize_rejects_unsupported_declared_dtype():
    weights = torch.tensor([[0.5, 0.25]], dtype=torch.float16)

    with pytest.raises(RuntimeError, match="only supports BF16 or FP32"):
        _make_deferred_finalize_output(
            make_result(weights),
            top_k=weights.shape[1],
            expected_expert_weights_dtype=torch.float16,
        )
