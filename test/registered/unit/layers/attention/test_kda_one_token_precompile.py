"""K3.1's init precompile runs the Triton extend exactly as a one-token batch does.

Each autotuned kernel compiles all its configs on a new key, and FlashInfer's packed
prefill leaves one-token batches (every /health) to this path. So the precompile only
helps if its keys match the serving ones. The expected values come from the Triton
disk cache of the K3 image, which has the same 12-heads-per-rank KDA at TP8:
`/jit/triton/*/chunk_kda_fwd_kernel_inter_solve_fused.autotune.json` and the kernels'
TTIR signatures. The kernels are stubbed; the Python around them is the real one.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.configs.model_config import is_kimi_k31
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BF16, F32, I32, I64 = "bf16", "f32", "i32", "i64"
_NAMES = {
    torch.bfloat16: BF16,
    torch.float32: F32,
    torch.int32: I32,
    torch.int64: I64,
}
# TP8: 96 heads over 8 ranks; [slots, H, V, K]
H, V, K = 12, 128, 128
LOWER_BOUND = -5.0

FUSED_AUTOTUNE_KEY = (
    (H, K, 16, V, True, True),
    {
        "q": BF16,
        "k": BF16,
        "g": F32,
        "beta": F32,
        "Aqk": BF16,
        "Akkd": F32,
        "Akk": BF16,
        "v_in": BF16,
        "w_out": BF16,
        "u_out": BF16,
        "kg_out": BF16,
        "cu_seqlens": I32,
        "chunk_indices": I32,
    },
)
GATE_DTYPES = {
    "s": BF16,
    "A_log": F32,
    "dt_bias": F32,
    "o": F32,
    "cu_seqlens": I32,
    "chunk_indices": I32,
    "beta": BF16,
    "beta_out": F32,
}
H_KERNEL_DTYPES = {
    "k": BF16,
    "v": BF16,
    "w": BF16,
    "v_new": BF16,
    "gk": F32,
    "h": BF16,
    "initial_state": F32,
    "initial_state_indices": I32,
    "cu_seqlens": I32,
    "chunk_offsets": I64,
}
O_KERNEL_DTYPES = {
    "q": BF16,
    "v": BF16,
    "g": F32,
    "h": BF16,
    "o": BF16,
    "A": BF16,
    "cu_seqlens": I32,
    "chunk_indices": I32,
}


class _Kernel:
    """`kernel[grid](**kwargs)` that records the call instead of launching."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda **kwargs: self.calls.append(kwargs)


def _dtypes(call):
    return {
        name: _NAMES[value.dtype]
        for name, value in call.items()
        if isinstance(value, torch.Tensor)
    }


class TestOneTokenPrecompile(CustomTestCase):
    def _run(self, max_seqs=20):
        fla = "sglang.kernels.ops.attention.fla"
        self.kernels = {
            name: _Kernel()
            for name in (
                f"{fla}.chunk_intra.chunk_kda_fwd_kernel_inter_solve_fused",
                f"{fla}.chunk_intra.chunk_kda_fwd_kernel_intra_sub_chunk",
                f"{fla}.chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
                f"{fla}.kda.kda_gate_chunk_cumsum_vector_kernel",
                f"{fla}.kda.chunk_gla_fwd_kernel_o",
                f"{fla}.kda.recompute_w_u_fwd_kernel",
            )
        }
        patches = [patch(name, kernel) for name, kernel in self.kernels.items()]
        patches += [
            patch(f"{fla}.kda.l2norm_fwd", side_effect=lambda x: x),
            patch(f"{fla}.kda._get_k3_recompute_w_u_config", return_value=None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.state = torch.full((3, 21, H, V, K), 7.0, dtype=torch.float32)
        TritonKDAKernel().precompile_one_token_extend(
            state=self.state,
            qkv_dtype=torch.bfloat16,
            lower_bound=LOWER_BOUND,
            max_seqs=max_seqs,
        )

    def _calls(self, suffix):
        return next(v.calls for k, v in self.kernels.items() if k.endswith(suffix))

    def test_autotune_key_is_the_serving_one(self):
        self._run()
        (call,) = self._calls("inter_solve_fused")
        constexprs, dtypes = FUSED_AUTOTUNE_KEY
        self.assertEqual(
            tuple(
                call[name]
                for name in ("H", "K", "BC", "V", "FUSE_RECOMPUTE", "FUSE_DIAGONAL")
            ),
            constexprs,
        )
        self.assertEqual(_dtypes(call), dtypes)
        self.assertTrue(call["USE_SAFE_GATE"])

    def test_other_kernels_take_the_serving_dtypes(self):
        self._run()
        (gate,) = self._calls("gate_chunk_cumsum_vector_kernel")
        self.assertEqual(_dtypes(gate), GATE_DTYPES)
        self.assertEqual(gate["lower_bound"], LOWER_BOUND)
        (h,) = self._calls("h_blockdim64")
        self.assertEqual(_dtypes(h), H_KERNEL_DTYPES)
        (o,) = self._calls("chunk_gla_fwd_kernel_o")
        self.assertEqual(_dtypes(o), O_KERNEL_DTYPES)

    def test_small_batches_never_reach_the_unfused_grid(self):
        # 21 sequences * 12 heads = 252 CTAs; K3.1's 0.46 mamba ratio runs 20.
        self._run(max_seqs=21)
        self.assertEqual(len(self._calls("inter_solve_fused")), 1)
        self.assertEqual(self._calls("intra_sub_chunk"), [])

    def test_larger_batches_also_compile_the_unfused_grid(self):
        self._run(max_seqs=64)
        fused, unfused = self._calls("inter_solve_fused")
        self.assertTrue(fused["FUSE_RECOMPUTE"] and fused["FUSE_DIAGONAL"])
        self.assertFalse(unfused["FUSE_RECOMPUTE"] or unfused["FUSE_DIAGONAL"])
        self.assertEqual(unfused["V"], 0)
        self.assertEqual(len(self._calls("intra_sub_chunk")), 1)
        self.assertEqual(len(self._calls("recompute_w_u_fwd_kernel")), 1)

    def test_pool_state_is_only_read_for_its_shape(self):
        self._run()
        self.assertTrue(torch.all(self.state == 7.0))
        (h,) = self._calls("h_blockdim64")
        self.assertEqual(h["initial_state"].shape, (1, H, V, K))
        self.assertTrue(torch.all(h["initial_state"] == 0))

    def test_gate_stays_off_k3(self):
        k3 = {"architectures": ["KimiK3ForConditionalGeneration"]}
        k31 = {"architectures": ["KimiK31ForConditionalGeneration"], "index_topk": 2048}
        self.assertFalse(is_kimi_k31(k3))
        self.assertTrue(is_kimi_k31(k31))


if __name__ == "__main__":
    unittest.main()
