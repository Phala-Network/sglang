"""TRTLLMMLABackend's ragged prefill must keep a NaN-poisoned request's values
in its own rows. The kernel computes a sequence's last KV tile over the whole
tile, reading the next sequence's first K/V rows and masking them by zero
(0 * NaN = NaN): the request packed before a NaN one got NaN too."""

import itertools
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEADS, QK_DIM, V_DIM = 8, 192, 128


class TestTRTLLMMLARaggedNaN(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("trtllm-gen MLA prefill requires SM100/SM103")
        cls.backend = SimpleNamespace(
            workspace_buffer=torch.zeros(128 << 20, dtype=torch.uint8, device="cuda"),
        )
        cls.layer = SimpleNamespace(
            scaling=QK_DIM**-0.5, k_scale_float=None, v_scale_float=None
        )

    def _run(self, lens, nan_row, data_type):
        self.backend.data_type = data_type
        total = sum(lens)
        g = torch.Generator(device="cuda").manual_seed(sum(lens))

        def randn(*shape):
            return torch.randn(*shape, generator=g, device="cuda").bfloat16()

        q = randn(total, HEADS, QK_DIM)
        k = randn(total, HEADS, QK_DIM)
        v = randn(total, HEADS, V_DIM)
        start = sum(lens[:nan_row])
        for x in (q, k, v):
            x[start : start + lens[nan_row]] = float("nan")
        cu = torch.tensor(
            [0, *itertools.accumulate(lens)], dtype=torch.int32, device="cuda"
        )
        out = torch.zeros(total, HEADS, V_DIM, dtype=torch.bfloat16, device="cuda")
        TRTLLMMLABackend._run_prefill_kernel(
            self.backend,
            q=q,
            k=k,
            v=v,
            layer=self.layer,
            batch_size=len(lens),
            cum_seq_lens_q=cu,
            max_q_len=max(lens),
            seq_lens_kv=torch.tensor(lens, dtype=torch.int32, device="cuda"),
            cum_seq_lens_kv=cu,
            max_kv_len=max(lens),
            is_causal=True,
            return_lse=False,
            out_buffer=out,
            o_sf_scale=1.0,
        )
        torch.cuda.synchronize()
        return out

    def test_nan_request_stays_in_its_rows(self):
        cases = (
            ([1, 246], 1),
            ([5, 9], 1),
            ([37, 100, 5], 1),
            ([64, 64], 1),
            ([100, 37], 0),
        )
        for data_type in (torch.bfloat16, torch.float8_e4m3fn):
            for lens, nan_row in cases:
                with self.subTest(data_type=data_type, lens=lens, nan_row=nan_row):
                    out = self._run(lens, nan_row, data_type)
                    for row in range(len(lens)):
                        if row == nan_row:
                            continue
                        start = sum(lens[:row])
                        self.assertTrue(
                            bool(torch.isfinite(out[start : start + lens[row]]).all()),
                            f"request {row} got the NaN request's values",
                        )


if __name__ == "__main__":
    unittest.main()
