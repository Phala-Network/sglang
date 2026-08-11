"""Regression coverage for PR #30642's DSA padding alignment."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.dsa import utils as dsa_utils

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _PaddingMode:
    def __init__(self, max_len: bool):
        self._max_len = max_len

    def is_max_len(self) -> bool:
        return self._max_len


class TestDsaPaddedTokensAlignment(CustomTestCase):
    def _calculate(
        self,
        global_num_tokens,
        *,
        attn_tp_size=8,
        attn_cp_size=1,
        cp_align_size=1,
        max_len=False,
        attn_dp_rank=0,
        round_robin_split=False,
    ):
        parallel = SimpleNamespace(
            attn_tp_size=attn_tp_size,
            attn_cp_size=attn_cp_size,
            attn_cp_rank=0,
            attn_dp_rank=attn_dp_rank,
        )
        forward_batch = SimpleNamespace(
            global_num_tokens_cpu=list(global_num_tokens),
            dp_padding_mode=_PaddingMode(max_len),
            is_extend_in_batch=True,
        )

        with (
            patch.object(dsa_utils, "get_parallel", return_value=parallel),
            patch.object(
                dsa_utils,
                "can_dsa_prefill_cp_round_robin_split",
                return_value=round_robin_split,
            ),
            patch(
                "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
                return_value=cp_align_size,
            ),
            patch(
                "sglang.srt.layers.cp.utils.is_cp_v2_active",
                return_value=False,
            ),
        ):
            return dsa_utils.cal_padded_tokens(forward_batch)

    def test_attention_tp_alignment_is_applied_without_cp_padding(self):
        self.assertEqual(self._calculate([5]), 8)

    def test_attention_tp_alignment_precedes_cp_alignment(self):
        self.assertEqual(self._calculate([9], cp_align_size=6), 18)

    def test_max_len_mode_uses_attention_tp_aligned_lengths(self):
        self.assertEqual(self._calculate([3, 9], max_len=True), 16)

    def test_cp8_non_divisible_padding_matches_rank_local_rows(self):
        self.assertEqual(
            self._calculate(
                [9],
                attn_tp_size=1,
                attn_cp_size=8,
                cp_align_size=8,
                round_robin_split=True,
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
