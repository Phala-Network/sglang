"""Packed mask contract used by llguidance strict-reasoning wrappers."""

import unittest

import torch

from sglang.srt.constrained.llguidance_backend import GuidanceBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLLGuidanceTokenFilter(unittest.TestCase):
    @staticmethod
    def allowed(row):
        return {
            i for i in range(row.numel() * 32) if (int(row[i // 32]) >> (i % 32)) & 1
        }

    def test_allow_sign_bit_duplicates_and_other_rows(self):
        mask = torch.full((3, 3), -1, dtype=torch.int32)
        GuidanceBackend.set_token_filter(mask, [0, 30, 31, 31, 32, 63, 95], 1)
        self.assertEqual(self.allowed(mask[1]), {0, 30, 31, 32, 63, 95})
        self.assertTrue(torch.all(mask[0] == -1))
        self.assertTrue(torch.all(mask[2] == -1))

    def test_deny_resets_row_and_clears_all_duplicate_bits(self):
        mask = torch.zeros((2, 3), dtype=torch.int32)
        blocked = {0, 31, 32, 63, 95}
        GuidanceBackend.set_token_filter(mask, [*blocked, 31], 1, is_allowed=False)
        self.assertEqual(self.allowed(mask[1]), set(range(96)) - blocked)
        self.assertTrue(torch.all(mask[0] == 0))

    def test_incremental_allow_and_deny_preserve_unrelated_bits(self):
        mask = torch.zeros((2, 3), dtype=torch.int32)
        GuidanceBackend.set_token_filter(mask, [1, 31, 32], 0)
        GuidanceBackend.set_token_filter(mask, [63, 95], 0, reset_vocab_mask=False)
        GuidanceBackend.set_token_filter(
            mask, [31, 95], 0, is_allowed=False, reset_vocab_mask=False
        )
        self.assertEqual(self.allowed(mask[0]), {1, 32, 63})
        self.assertTrue(torch.all(mask[1] == 0))

    def test_empty_filters_and_reset_semantics(self):
        mask = torch.full((2, 2), 7, dtype=torch.int32)
        GuidanceBackend.set_token_filter(mask, [], 0)
        self.assertTrue(torch.all(mask[0] == 0))
        GuidanceBackend.set_token_filter(mask, [], 0, is_allowed=False)
        self.assertTrue(torch.all(mask[0] == -1))
        GuidanceBackend.set_token_filter(mask, [], 1, reset_vocab_mask=False)
        self.assertTrue(torch.all(mask[1] == 7))

    def test_apply_filter_to_actual_logits(self):
        mask = torch.zeros((2, 3), dtype=torch.int32)
        GuidanceBackend.set_token_filter(mask, [31, 32, 95], 0)
        GuidanceBackend.set_token_filter(mask, [1, 63], 1)
        logits = torch.ones((2, 96), dtype=torch.float32)
        GuidanceBackend.apply_vocab_mask(logits, mask)
        self.assertEqual(
            set(torch.where(torch.isfinite(logits[0]))[0].tolist()), {31, 32, 95}
        )
        self.assertEqual(
            set(torch.where(torch.isfinite(logits[1]))[0].tolist()), {1, 63}
        )


if __name__ == "__main__":
    unittest.main()
