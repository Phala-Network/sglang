"""`async_h2d` under Confidential Computing: blocking unless asked, and only when small."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SWITCH = "SGLANG_CC_ASYNC_SMALL_H2D"
FORCE_CC = "SGLANG_FORCE_CONFIDENTIAL_COMPUTE"


class TestAsyncH2dUnderCc(unittest.TestCase):
    def setUp(self):
        self.addCleanup(common.is_confidential_compute.cache_clear)

    def _non_blocking(self, env, args=(1, 2, 3)):
        """(non_blocking of the copy, the values it was handed)."""
        seen = {}
        real_to = torch.Tensor.to

        def to(tensor, device, non_blocking=False):
            seen.update(non_blocking=non_blocking, values=tensor.tolist())
            return real_to(tensor, device)

        with (
            patch.dict("os.environ", env),
            patch.object(common, "is_pin_memory_available", lambda device: False),
            patch.object(torch.Tensor, "to", to),
        ):
            # Detected once per process.
            common.is_confidential_compute.cache_clear()
            out = common.async_h2d(list(args), dtype=torch.int64, device="cpu")
        self.assertEqual(out.tolist(), list(args))
        self.assertEqual(seen["values"], list(args))
        return seen["non_blocking"]

    def test_cc_blocks_by_default(self):
        self.assertFalse(self._non_blocking({FORCE_CC: "1"}))

    def test_cc_with_the_switch_does_not(self):
        self.assertTrue(self._non_blocking({FORCE_CC: "1", SWITCH: "1"}))

    def test_the_switch_alone_does_nothing_outside_cc(self):
        self.assertFalse(self._non_blocking({FORCE_CC: "0", SWITCH: "1"}))

    def test_a_large_copy_still_blocks(self):
        at_limit = [0] * (65536 // 8)
        self.assertTrue(self._non_blocking({FORCE_CC: "1", SWITCH: "1"}, at_limit))
        self.assertFalse(
            self._non_blocking({FORCE_CC: "1", SWITCH: "1"}, [*at_limit, 0])
        )


if __name__ == "__main__":
    unittest.main()
