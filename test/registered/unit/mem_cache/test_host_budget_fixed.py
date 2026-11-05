"""A fixed HiCache host budget: the same answer however far other ranks have allocated."""
import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host import base

GIB = 1024**3


class FixedHostBudgetTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(envs.SGLANG_HICACHE_HOST_BUDGET_GIB.clear)
        for name, value in (("ranks_per_host", 8),):
            p = mock.patch.object(base, name, return_value=value)
            p.start()
            self.addCleanup(p.stop)

    def budget(self, free_gib, **kwargs):
        with mock.patch.object(
            base, "available_host_memory_bytes", return_value=free_gib * GIB
        ):
            return base.host_memory_budget_bytes(40 * GIB, **kwargs)

    def test_sampling_races_other_ranks(self):
        # What the fixed budget is for: the same size fits or not by allocation order.
        self.assertGreaterEqual(self.budget(361), 40 * GIB)
        self.assertLess(self.budget(321), 40 * GIB)

    def test_fixed_does_not_depend_on_free_memory(self):
        envs.SGLANG_HICACHE_HOST_BUDGET_GIB.set(45)
        for free in (361, 321, 40, 0):
            self.assertEqual(self.budget(free), 45 * GIB)

    def test_auto_size_still_samples(self):
        envs.SGLANG_HICACHE_HOST_BUDGET_GIB.set(45)
        self.assertEqual(self.budget(90, auto_size=True), 10 * GIB)

    def test_scope_still_wins(self):
        envs.SGLANG_HICACHE_HOST_BUDGET_GIB.set(45)
        with base.host_memory_budget_scope(50 * GIB):
            self.assertEqual(self.budget(0), 50 * GIB)

    def test_default_samples(self):
        self.assertEqual(self.budget(90), 10 * GIB)


if __name__ == "__main__":
    unittest.main()
