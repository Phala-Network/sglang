"""CPU-only regression of the actual HugeTLB budget functions."""

import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

SOURCE = Path(os.environ.get(
    "KIMI_BUDGET_SOURCE",
    str(Path(__file__).resolve().parents[4] / "python/sglang/srt/mem_cache/pool_host/base.py"),
))
GIB = 1024**3


class StaggeredBudgetTests(unittest.TestCase):
    def setUp(self):
        self.meminfo = Mock()
        self.mode = Mock(return_value="1GB")
        self.normal = Mock(return_value=SimpleNamespace(available=268 * GIB))
        self.ranks = Mock(return_value=8)
        self.ns = {
            "Path": lambda _: SimpleNamespace(read_text=self.meminfo),
            "envs": SimpleNamespace(
                SGLANG_HUGEPAGE_SIZE=SimpleNamespace(get=self.mode)
            ),
            "psutil": SimpleNamespace(virtual_memory=self.normal),
            "ranks_per_host": self.ranks,
            "HICACHE_HOST_MEMORY_RESERVE_BYTES": 10 * GIB,
        }
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"_available_1g_hugetlb_bytes", "host_memory_budget_bytes"}
        nodes = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        self.assertEqual({node.name for node in nodes}, names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), self.ns)

    def budget(self, free, reserved=0, size=1048576):
        self.meminfo.return_value = (
            f"HugePages_Free: {free}\nHugePages_Rsvd: {reserved}\n"
            f"Hugepagesize: {size} kB\n"
        )
        return self.ns["host_memory_budget_bytes"]()

    def test_all_eight_staggered_kv_and_mamba_pools_fit(self):
        for interleaved in (False, True):
            free = 2680
            pools = [254_490_000_000, 33_530_000_000]
            sequence = pools * 8 if interleaved else [pools[0]] * 8 + [pools[1]] * 8
            for request in sequence:
                self.assertGreaterEqual(self.budget(free), request)
                free -= math.ceil(request / GIB)
            self.assertGreaterEqual(free, 0)

    def test_observed_r3_failure_no_longer_double_splits(self):
        remaining = 1728
        request = 254_490_000_000
        self.assertLess(remaining * GIB // 8, request)
        self.assertGreaterEqual(self.budget(remaining), request)
        self.ranks.assert_not_called()
        self.normal.assert_not_called()

    def test_insufficient_or_reserved_pages_are_not_available(self):
        self.assertLess(self.budget(237), 254_490_000_000)
        self.assertEqual(self.budget(2680, 2679), GIB)
        self.assertEqual(self.budget(1, 2), 0)

    def test_bad_accounting_fails_closed(self):
        with self.assertRaises(ValueError):
            self.budget(2680, size=2048)
        self.meminfo.return_value = "HugePages_Free: 2680\n"
        with self.assertRaises(ValueError):
            self.ns["host_memory_budget_bytes"]()

    def test_plain_and_2m_keep_rank_split_and_reserve(self):
        for mode in ("", None, "2MB"):
            self.mode.return_value = mode
            self.assertEqual(self.budget(2680), 258 * GIB // 8)
        self.meminfo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
