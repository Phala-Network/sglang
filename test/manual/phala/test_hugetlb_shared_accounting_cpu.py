"""Real source functions with small proc/sys fixtures; no mmap or GPU allocation."""

import ast
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

SOURCE = Path(
    os.environ.get(
        "SGLANG_TEST_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[3] / "python/sglang/srt"),
    )
)
GIB = 1024**3
MIB = 1024**2


def functions(relative, names, namespace):
    path = SOURCE / relative
    nodes = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class HugeTLBAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.page = GIB
        self.normal = Mock(return_value=SimpleNamespace(available=268 * GIB))
        self.ranks = Mock(return_value=8)
        self.limits = {}
        self.headroom = Mock(side_effect=lambda resource: self.limits.get(resource))
        self.budget = functions(
            "mem_cache/pool_host/allocation_budget.py",
            {"_parse_nodes", "available_hugepage_bytes", "_resource_snapshot"},
            {
                "Path": lambda value: self.root / str(value).lstrip("/"),
                "psutil": SimpleNamespace(virtual_memory=self.normal),
                "_cgroup_headroom": self.headroom,
                "host_memory_reserve_bytes": lambda: 128 * GIB,
            },
        )
        self.available = Mock(
            side_effect=lambda size: self.budget["available_hugepage_bytes"](size)
        )
        self.base = functions(
            "mem_cache/pool_host/base.py",
            {"host_memory_budget_bytes"},
            {
                "psutil": SimpleNamespace(virtual_memory=self.normal),
                "ranks_per_host": self.ranks,
                "HICACHE_HOST_MEMORY_RESERVE_BYTES": 10 * GIB,
                "requested_hugepage_bytes": lambda: self.page,
                "available_hugepage_bytes": self.available,
            },
        )
        self.put("/proc/self/status", "Mems_allowed_list:\t0-1\n")
        self.put("/sys/devices/system/node/online", "0-1\n")
        self.pages(2680)

    def put(self, name, value):
        path = self.root / name.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(value), encoding="utf-8")

    def pages(self, free, reserved=0, size=GIB):
        directory = f"/sys/kernel/mm/hugepages/hugepages-{size // 1024}kB"
        self.put(directory + "/free_hugepages", free)
        self.put(directory + "/resv_hugepages", reserved)

    def budget_bytes(self):
        return self.base["host_memory_budget_bytes"]()

    def test_observed_late_rank_is_not_rejected_by_a_second_split(self):
        self.pages(1728)
        request = 254_490_000_000
        self.assertLess(1728 * GIB // 8, request)
        self.assertGreaterEqual(self.budget_bytes(), request)
        self.normal.assert_not_called()
        self.ranks.assert_not_called()

    def test_eight_staggered_kv_and_mamba_pools_fit(self):
        pools = [254_490_000_000, 33_530_000_000]
        for sequence in (pools * 8, [pools[0]] * 8 + [pools[1]] * 8):
            free = 2680
            for request in sequence:
                self.pages(free)
                self.assertGreaterEqual(self.budget_bytes(), request)
                free -= math.ceil(request / GIB)
            self.assertGreaterEqual(free, 0)

    def test_reserved_and_exhausted_pages_are_not_available(self):
        self.pages(237)
        self.assertLess(self.budget_bytes(), 254_490_000_000)
        self.pages(2680, 2679)
        self.assertEqual(self.budget_bytes(), GIB)
        self.pages(1, 2)
        self.assertEqual(self.budget_bytes(), 0)

    def test_explicit_pool_does_not_depend_on_meminfo_default_page_size(self):
        self.put("/proc/meminfo", "Hugepagesize: 2048 kB\nHugePages_Free: 0\n")
        self.pages(7, 2)
        self.assertEqual(self.budget_bytes(), 5 * GIB)

    def test_numa_restriction_deducts_global_reservations_conservatively(self):
        self.put("/sys/devices/system/node/online", "0-3\n")
        self.pages(100, 5)
        for node, free in ((0, 8), (1, 7)):
            self.put(
                f"/sys/devices/system/node/node{node}/hugepages/"
                "hugepages-1048576kB/free_hugepages",
                free,
            )
        self.assertEqual(self.budget_bytes(), 10 * GIB)
        self.pages(100, 16)
        self.assertEqual(self.budget_bytes(), 0)

    def test_invalid_or_missing_pool_accounting_fails_closed(self):
        directory = "/sys/kernel/mm/hugepages/hugepages-1048576kB/"
        for name in ("free_hugepages", "resv_hugepages"):
            for value in ("-1", "not-a-number", "+1", "1.0"):
                with self.subTest(name=name, value=value):
                    self.pages(10, 0)
                    self.put(directory + name, value)
                    with self.assertRaises(ValueError):
                        self.budget_bytes()
        (self.root / (directory + "resv_hugepages").lstrip("/")).unlink()
        with self.assertRaises(OSError):
            self.budget_bytes()
        self.normal.assert_not_called()

    def test_invalid_numa_counter_cannot_inflate_available_pages(self):
        self.put("/proc/self/status", "Mems_allowed_list:\t0\n")
        self.put(
            "/sys/devices/system/node/node0/hugepages/hugepages-1048576kB/free_hugepages",
            -2,
        )
        with self.assertRaises(ValueError):
            self.budget_bytes()

    def test_plain_and_two_mib_preserve_legacy_rank_split_and_reserve(self):
        for page in (0, 2 * MIB):
            self.page = page
            self.assertEqual(self.budget_bytes(), 258 * GIB // 8)
        self.available.assert_not_called()

    def test_dsa_snapshot_keeps_ordinary_reserve_and_cgroup_caps(self):
        self.normal.return_value = SimpleNamespace(available=320 * GIB)
        self.pages(1728, 28)
        self.limits.update(
            {
                "memory": 248 * GIB,
                "hugetlb.1GB": 500 * GIB,
                "hugetlb.1GB.rsvd": 400 * GIB,
            }
        )
        self.assertEqual(self.budget["_resource_snapshot"](GIB), (120 * GIB, 400 * GIB))
        self.assertEqual(self.ranks.call_count, 0)

    def test_dsa_two_mib_and_plain_controls(self):
        self.normal.return_value = SimpleNamespace(available=320 * GIB)
        self.pages(20, 4, size=2 * MIB)
        self.limits.update({"memory": 248 * GIB, "hugetlb.2MB.rsvd": 18 * MIB})
        self.assertEqual(
            self.budget["_resource_snapshot"](2 * MIB), (120 * GIB, 18 * MIB)
        )
        self.assertEqual(self.budget["_resource_snapshot"](0), (120 * GIB, 0))

    def test_unknown_hugepage_size_is_rejected(self):
        for size in (0, -1, 1024, 3 * GIB):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.budget["available_hugepage_bytes"](size)


if __name__ == "__main__":
    unittest.main()
