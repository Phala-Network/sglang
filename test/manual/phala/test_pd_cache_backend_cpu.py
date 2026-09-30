"""Run cache reporting methods from source without CUDA dependencies."""

import ast
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/managers/scheduler_components/output_streamer.py"


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


def load_streamer():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.decorator_list = []
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_get_storage_backend_type", "get_cached_tokens_details")
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    namespace = {"DisaggregationMode": DisaggregationMode}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["SchedulerOutputStreamer"]


class CacheBackendReportingTest(unittest.TestCase):
    def report(
        self, mode, *, backend=None, enabled=True, counts=(0, 0, 1280), total=1280
    ):
        streamer = load_streamer()()
        streamer.disaggregation_mode = mode
        streamer.enable_hicache_storage = lambda: enabled
        streamer.tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(storage_backend=backend)
        )
        req = SimpleNamespace(
            cached_tokens=total,
            cached_tokens_device=counts[0],
            cached_tokens_host=counts[1],
            cached_tokens_storage=counts[2],
        )
        return streamer.get_cached_tokens_details(req)

    def test_prefill_hits_do_not_take_decode_backend_name(self):
        # The captured PD response had storage=1280 and backend="none".
        # Even a configured decode backend would not prove prefill's identity.
        for backend in (None, type("DifferentDecodeStore", (), {})()):
            with self.subTest(backend=backend):
                self.assertEqual(
                    self.report(DisaggregationMode.DECODE, backend=backend),
                    {"device": 0, "host": 0, "storage": 1280},
                )

    def test_decode_without_local_storage_keeps_prefill_counts(self):
        self.assertEqual(
            self.report(
                DisaggregationMode.DECODE, enabled=False, counts=(64, 128, 1024)
            ),
            {"device": 64, "host": 128, "storage": 1024},
        )

    def test_local_and_prefill_backend_reporting_remains(self):
        for mode in (DisaggregationMode.NULL, DisaggregationMode.PREFILL):
            for backend, expected in (
                (None, "none"),
                (type("MooncakeStore", (), {})(), "MooncakeStore"),
            ):
                with self.subTest(mode=mode, backend=expected):
                    self.assertEqual(
                        self.report(mode, backend=backend),
                        {
                            "device": 0,
                            "host": 0,
                            "storage": 1280,
                            "storage_backend": expected,
                        },
                    )

    def test_cache_without_storage_retains_original_breakdown(self):
        self.assertEqual(
            self.report(DisaggregationMode.NULL, enabled=False, counts=(64, 128, 0)),
            {"device": 64, "host": 128},
        )

    def test_no_hits_and_legacy_total(self):
        for mode in DisaggregationMode:
            with self.subTest(mode=mode):
                self.assertIsNone(self.report(mode, counts=(0, 0, 0), total=0))
                self.assertEqual(
                    self.report(mode, counts=(0, 0, 0), total=1280),
                    {"device": 1280, "host": 0},
                )


if __name__ == "__main__":
    unittest.main()
