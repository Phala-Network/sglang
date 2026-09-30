"""Execute the actual prefill completion diagnostic hook without GPU imports."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]


class StopAfterDiagnostic(Exception):
    pass


class Tests(unittest.TestCase):
    def test_ordinary_capture_duration_and_request_bounds(self):
        source = ROOT / "python/sglang/srt/mem_cache/shared_cache_diagnostics.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module.startswith("sglang"):
                continue
            if isinstance(node, ast.ClassDef):
                if node.name == "SharedCacheDiagnostics":
                    nodes.append(node)
                continue
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name)
                and not t.id.startswith("_")
                and t.id != "logger"
                for t in node.targets
            ):
                continue
            nodes.append(node)
        scope = {}
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope
        )
        collector = scope["SharedCacheDiagnostics"]
        self.assertFalse(collector().enabled)
        capture = collector(
            enabled=True,
            key_salt="test-salt",
            case_id="case",
            epoch="epoch",
            key_ids="0" * 64,
            max_duration_ms=1000000,
        )
        self.assertEqual(capture._max_duration_s, 900)
        self.assertEqual(capture._max_requests, 32)
        self.assertEqual(scope["_SEED_MAX_DURATION_MS"], 120000)
        for i in range(15):
            self.assertIsNotNone(capture._register_request(str(i), "default"))
        self.assertEqual(len(capture._requests), 15)
        self.assertTrue(capture._capture_open())

    def test_actual_extend_boundary_after_host_load_and_chunking(self):
        source = ROOT / "python/sglang/srt/disaggregation/prefill.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "process_batch_result_disagg_prefill"
        )
        method.decorator_list = []
        for start, end, n, prefix, host, chunks in (
            (128, 385, 385, 128, 128, 0),
            (128, 385, 1000, 128, 128, 1),
            (256, 385, 385, 256, 128, 0),
            (0, 129, 129, 0, 0, 0),
        ):
            with self.subTest(start=start, n=n):
                records, order = [], []

                def record(**kwargs):
                    records.append(kwargs)
                    order.append("diagnostic")

                def snapshot(*args):
                    order.append("snapshot")
                    raise StopAfterDiagnostic()

                req = SimpleNamespace(
                    rid="cpu-only",
                    extend_range=SimpleNamespace(start=start, end=end),
                    prefix_indices=list(range(prefix)),
                    host_hit_length=host,
                    origin_input_ids=list(range(n)),
                    inflight_middle_chunks=chunks,
                )
                before = (
                    list(req.prefix_indices),
                    req.host_hit_length,
                    list(req.origin_input_ids),
                )
                result = SimpleNamespace(
                    logits_output=None,
                    next_token_ids=None,
                    extend_input_len_per_req=None,
                    extend_logprob_start_len_per_req=None,
                    copy_done=SimpleNamespace(synchronize=lambda: order.append("copy")),
                )
                scope = {
                    "Scheduler": object,
                    "ScheduleBatch": object,
                    "GenerationBatchResult": object,
                    "shared_cache_diagnostics": SimpleNamespace(
                        enabled=True, record_prefill_forward=record
                    ),
                    "_record_shared_cache_diagnostic_failure": lambda: self.fail(
                        "diagnostic failed"
                    ),
                }
                exec(
                    compile(
                        ast.Module(body=[method], type_ignores=[]), str(source), "exec"
                    ),
                    scope,
                )
                scheduler = SimpleNamespace(
                    batch_result_processor=SimpleNamespace(
                        snapshot_auxiliary_output_starts=snapshot
                    )
                )
                with self.assertRaises(StopAfterDiagnostic):
                    scope[method.name](scheduler, SimpleNamespace(reqs=[req]), result)
                self.assertEqual(order, ["copy", "diagnostic", "snapshot"])
                self.assertEqual(
                    records,
                    [
                        {
                            "request_id": "cpu-only",
                            "h_tokens": start,
                            "n_tokens": n,
                            "forward_start": start,
                            "forward_end": end,
                            "tail_complete": chunks <= 0,
                        }
                    ],
                )
                self.assertEqual(
                    before,
                    (req.prefix_indices, req.host_hit_length, req.origin_input_ids),
                )


if __name__ == "__main__":
    unittest.main()
