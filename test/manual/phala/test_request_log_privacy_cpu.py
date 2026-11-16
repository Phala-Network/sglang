"""Execute the actual request/queue log methods with stdlib dependency fixtures.

No model or CUDA import is required. Real framework and HTTP fixtures are
separate; these tests deliberately preserve logger targets and JSON events.
"""

import ast
import io
import importlib.util
import json
import logging
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"
MARKER = "SYNTHETIC_PRIVATE_PROMPT_RESPONSE_AUTH_RID_CIPHERTEXT"


def extract(relative, names, namespace):
    path = SRT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)


class RequestLogPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.streams = [io.StringIO(), io.StringIO()]
        self.targets = []
        for index, stream in enumerate(self.streams):
            logger = logging.getLogger(f"privacy-fixture.{id(self)}.{index}")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            handler = logging.StreamHandler(stream)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            self.addCleanup(logger.removeHandler, handler)
            self.addCleanup(handler.close)
            self.targets.append(logger)

        self.ns = {
            "logging": logging,
            "json": json,
            "datetime": datetime,
            "__name__": "privacy-fixture",
            "envs": NS(SGLANG_LOG_REQUEST_EXCEEDED_MS=NS(get=lambda: 0)),
            "create_log_targets": lambda **_: self.targets,
            "time": NS(time=lambda: 100),
            "dist": NS(is_initialized=lambda: False),
        }
        extract("utils/log_utils.py", {"log_json"}, self.ns)
        extract("utils/request_logger.py", {"RequestLogger"}, self.ns)
        extract("utils/scheduler_status_logger.py", {"SchedulerStatusLogger"}, self.ns)
        self.obj = NS(
            rid=MARKER,
            text=MARKER,
            input_ids=[1, 2],
            image_data=f"https://media.example/image?auth={MARKER}",
        )
        self.request = NS(
            headers={"authorization": MARKER, "x-smg-routing-key": MARKER}
        )
        self.out = {
            "text": MARKER,
            "output_ids": [3, 4],
            "meta_info": {"e2e_latency": 0.1},
        }

    def request_logger(self, format="json", level=3, enabled=True):
        return self.ns["RequestLogger"](enabled, level, format, ["stdout", "/fixture"])

    def output(self):
        for stream in self.streams:
            self.assertNotIn(MARKER, stream.getvalue())
        self.assertEqual(self.streams[0].getvalue(), self.streams[1].getvalue())
        return self.streams[0].getvalue()

    def test_json_events_targets_and_log_levels_survive_without_payloads(self):
        logger = self.request_logger()
        logger.log_received_request(self.obj, request=self.request)
        logger.log_openai_received_request({"messages": MARKER}, self.request)
        logger.log_finished_request(self.obj, self.out, self.request)
        rows = [json.loads(line) for line in self.output().splitlines()]
        self.assertEqual(
            [row["event"] for row in rows],
            ["request.received", "request.received.openai", "request.finished"],
        )
        self.assertTrue(all(row["redacted"] for row in rows))
        self.assertTrue(all("timestamp" in row for row in rows))
        self.assertTrue(all(target.level == logging.INFO for target in self.targets))

    def test_text_events_preserve_both_targets(self):
        logger = self.request_logger(format="text")
        logger.log_received_request(self.obj, request=self.request)
        logger.log_openai_received_request({"messages": MARKER}, self.request)
        logger.log_finished_request(self.obj, self.out, self.request)
        output = self.output()
        self.assertIn("Receive:", output)
        self.assertIn("Receive OpenAI:", output)
        self.assertIn("Finish:", output)

    def test_logging_does_not_evaluate_customer_repr_or_openai_dump(self):
        class PrivatePayload:
            def __repr__(self):
                raise AssertionError("customer repr was evaluated")

            def model_dump(self, **_):
                raise AssertionError("customer model_dump was evaluated")

        logger = self.request_logger()
        logger.log_openai_received_request(PrivatePayload(), self.request)
        self.obj.text = PrivatePayload()
        logger.log_received_request(self.obj, request=self.request)
        self.output()

    def test_tokenizer_decode_assignment_remains_business_behavior(self):
        logger = self.request_logger(level=2)
        self.obj.text = None
        calls = []

        def decode(tokens, **_):
            calls.append(tokens)
            return MARKER

        logger.log_received_request(self.obj, tokenizer=NS(decode=decode))
        self.assertEqual(calls, [[1, 2]])
        self.assertEqual(self.obj.text, MARKER)
        self.output()

    def test_disabled_and_latency_threshold_contracts_remain(self):
        logger = self.request_logger(enabled=False)
        logger.log_received_request(self.obj)
        logger.log_finished_request(self.obj, self.out)
        self.assertEqual(self.output(), "")
        logger = self.request_logger()
        logger.log_exceeded_ms = 200
        logger.log_finished_request(self.obj, self.out)
        self.assertEqual(self.output(), "")

    def test_queue_counts_remain_visible_and_rids_are_private(self):
        logger = self.ns["SchedulerStatusLogger"](["stdout"], dump_interval=5)
        batch = NS(reqs=[self.obj, self.obj])
        logger.maybe_dump(batch, [self.obj])
        logger.maybe_dump(batch, [self.obj])
        rows = [json.loads(line) for line in self.output().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event"], "scheduler.status")
        self.assertEqual(rows[0]["running_count"], 2)
        self.assertEqual(rows[0]["queued_count"], 1)
        self.assertEqual(rows[0]["rank"], 0)

    def test_actual_media_error_log_sites_keep_type_without_exception_text(self):
        for target in self.targets:
            target.setLevel(logging.DEBUG)
        spec = importlib.util.spec_from_file_location(
            "media_privacy_fixture", SRT / "utils/framework_log_privacy.py"
        )
        privacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(privacy)
        count = 0
        for relative in (
            "multimodal/processors/base_processor.py",
            "multimodal/processors/qwen_vl.py",
        ):
            path = SRT / relative
            tree = ast.parse(path.read_text(encoding="utf-8"))
            calls = []
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "logger"
                ):
                    continue
                names = {
                    item.id for item in ast.walk(node) if isinstance(item, ast.Name)
                }
                if (
                    names & {"e", "exc"}
                    or node.func.attr == "exception"
                    or any(kw.arg == "exc_info" for kw in node.keywords)
                ):
                    calls.append(node)
            for node in calls:
                count += 1
                module = ast.fix_missing_locations(
                    ast.Module(body=[ast.Expr(value=node)], type_ignores=[])
                )
                for target in self.targets:
                    try:
                        raise RuntimeError(MARKER)
                    except RuntimeError as error:
                        exec(
                            compile(module, str(path), "exec"),
                            {
                                "logger": target,
                                "e": error,
                                "exc": error,
                                "modality": NS(name="IMAGE"),
                                "idx": 2,
                                "safe_exception_type": privacy.safe_exception_type,
                            },
                        )
        self.assertGreater(count, 5)
        self.assertIn("RuntimeError", self.output())

    def test_qwen_stage_timing_log_survives_without_customer_rid(self):
        path = SRT / "multimodal/processors/qwen_vl.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_process_mm_data_uncached"
        )
        call = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
            and node.func.attr == "debug"
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[ast.Expr(value=call)], type_ignores=[])
        )
        for target in self.targets:
            target.setLevel(logging.DEBUG)
            exec(
                compile(module, str(path), "exec"),
                {
                    "logger": target,
                    "rid": MARKER,
                    "entry_time": 0,
                    "load_time": 1,
                    "preprocess_time": 2,
                    "process_time": 3,
                    "get_rope_index_time": 4,
                },
            )
        output = self.output()
        self.assertIn("load_time", output)
        self.assertIn("total_time", output)


if __name__ == "__main__":
    unittest.main()
