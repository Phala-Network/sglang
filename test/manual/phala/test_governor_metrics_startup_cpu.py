"""Run the real Governor metrics source methods with lightweight CPU stubs."""

import ast
import copy
import sys
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/managers/scheduler_components/metrics_reporter.py"


class FakeGovernorMetrics:
    instances = []

    def __init__(self, labels):
        self.labels = labels
        self.publications = []
        self.instances.append(self)

    def publish(self, governor, *, now, waiting_count):
        self.publications.append(
            SimpleNamespace(
                governor=governor,
                now=now,
                waiting_count=waiting_count,
            )
        )


def load_reporter_class():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SchedulerMetricsReporter"
    )
    method_names = {"_init_governor_metrics", "_publish_governor_metrics"}
    methods = [
        copy.deepcopy(node)
        for node in source_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    ]
    if {method.name for method in methods} != method_names:
        raise AssertionError("Governor metrics source methods are missing")

    reporter_class = copy.deepcopy(source_class)
    reporter_class.bases = []
    reporter_class.keywords = []
    reporter_class.decorator_list = []
    reporter_class.body = methods
    module = ast.fix_missing_locations(
        ast.Module(body=[reporter_class], type_ignores=[])
    )
    namespace = {"time": time}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["SchedulerMetricsReporter"]


SOURCE_REPORTER = load_reporter_class()


class GovernorMetricsStartupSourceTests(unittest.TestCase):
    def setUp(self):
        FakeGovernorMetrics.instances.clear()
        package = types.ModuleType("pig_governor")
        package.__path__ = []
        metrics = types.ModuleType("pig_governor.metrics")
        metrics.GovernorMetrics = FakeGovernorMetrics
        package.metrics = metrics
        self.modules = {"pig_governor": package, "pig_governor.metrics": metrics}

    def reporter(self, scheduler, *, enabled=True):
        reporter = object.__new__(SOURCE_REPORTER)
        reporter.current_scheduler_metrics_enabled = enabled
        reporter.metrics_collector = SimpleNamespace(
            labels={"model_name": "fixed-model", "reason": "reserved"}
        )
        reporter.scheduler = scheduler
        return reporter

    def initialize(self, reporter):
        with patch.dict(sys.modules, self.modules):
            reporter._init_governor_metrics()

    def test_startup_without_grammar_then_counts_and_drains_live_queues(self):
        governor = object()
        scheduler = SimpleNamespace(
            governor=governor,
            waiting_queue=[object()],
            chunked_req=None,
        )
        reporter = self.reporter(scheduler)

        self.initialize(reporter)
        metrics = FakeGovernorMetrics.instances[-1]
        self.assertEqual(metrics.labels, {"model_name": "fixed-model"})
        self.assertEqual(metrics.publications[-1].waiting_count, 1)

        scheduler.waiting_queue = [object(), object()]
        scheduler.grammar_manager = [object(), object(), object()]
        scheduler.chunked_req = object()
        reporter._publish_governor_metrics(1.0)
        self.assertEqual(metrics.publications[-1].waiting_count, 6)

        scheduler.waiting_queue.clear()
        scheduler.grammar_manager.clear()
        scheduler.chunked_req = None
        reporter._publish_governor_metrics(2.0)
        self.assertEqual(metrics.publications[-1].waiting_count, 0)

    def test_scheduler_metrics_disabled_skips_governor_metrics(self):
        reporter = self.reporter(SimpleNamespace(), enabled=False)
        self.initialize(reporter)
        self.assertIsNone(reporter._governor_metrics)
        self.assertEqual(FakeGovernorMetrics.instances, [])

    def test_disabled_governor_publishes_without_queue_count(self):
        reporter = self.reporter(SimpleNamespace(governor=None))
        self.initialize(reporter)
        publication = FakeGovernorMetrics.instances[-1].publications[-1]
        self.assertIsNone(publication.governor)
        self.assertIsNone(publication.waiting_count)


if __name__ == "__main__":
    unittest.main()
