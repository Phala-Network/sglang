"""CPU checks for the native telemetry hook, with no model or GPU."""
import builtins
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from prometheus_client import CollectorRegistry, Gauge

from pig_governor import Governor
from pig_governor.sglang import SglangGovernor
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)


class GovernorMetricsHookTests(unittest.TestCase):
    def setUp(self):
        self.core = Governor(0, max_running_requests=4)
        self.addCleanup(self.core.close)
        self.governor = SglangGovernor(self.core, max_running_requests=4)
        self.registry = CollectorRegistry()
        self.labels = {"model_name": "fixed-model"}
        self.reporter = object.__new__(SchedulerMetricsReporter)
        self.reporter.current_scheduler_metrics_enabled = True
        self.reporter.metrics_collector = MagicMock(labels=self.labels)
        self.reporter.scheduler = SimpleNamespace(
            governor=self.governor, waiting_queue=[], grammar_manager=[], chunked_req=None,
        )

    def initialize(self):
        with patch("prometheus_client.REGISTRY", self.registry), \
             patch("sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic", return_value=0):
            self.reporter._init_governor_metrics()

    def value(self, name):
        return self.registry.get_sample_value("pig_governor_" + name, self.labels)

    def test_current_native_count_includes_grammar_and_chunked_owner(self):
        self.initialize()
        self.reporter.scheduler.waiting_queue = [object(), object()]
        self.reporter.scheduler.grammar_manager = [object()]
        self.reporter.scheduler.chunked_req = object()
        self.reporter._publish_governor_metrics(0.1)
        self.assertEqual(self.value("waiting_requests"), 4)
        self.assertEqual(self.value("waiting_valid"), 1)
        self.reporter.scheduler.waiting_queue.clear()
        self.reporter.scheduler.grammar_manager.clear()
        self.reporter.scheduler.chunked_req = None
        self.reporter._publish_governor_metrics(0.2)
        self.assertEqual(self.value("waiting_requests"), 0)

    def test_active_and_idle_time_accounting_publish_without_batch_logs(self):
        self.initialize()
        self.reporter.enable_metrics = True
        self.reporter._scheduler_time_accounting = None
        self.reporter.scheduler_stage_metrics = MagicMock()
        self.reporter.scheduler_stage_metrics.drain.return_value = {}
        self.reporter.scheduler.waiting_queue = [object()]
        with patch("sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic_ns", side_effect=[0, 10**9, 2 * 10**9]):
            self.reporter.record_scheduler_active()
            self.reporter.record_scheduler_active()
            self.assertEqual(self.value("waiting_requests"), 1)
            self.reporter.scheduler.waiting_queue.clear()
            self.reporter.record_scheduler_idle()
        self.assertEqual(self.value("waiting_requests"), 0)
        self.reporter.metrics_collector.log_stats.assert_not_called()

    def test_disabled_governor_remains_scrapeable(self):
        self.reporter.scheduler.governor = None
        self.initialize()
        self.assertEqual(self.value("available"), 1)
        self.assertEqual(self.value("enabled"), 0)
        self.assertEqual(self.value("state_valid"), 1)

    def test_reserved_extra_labels_do_not_break_legacy_scraping(self):
        self.reporter.metrics_collector.labels = {
            **self.labels, "reason": "deployment-label", "window": "deployment-label",
        }
        self.initialize()
        self.assertEqual(self.value("available"), 1)
        self.assertEqual(self.value("enabled"), 1)

    def test_real_native_rejection_advances_exported_counter(self):
        from array import array
        from sglang.test.test_utils import maybe_stub_sgl_kernel

        maybe_stub_sgl_kernel()
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.sampling.sampling_params import SamplingParams

        self.initialize()
        self.governor.max_waiting = 0
        scheduler = object.__new__(Scheduler)
        scheduler.governor = self.governor
        scheduler.waiting_queue = []
        scheduler.grammar_manager = []
        scheduler.chunked_req = None
        sent = []
        scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=SimpleNamespace(
            send_output=lambda abort, req: sent.append(abort),
        ))
        params = SamplingParams(max_new_tokens=1)
        params.normalize(None)
        req = Req(rid="private-request", origin_input_text="", origin_input_ids=array("q", [1]),
                  sampling_params=params)
        req.time_stats.trace_ctx = SimpleNamespace(abort=lambda **kwargs: None)
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=0.1), \
             patch("sglang.srt.managers.scheduler.get_serving", return_value=SimpleNamespace(weight_version="test")):
            self.assertTrue(scheduler._abort_on_governor_admission(req))
        self.assertEqual(sent[0].finished_reason["status_code"], 429)
        self.reporter._publish_governor_metrics(0.2)
        self.assertEqual(self.value("admission_rejects_total"), 1)
        self.assertEqual(self.registry.get_sample_value(
            "pig_governor_admission_rejections_total", {**self.labels, "reason": "waiting_limit"}), 1)
        self.assertEqual(scheduler.waiting_queue, [])

    def test_missing_legacy_metrics_module_exports_explicit_unavailability(self):
        original_import = builtins.__import__

        def legacy_import(name, *args, **kwargs):
            if name == "pig_governor.metrics":
                raise ModuleNotFoundError("Legacy package", name=name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=legacy_import), \
             patch("prometheus_client.Gauge", side_effect=lambda *args, **kwargs: Gauge(
                 *args, **kwargs, registry=self.registry)):
            self.reporter._init_governor_metrics()
        self.assertEqual(self.value("available"), 0)
        self.assertEqual(self.value("enabled"), 1)
        self.reporter._publish_governor_metrics(0.1)


if __name__ == "__main__":
    unittest.main()
