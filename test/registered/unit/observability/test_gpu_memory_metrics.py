"""CPU tests for sampled device-memory observability."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.observability.metrics_collector import _GpuMemorySampler
from sglang.srt.platforms.cpu import CpuSRTPlatform


class _BoundRecordingGauge:
    def __init__(self, gauge, labels):
        self.gauge = gauge
        self.labels = labels

    def set(self, value):
        self.gauge.values.append((self.labels, value))


class _RecordingGauge:
    def __init__(self, *, name, documentation, labelnames, **kwargs):
        self.name = name
        self.documentation = documentation
        self.labelnames = tuple(labelnames)
        self.values = []

    def labels(self, **labels):
        return _BoundRecordingGauge(self, labels)


def _make_sampler(interval_seconds=1.0):
    return _GpuMemorySampler(
        labels={"model_name": "cpu-test", "tp_rank": 0},
        gauge_cls=_RecordingGauge,
        interval_seconds=interval_seconds,
    )


class TestGpuMemorySampler(unittest.TestCase):
    def test_cpu_platform_does_not_report_host_memory_as_gpu(self):
        sampler = _make_sampler()
        cpu_platform = CpuSRTPlatform()
        with (
            patch(
                "sglang.srt.observability.metrics_collector.current_platform",
                cpu_platform,
            ),
            patch(
                "sglang.srt.observability.metrics_collector.time.monotonic",
                return_value=10.0,
            ),
            patch(
                "sglang.srt.observability.metrics_collector.time.time",
                return_value=1_700_000_010.25,
            ),
            patch.object(cpu_platform, "empty_cache", side_effect=AssertionError),
        ):
            self.assertFalse(sampler.sample_if_due(device_id=0))

        self.assertTrue(math.isnan(sampler.free_memory_gb.values[-1][1]))
        self.assertEqual(
            sampler.sample_timestamp_seconds.values[-1][1], 1_700_000_010.25
        )
        self.assertEqual(sampler.sample_valid.values[-1][1], 0.0)
        self.assertEqual(sampler.sample_interval_seconds.values[-1][1], 1.0)

    def test_sampling_interval_throttles_platform_queries(self):
        sampler = _make_sampler(interval_seconds=2.0)
        get_available_memory = Mock(
            side_effect=[
                (8 * (1 << 30), 16 * (1 << 30)),
                (6 * (1 << 30), 16 * (1 << 30)),
            ]
        )
        platform = SimpleNamespace(
            device_name="test-gpu",
            is_cuda_alike=lambda: True,
            get_available_memory=get_available_memory,
        )
        with (
            patch(
                "sglang.srt.observability.metrics_collector.current_platform",
                platform,
            ),
            patch(
                "sglang.srt.observability.metrics_collector.time.monotonic",
                side_effect=[10.0, 11.0, 12.0],
            ),
            patch(
                "sglang.srt.observability.metrics_collector.time.time",
                side_effect=[100.0, 101.0],
            ),
        ):
            self.assertTrue(sampler.sample_if_due(device_id=3))
            self.assertFalse(sampler.sample_if_due(device_id=3))
            self.assertTrue(sampler.sample_if_due(device_id=3))

        self.assertEqual(sampler.free_memory_gb.values[-1][1], 6.0)
        self.assertEqual(get_available_memory.call_count, 2)
        get_available_memory.assert_called_with(3)

    def test_query_failure_clears_current_value_and_marks_sample_invalid(self):
        sampler = _make_sampler()
        successful_platform = SimpleNamespace(
            device_name="test-gpu",
            is_cuda_alike=lambda: True,
            get_available_memory=lambda _: (4, 8),
        )
        with (
            patch(
                "sglang.srt.observability.metrics_collector.current_platform",
                successful_platform,
            ),
            patch(
                "sglang.srt.observability.metrics_collector.time.monotonic",
                return_value=10.0,
            ),
            patch(
                "sglang.srt.observability.metrics_collector.time.time",
                return_value=100.0,
            ),
        ):
            self.assertTrue(sampler.sample_if_due(device_id=0))

        failing_platform = SimpleNamespace(
            device_name="test-gpu",
            is_cuda_alike=lambda: True,
            get_available_memory=Mock(side_effect=RuntimeError("probe")),
        )
        with (
            patch(
                "sglang.srt.observability.metrics_collector.current_platform",
                failing_platform,
            ),
            patch(
                "sglang.srt.observability.metrics_collector.time.monotonic",
                return_value=11.0,
            ),
            patch(
                "sglang.srt.observability.metrics_collector.time.time",
                return_value=101.0,
            ),
        ):
            self.assertFalse(sampler.sample_if_due(device_id=0))

        self.assertTrue(math.isnan(sampler.free_memory_gb.values[-1][1]))
        self.assertEqual(sampler.sample_timestamp_seconds.values[-1][1], 101.0)
        self.assertEqual(sampler.sample_valid.values[-1][1], 0.0)

    def test_invalid_platform_values_are_not_published_as_current(self):
        sampler = _make_sampler(interval_seconds=0.0)
        platform = SimpleNamespace(
            device_name="test-gpu",
            is_cuda_alike=lambda: True,
            get_available_memory=lambda _: (9, 8),
        )
        with (
            patch(
                "sglang.srt.observability.metrics_collector.current_platform",
                platform,
            ),
            patch(
                "sglang.srt.observability.metrics_collector.time.monotonic",
                return_value=10.0,
            ),
            patch(
                "sglang.srt.observability.metrics_collector.time.time",
                return_value=100.0,
            ),
        ):
            self.assertFalse(sampler.sample_if_due(device_id=0))

        self.assertTrue(math.isnan(sampler.free_memory_gb.values[-1][1]))
        self.assertEqual(sampler.sample_valid.values[-1][1], 0.0)


if __name__ == "__main__":
    unittest.main()
