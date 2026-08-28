"""Deterministic tests for watchdog fail-hard behavior after diagnostic faults."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import signal
from unittest.mock import MagicMock, patch

from sglang.srt.utils import watchdog as watchdog_module
from sglang.srt.utils.watchdog import WatchdogRaw
from sglang.test.test_utils import CustomTestCase


class FakeAcceleratorError(RuntimeError):
    """CUDA-like diagnostic failure without requiring an accelerator in CI."""


class TestWatchdogFailHard(CustomTestCase):
    @staticmethod
    def _raw(*, soft=False, dump_info=None):
        raw = WatchdogRaw.__new__(WatchdogRaw)
        raw.debug_name = "Scheduler"
        raw.get_counter = MagicMock(return_value=0)
        raw.is_active = MagicMock(return_value=True)
        raw.watchdog_timeout = 1.0
        raw.soft = soft
        raw.dump_info = dump_info
        raw.parent_process = MagicMock()
        return raw

    @staticmethod
    def _run_once(raw, *, pyspy_side_effect=None):
        with patch.object(
            watchdog_module.time, "perf_counter", side_effect=[0.0, 2.0]
        ), patch.object(watchdog_module.time, "sleep"), patch.object(
            watchdog_module,
            "pyspy_dump_schedulers",
            side_effect=pyspy_side_effect,
        ):
            raw._watchdog_once()

    def test_normal_diagnostics_still_fail_hard_once(self):
        raw = self._raw(dump_info=MagicMock(return_value="state"))
        self._run_once(raw)
        raw.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)

    def test_cuda_diagnostic_failure_still_fail_hard_once(self):
        raw = self._raw(
            dump_info=MagicMock(
                side_effect=FakeAcceleratorError(
                    "CUDA error: unspecified launch failure"
                )
            )
        )
        self._run_once(raw)
        raw.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)

    def test_pyspy_failure_still_fail_hard_once(self):
        raw = self._raw(dump_info=MagicMock(return_value="state"))
        self._run_once(raw, pyspy_side_effect=RuntimeError("pyspy failed"))
        raw.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)

    def test_soft_watchdog_never_signals_on_diagnostic_failure(self):
        raw = self._raw(
            soft=True,
            dump_info=MagicMock(side_effect=RuntimeError("diagnostic failed")),
        )
        self._run_once(raw)
        raw.parent_process.send_signal.assert_not_called()
