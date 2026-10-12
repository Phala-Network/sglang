"""Deterministic tests for termination-first hard watchdog behavior."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import signal
from unittest.mock import MagicMock, patch

import psutil

from sglang.srt.utils import watchdog as watchdog_module
from sglang.srt.utils.watchdog import WatchdogRaw
from sglang.test.test_utils import CustomTestCase


class TestWatchdogFailHard(CustomTestCase):
    @staticmethod
    def _raw(*, soft=False, dump_info=None):
        # Bypass __init__: production __init__ starts the daemon thread before a
        # deterministic test can install its clock and process mocks.
        raw = WatchdogRaw.__new__(WatchdogRaw)
        raw.debug_name = "Scheduler"
        raw.get_counter = MagicMock(return_value=0)
        raw.is_active = MagicMock(return_value=True)
        raw.watchdog_timeout = 1.0
        raw.soft = soft
        raw.dump_info = dump_info
        raw.parent_process = MagicMock()
        raw.parent_process.wait.return_value = None
        return raw

    @staticmethod
    def _run_once(raw, *, pyspy_side_effect=None):
        # First timestamp seeds watchdog_last_time; the second crosses timeout.
        with patch.object(
            watchdog_module.time, "perf_counter", side_effect=[0.0, 2.0]
        ), patch.object(watchdog_module.time, "sleep"), patch.object(
            watchdog_module,
            "pyspy_dump_schedulers",
            side_effect=pyspy_side_effect,
        ):
            raw._watchdog_once()

    def test_hard_timeout_signals_before_and_skips_diagnostics(self):
        dump_info = MagicMock(side_effect=AssertionError("must not run"))
        raw = self._raw(dump_info=dump_info)

        with patch.object(watchdog_module, "pyspy_dump_schedulers") as pyspy:
            raw._handle_timeout()

        raw.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)
        raw.parent_process.wait.assert_called_once_with(
            timeout=watchdog_module.HARD_WATCHDOG_GRACE_SECONDS
        )
        raw.parent_process.kill.assert_not_called()
        dump_info.assert_not_called()
        pyspy.assert_not_called()

    def test_parent_still_alive_after_grace_receives_sigkill(self):
        raw = self._raw(dump_info=MagicMock(return_value="state"))
        raw.parent_process.wait.side_effect = psutil.TimeoutExpired(
            watchdog_module.HARD_WATCHDOG_GRACE_SECONDS,
            pid=123,
        )

        self._run_once(raw)

        raw.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)
        raw.parent_process.kill.assert_called_once_with()

    def test_parent_gone_after_sigquit_does_not_receive_sigkill(self):
        raw = self._raw()
        raw.parent_process.wait.side_effect = psutil.NoSuchProcess(pid=123)

        self._run_once(raw)

        raw.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)
        raw.parent_process.kill.assert_not_called()

    def test_soft_watchdog_never_signals_on_diagnostic_failure(self):
        dump_info = MagicMock(side_effect=RuntimeError("diagnostic failed"))
        raw = self._raw(soft=True, dump_info=dump_info)
        self._run_once(raw, pyspy_side_effect=RuntimeError("pyspy failed"))

        raw.parent_process.send_signal.assert_not_called()
        raw.parent_process.wait.assert_not_called()
        raw.parent_process.kill.assert_not_called()
        dump_info.assert_called_once_with()
