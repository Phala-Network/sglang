"""Deterministic tests for watchdog fail-hard behavior after diagnostic faults."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import multiprocessing as mp
import os
import signal
import time
from unittest.mock import MagicMock, patch

from sglang.srt.utils import watchdog as watchdog_module
from sglang.srt.utils.watchdog import WatchdogRaw
from sglang.test.test_utils import CustomTestCase


class FakeAcceleratorError(RuntimeError):
    """CUDA-like diagnostic failure without requiring an accelerator in CI."""


def _blocking_diagnostic_worker():
    """Exercise the real timer + os._exit path in a disposable process."""
    raw = WatchdogRaw.__new__(WatchdogRaw)
    raw.debug_name = "Scheduler"
    raw.get_counter = lambda: 0
    raw.is_active = lambda: True
    raw.watchdog_timeout = 0.01
    raw.soft = False
    raw.dump_info = lambda: time.sleep(60)
    raw.parent_process = None
    raw.hard_exit_timeout = 0.1
    raw.hard_exit_fn = os._exit
    raw._watchdog_once()


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
        raw.hard_exit_timeout = 30.0
        raw.hard_exit_fn = MagicMock()
        return raw

    @staticmethod
    def _run_once(raw, *, pyspy_side_effect=None):
        with (
            patch.object(watchdog_module.time, "perf_counter", side_effect=[0.0, 2.0]),
            patch.object(watchdog_module.time, "sleep"),
            patch.object(
                watchdog_module,
                "pyspy_dump_schedulers",
                side_effect=pyspy_side_effect,
            ),
        ):
            raw._watchdog_once()

    def test_normal_diagnostics_still_fail_hard_once(self):
        raw = self._raw(dump_info=MagicMock(return_value="state"))
        self._run_once(raw)
        raw.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)
        raw.hard_exit_fn.assert_called_once_with(1)

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
        raw.hard_exit_fn.assert_called_once_with(1)

    def test_pyspy_failure_still_fail_hard_once(self):
        raw = self._raw(dump_info=MagicMock(return_value="state"))
        self._run_once(raw, pyspy_side_effect=RuntimeError("pyspy failed"))
        raw.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)
        raw.hard_exit_fn.assert_called_once_with(1)

    def test_parent_signal_failure_still_exits_scheduler(self):
        raw = self._raw(dump_info=MagicMock(return_value="state"))
        raw.parent_process.send_signal.side_effect = ProcessLookupError(
            "parent already exited"
        )
        with self.assertRaises(ProcessLookupError):
            self._run_once(raw)
        raw.hard_exit_fn.assert_called_once_with(1)

    def test_soft_watchdog_never_signals_on_diagnostic_failure(self):
        raw = self._raw(
            soft=True,
            dump_info=MagicMock(side_effect=RuntimeError("diagnostic failed")),
        )
        self._run_once(raw)
        raw.parent_process.send_signal.assert_not_called()
        raw.hard_exit_fn.assert_not_called()

    def test_hard_watchdog_arms_and_cancels_diagnostic_deadline(self):
        raw = self._raw(dump_info=MagicMock(return_value="state"))
        timer = MagicMock()
        with patch.object(
            watchdog_module.threading, "Timer", return_value=timer
        ) as timer_type:
            self._run_once(raw)

        timer_type.assert_called_once_with(
            30.0, raw._hard_exit_after_diagnostic_timeout
        )
        self.assertTrue(timer.daemon)
        timer.start.assert_called_once_with()
        timer.cancel.assert_called_once_with()

    def test_diagnostic_deadline_force_exits_scheduler(self):
        raw = self._raw()
        raw._hard_exit_after_diagnostic_timeout()
        raw.hard_exit_fn.assert_called_once_with(1)

    def test_blocked_diagnostic_process_is_force_exited(self):
        process = mp.Process(target=_blocking_diagnostic_worker)
        process.start()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            self.fail("hard-watchdog deadline did not terminate blocked diagnostics")
        self.assertEqual(process.exitcode, 1)
