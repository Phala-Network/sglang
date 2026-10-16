"""CPU regressions for per-launch failure propagation and watchdog ordering."""

import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sglang.srt.utils import watchdog as watchdog_module
from sglang.srt.utils.engine_failure import (
    create_engine_failure_state,
    engine_has_failed,
    mark_engine_failed,
)
from sglang.srt.utils.watchdog import SubprocessWatchdog, WatchdogRaw
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEngineFailure(unittest.TestCase):
    def test_latch_is_terminal_and_generations_are_isolated(self):
        first = create_engine_failure_state()
        second = create_engine_failure_state()
        self.addCleanup(Path(first).unlink, missing_ok=True)
        self.addCleanup(Path(second).unlink, missing_ok=True)
        self.assertNotEqual(first, second)
        self.assertFalse(engine_has_failed(first))
        mark_engine_failed(first)
        mark_engine_failed(first)
        self.assertTrue(engine_has_failed(first))
        self.assertFalse(engine_has_failed(second))

    def test_missing_corrupt_and_unreadable_state_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "state")
            self.assertTrue(engine_has_failed(path))
            for payload in (b"", b"00", b"garbage", b"1"):
                Path(path).write_bytes(payload)
                self.assertTrue(engine_has_failed(path))
            with patch("builtins.open", side_effect=PermissionError):
                self.assertTrue(engine_has_failed(path))

    def test_mark_never_recreates_a_missing_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "removed")
            with self.assertRaises(FileNotFoundError):
                mark_engine_failed(path)
            self.assertFalse(Path(path).exists())


class TestFailureOrdering(unittest.TestCase):
    @staticmethod
    def raw(*, soft=False):
        raw = WatchdogRaw.__new__(WatchdogRaw)
        raw.debug_name = "Scheduler"
        raw.get_counter = lambda: 0
        raw.is_active = lambda: True
        raw.watchdog_timeout = 1.0
        raw.soft = soft
        raw.dump_info = MagicMock()
        raw.parent_process = MagicMock()
        raw.on_failure = MagicMock()
        raw._terminate_engine = MagicMock()
        return raw

    def test_hard_marks_before_signal_and_never_runs_diagnostics(self):
        raw = self.raw()
        order = []
        raw.on_failure.side_effect = lambda: order.append("mark")
        raw.parent_process.send_signal.side_effect = lambda _: order.append("signal")
        with patch.object(watchdog_module, "pyspy_dump_schedulers") as dump:
            raw._handle_timeout()
        self.assertEqual(order, ["mark", "signal"])
        raw.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)
        raw.parent_process.wait.assert_called_once_with(
            timeout=watchdog_module.HARD_WATCHDOG_GRACE_SECONDS
        )
        raw.dump_info.assert_not_called()
        dump.assert_not_called()
        raw._terminate_engine.assert_called_once_with(
            raw.parent_process.children.return_value
        )

    def test_failed_publication_does_not_suppress_fatal_signal(self):
        raw = self.raw()
        raw.on_failure.side_effect = OSError("unwritable state")
        raw._handle_timeout()
        raw.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)

    def test_soft_timeout_never_marks_or_signals(self):
        raw = self.raw(soft=True)
        with patch.object(watchdog_module, "pyspy_dump_schedulers"):
            raw._handle_timeout()
        raw.on_failure.assert_not_called()
        raw.parent_process.send_signal.assert_not_called()

    def test_idle_then_busy_receives_a_fresh_timeout(self):
        raw = self.raw()
        raw.is_active = MagicMock(side_effect=[False, False, True, True, True])
        with (
            patch.object(
                watchdog_module.time,
                "perf_counter",
                side_effect=[0, 0, 100, 101, 102, 103],
            ),
            patch.object(watchdog_module.time, "sleep"),
            patch.object(raw, "_handle_timeout") as timeout,
        ):
            raw._watchdog_once()
        self.assertEqual(raw.is_active.call_count, 5)
        timeout.assert_called_once_with()

    def test_child_crash_is_published_before_signal(self):
        process = MagicMock(pid=1, exitcode=1)
        process.is_alive.return_value = False
        order = []
        watchdog = SubprocessWatchdog(
            [process], on_failure=lambda: order.append("mark")
        )
        with patch.object(os, "kill", side_effect=lambda *_: order.append("signal")):
            self.assertTrue(watchdog._check_processes())
        self.assertEqual(order, ["mark", "signal"])


if __name__ == "__main__":
    unittest.main()
