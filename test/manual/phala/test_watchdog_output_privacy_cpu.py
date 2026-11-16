"""Exercise actual watchdog/dump implementations with fake processes and I/O.

Source loading avoids importing model/CUDA code. Signals, subprocesses, pipe
writes, logger failures and stream failures are all fixtures; this suite must
run in the authorized remote CPU window.
"""

import errno
import importlib.util
import signal
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"
MARKER = "SYNTHETIC_PRIVATE_WATCHDOG_PROMPT_RID_ARGV_CIPHERTEXT"


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, SRT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WatchdogOutputPrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        privacy = load_source(
            "watchdog_privacy_helper", "utils/framework_log_privacy.py"
        )
        cls.privacy = privacy
        with patch.dict(
            sys.modules, {"sglang.srt.utils.framework_log_privacy": privacy}
        ):
            cls.dump = load_source(
                "watchdog_dump_source", "utils/cudacore_pyspy_dump_utils.py"
            )
            with patch.dict(
                sys.modules, {"sglang.srt.utils.cudacore_pyspy_dump_utils": cls.dump}
            ):
                cls.watchdog = load_source("watchdog_source", "utils/watchdog.py")

    def raw(self, soft=False):
        raw = self.watchdog.WatchdogRaw.__new__(self.watchdog.WatchdogRaw)
        raw.debug_name = MARKER
        raw.get_counter = MagicMock(return_value=7)
        raw.is_active = MagicMock(return_value=True)
        raw.watchdog_timeout = 1.0
        raw.soft = soft
        raw.dump_info = MagicMock(return_value=MARKER)
        raw.parent_process = MagicMock()
        raw.parent_process.wait.return_value = None
        return raw

    def test_recovery_error_types_remain_known_but_custom_names_are_private(self):
        for error in (
            PermissionError(13, MARKER),
            self.watchdog.psutil.AccessDenied(pid=123),
            self.watchdog.psutil.NoSuchProcess(pid=123),
            self.watchdog.psutil.TimeoutExpired(5, pid=123),
        ):
            self.assertEqual(
                self.privacy.safe_exception_type(error), type(error).__name__
            )
        private_type = type(MARKER, (RuntimeError,), {})
        self.assertEqual(
            self.privacy.safe_exception_type(private_type(MARKER)), "Exception"
        )

    def test_hard_signal_wait_kill_precede_every_broken_output(self):
        raw = self.raw()
        events = []
        raw.parent_process.send_signal.side_effect = lambda sig: events.append(
            "SIGQUIT"
        )

        def wait(**_):
            events.append("wait")
            raise self.watchdog.psutil.TimeoutExpired(5, pid=123)

        raw.parent_process.wait.side_effect = wait
        raw.parent_process.kill.side_effect = lambda: events.append("SIGKILL")

        def broken_log(*_, **__):
            events.append("log")
            raise OSError(errno.ENOSPC, MARKER)

        def broken_flush(name):
            events.append(name)
            raise BrokenPipeError(MARKER)

        stderr = NS(flush=lambda: broken_flush("stderr"))
        stdout = NS(flush=lambda: broken_flush("stdout"))
        with patch.object(
            self.watchdog.logger, "error", side_effect=broken_log
        ), patch.object(self.watchdog.sys, "stderr", stderr), patch.object(
            self.watchdog.sys, "stdout", stdout
        ), patch.object(
            self.watchdog, "pyspy_dump_schedulers"
        ) as pyspy:
            raw._handle_timeout()
        self.assertEqual(
            events, ["SIGQUIT", "wait", "SIGKILL", "log", "stderr", "stdout"]
        )
        raw.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)
        raw.parent_process.wait.assert_called_once_with(
            timeout=self.watchdog.HARD_WATCHDOG_GRACE_SECONDS
        )
        raw.dump_info.assert_not_called()
        pyspy.assert_not_called()

    def test_hard_signal_and_wait_exceptions_still_attempt_kill(self):
        raw = self.raw()
        error = type(MARKER, (RuntimeError,), {})
        raw.parent_process.send_signal.side_effect = error(MARKER)
        raw.parent_process.wait.side_effect = OSError(errno.EIO, MARKER)
        with self.assertLogs(self.watchdog.logger, level="ERROR") as logs, patch.object(
            self.watchdog, "_best_effort_flush"
        ):
            raw._handle_timeout()
        raw.parent_process.kill.assert_called_once_with()
        self.assertNotIn(MARKER, "\n".join(logs.output))
        self.assertIn("OSError", "\n".join(logs.output))
        self.assertIn("SIGKILL requested", "\n".join(logs.output))

    def test_hard_no_such_parent_ignores_output_failure(self):
        raw = self.raw()
        raw.parent_process.send_signal.side_effect = self.watchdog.psutil.NoSuchProcess(
            pid=123
        )
        with patch.object(
            self.watchdog.logger, "error", side_effect=BrokenPipeError(MARKER)
        ), patch.object(self.watchdog, "_best_effort_flush"):
            raw._handle_timeout()
        raw.parent_process.wait.assert_not_called()
        raw.parent_process.kill.assert_not_called()

    def test_soft_opaque_snapshot_is_not_formatted_or_tested(self):
        class PrivateSnapshot:
            def __repr__(self):
                raise AssertionError("private repr evaluated")

            __str__ = __repr__

            def __bool__(self):
                raise AssertionError("private truthiness evaluated")

        raw = self.raw(soft=True)
        raw.dump_info.return_value = PrivateSnapshot()
        with self.assertLogs(self.watchdog.logger, level="ERROR") as logs, patch.object(
            self.watchdog, "pyspy_dump_schedulers"
        ), patch.object(self.watchdog, "_best_effort_flush"):
            raw._handle_timeout()
        output = "\n".join(logs.output)
        self.assertNotIn(MARKER, output)
        self.assertIn("counter=7, active=True", output)
        self.assertIn("snapshot completed", output)
        raw.dump_info.assert_called_once_with()
        raw.parent_process.send_signal.assert_not_called()
        raw.parent_process.kill.assert_not_called()

    def test_soft_broken_log_and_flush_do_not_skip_later_diagnostics(self):
        raw = self.raw(soft=True)
        raw.dump_info.side_effect = RuntimeError(MARKER)
        stderr, stdout = MagicMock(), MagicMock()
        stderr.flush.side_effect = BrokenPipeError(MARKER)
        stdout.flush.side_effect = OSError(errno.ENOSPC, MARKER)
        with patch.object(
            self.watchdog.logger, "error", side_effect=BrokenPipeError(MARKER)
        ), patch.object(self.watchdog.sys, "stderr", stderr), patch.object(
            self.watchdog.sys, "stdout", stdout
        ), patch.object(
            self.watchdog, "pyspy_dump_schedulers", side_effect=RuntimeError(MARKER)
        ) as pyspy:
            raw._handle_timeout()
        raw.dump_info.assert_called_once_with()
        pyspy.assert_called_once_with()
        stderr.flush.assert_called_once_with()
        stdout.flush.assert_called_once_with()
        raw.parent_process.send_signal.assert_not_called()
        raw.parent_process.wait.assert_not_called()
        raw.parent_process.kill.assert_not_called()

    def test_soft_custom_exception_name_and_message_are_private(self):
        raw = self.raw(soft=True)
        error = type(MARKER, (RuntimeError,), {})
        raw.dump_info.side_effect = error(MARKER)
        with self.assertLogs(self.watchdog.logger, level="ERROR") as logs, patch.object(
            self.watchdog, "pyspy_dump_schedulers", side_effect=error(MARKER)
        ), patch.object(self.watchdog, "_best_effort_flush"):
            raw._handle_timeout()
        self.assertNotIn(MARKER, "\n".join(logs.output))
        self.assertIn("type=Exception", "\n".join(logs.output))

    def test_subprocess_crash_signal_precedes_broken_logger(self):
        proc = MagicMock(pid=456, exitcode=2)
        proc.is_alive.return_value = False
        watchdog = self.watchdog.SubprocessWatchdog([proc], process_names=[MARKER])
        events = []

        def log(*_, **__):
            events.append("log")
            raise BrokenPipeError(MARKER)

        with patch.object(self.watchdog.os, "getpid", return_value=123), patch.object(
            self.watchdog.os, "kill", side_effect=lambda *args: events.append("SIGQUIT")
        ) as kill, patch.object(self.watchdog.logger, "error", side_effect=log):
            self.assertTrue(watchdog._check_processes())
        kill.assert_called_once_with(123, signal.SIGQUIT)
        self.assertEqual(events, ["SIGQUIT", "log"])

    def test_pyspy_success_captures_but_never_logs_stdout_or_stderr(self):
        result = NS(stdout=MARKER, stderr=MARKER)
        with patch.object(
            self.dump.psutil, "Process", return_value=NS(pid=123)
        ), patch.object(
            self.dump.subprocess, "run", return_value=result
        ) as run, self.assertLogs(
            self.dump.logger, level="ERROR"
        ) as logs:
            self.dump.pyspy_dump_schedulers()
        output = "\n".join(logs.output)
        self.assertNotIn(MARKER, output)
        self.assertIn("completed", output)
        self.assertIn("pid=123", output)
        self.assertEqual(
            run.call_args.args[0], ["py-spy", "dump", "--native", "--pid", "123"]
        )
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertTrue(run.call_args.kwargs["capture_output"])
        self.assertEqual(
            run.call_args.kwargs["timeout"], self.dump.PYSPY_DUMP_TIMEOUT_SECONDS
        )

    def test_pyspy_failure_does_not_log_command_or_output(self):
        error = subprocess.CalledProcessError(1, MARKER, output=MARKER, stderr=MARKER)
        with patch.object(
            self.dump.psutil, "Process", return_value=NS(pid=123)
        ), patch.object(
            self.dump.subprocess, "run", side_effect=error
        ) as run, self.assertLogs(
            self.dump.logger, level="ERROR"
        ) as logs:
            self.dump.pyspy_dump_schedulers()
        self.assertEqual(run.call_count, 2)
        self.assertNotIn(MARKER, "\n".join(logs.output))
        self.assertIn("CalledProcessError", "\n".join(logs.output))

    def test_pyspy_retry_survives_broken_logger_and_partial_timeout_output(self):
        timeout = subprocess.TimeoutExpired(MARKER, 10, output=MARKER, stderr=MARKER)
        with patch.object(
            self.dump.psutil, "Process", return_value=NS(pid=123)
        ), patch.object(
            self.dump.subprocess,
            "run",
            side_effect=[timeout, NS(stdout=MARKER, stderr=MARKER)],
        ) as run, patch.object(
            self.dump.logger, "error", side_effect=OSError(errno.ENOSPC, MARKER)
        ):
            self.dump.pyspy_dump_schedulers()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[1].args[0], ["py-spy", "dump", "--pid", "123"]
        )
        self.assertTrue(
            all(
                call.kwargs["timeout"] == self.dump.PYSPY_DUMP_TIMEOUT_SECONDS
                for call in run.call_args_list
            )
        )

    def test_cuda_pipe_failure_and_broken_log_do_not_skip_next_target(self):
        procs = [NS(pid=123), NS(pid=456)]
        with patch.object(
            self.dump, "collect_scheduler_processes", return_value=procs
        ), patch.object(
            self.dump, "_resolve_cuda_coredump_pipe_path", return_value=Path(MARKER)
        ), patch.object(
            self.dump.os, "open", side_effect=[OSError(errno.EIO, MARKER), 7]
        ) as opening, patch.object(
            self.dump.os, "write"
        ) as writing, patch.object(
            self.dump.os, "close"
        ) as closing, patch.object(
            self.dump.logger, "error", side_effect=BrokenPipeError(MARKER)
        ):
            self.dump.trigger_cuda_user_coredump(scheduler_only=True)
        self.assertEqual(opening.call_count, 2)
        writing.assert_called_once_with(7, b"1")
        closing.assert_called_once_with(7)

    def test_cuda_pipe_path_and_exception_text_are_private(self):
        with patch.object(
            self.dump.psutil, "Process", return_value=NS(pid=123)
        ), patch.object(
            self.dump, "_resolve_cuda_coredump_pipe_path", return_value=Path(MARKER)
        ), patch.object(
            self.dump.os, "open", side_effect=OSError(errno.EIO, MARKER)
        ), self.assertLogs(
            self.dump.logger, level="ERROR"
        ) as logs:
            self.dump.trigger_cuda_user_coredump()
        self.assertNotIn(MARKER, "\n".join(logs.output))
        self.assertIn("pid=123", "\n".join(logs.output))
        self.assertIn("OSError", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
