"""Focused tests for bounded watchdog diagnostics and process reaping."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.utils import common
from sglang.srt.utils import cudacore_pyspy_dump_utils as dump_utils
from sglang.test.test_utils import CustomTestCase


class TestPyspyTimeout(CustomTestCase):
    def test_each_pyspy_attempt_has_a_bounded_timeout(self):
        process = MagicMock(pid=123)
        timeout = subprocess.TimeoutExpired("py-spy", 10)

        with patch.object(dump_utils.psutil, "Process", return_value=process), patch.object(
            dump_utils.subprocess,
            "run",
            side_effect=timeout,
        ) as run:
            dump_utils.pyspy_dump_schedulers()

        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(
                call.kwargs["timeout"], dump_utils.PYSPY_DUMP_TIMEOUT_SECONDS
            )


class TestKillProcessTreeReap(CustomTestCase):
    def test_default_waits_for_reap(self):
        default = inspect.signature(common.kill_process_tree).parameters[
            "wait_timeout"
        ].default
        self.assertEqual(default, 60)

        parent = MagicMock()
        child = MagicMock(pid=456)
        parent.children.return_value = [child]

        with patch.object(common.psutil, "Process", return_value=parent), patch.object(
            common, "_wait_for_reap_or_raise"
        ) as wait:
            common.kill_process_tree(123, include_parent=False)

        child.kill.assert_called_once_with()
        wait.assert_called_once_with([child], 60)

    def test_target_disappearing_during_child_walk_is_ignored(self):
        parent = MagicMock()
        parent.children.side_effect = common.psutil.NoSuchProcess(pid=123)

        with patch.object(common.psutil, "Process", return_value=parent):
            common.kill_process_tree(123)


if __name__ == "__main__":
    unittest.main(verbosity=2)
