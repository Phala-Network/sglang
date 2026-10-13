"""Runner contracts; actual image execution is separate builder evidence."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "verify_unified_test", HERE / "verify_unified_image.py"
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class ImageRunnerTests(unittest.TestCase):
    def test_moving_tag_is_rejected_without_starting_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(VERIFY.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "exact builder image"):
                    VERIFY.verify(
                        "repo:latest", Path("unused"), Path(directory) / "out"
                    )
                run.assert_not_called()

    def run_fixture(self, cpu_exit):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = root / "context.json"
            context.write_text(
                json.dumps(
                    {"engine": {"commit": "fixture"}, "patches_commit": "fixture"}
                )
            )
            commands = []

            def run(argv, **kwargs):
                commands.append(argv)
                self.assertNotIn("-v", argv)
                self.assertFalse(any(arg.startswith("--mount") for arg in argv))
                self.assertIn("--pull=never", argv)
                self.assertEqual(argv[argv.index("--runtime") + 1], "runc")
                self.assertEqual(argv[argv.index("--network") + 1], "none")
                kwargs["stdout"].write("synthetic runner output\n")
                return SimpleNamespace(
                    returncode=1
                    if argv[-1] == "check"
                    else cpu_exit
                    if kwargs["input"] is not None
                    else 0
                )

            with patch.object(VERIFY.subprocess, "run", side_effect=run):
                report = VERIFY.verify("sha256:" + "1" * 64, context, root / "out")
            self.assertEqual(len(commands), 5)
            self.assertTrue((root / "out/result.json").is_file())
            return report

    def test_dependency_findings_are_not_hidden_by_successful_cpu_gates(self):
        report = self.run_fixture(0)
        self.assertTrue(report["cpu_gates_passed"])
        self.assertTrue(report["dependency_findings_require_review"])
        self.assertFalse(report["gpu_or_production_qualified"])

    def test_native_gate_failure_cannot_become_success(self):
        report = self.run_fixture(1)
        self.assertFalse(report["cpu_gates_passed"])
        self.assertFalse(report["gpu_or_production_qualified"])


if __name__ == "__main__":
    unittest.main()
