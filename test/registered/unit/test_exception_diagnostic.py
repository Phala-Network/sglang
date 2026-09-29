"""CPU contract tests for bounded scheduler exception diagnostics."""

import json
import unittest

from sglang.utils import get_exception_diagnostic


class TestExceptionDiagnostic(unittest.TestCase):
    def test_chained_exception_metadata_excludes_message_and_locals(self):
        secret = "fixture-secret-that-must-not-leak"
        try:
            try:
                raise RuntimeError(secret)
            except RuntimeError as inner:
                raise ValueError("outer message") from inner
        except ValueError:
            diagnostic = get_exception_diagnostic()

        encoded = json.dumps(diagnostic, sort_keys=True)
        self.assertNotIn(secret, encoded)
        self.assertNotIn("outer message", encoded)
        self.assertEqual(
            [entry["type"] for entry in diagnostic["exceptions"]],
            ["builtins.ValueError", "builtins.RuntimeError"],
        )
        for entry in diagnostic["exceptions"]:
            for frame in entry["frames"]:
                self.assertEqual(set(frame), {"file", "function", "line"})


if __name__ == "__main__":
    unittest.main()
