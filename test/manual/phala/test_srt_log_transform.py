"""CPU contracts for deterministic privacy migration and preserved evaluations."""

import contextlib
import importlib.util
import io
import logging
import sys
import traceback
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "srt_privacy_transform", ROOT / "scripts/phala/redact_srt_logs.py"
)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "srt_privacy_verify", ROOT / "scripts/phala/verify_srt_privacy.py"
)
VERIFY = importlib.util.module_from_spec(VERIFY_SPEC)
with patch.dict(sys.modules, {"redact_srt_logs": TOOL}):
    VERIFY_SPEC.loader.exec_module(VERIFY)
RELATIVE = "entrypoints/privacy_fixture.py"
MARKER = "SYNTHETIC_PRIVATE_SRT_MARKER"


class Sink(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1
        super().flush()


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.policy = TOOL.load_policy()
        self.stream = io.StringIO()
        self.logger = logging.Logger("privacy-fixture", logging.DEBUG)
        handler = logging.StreamHandler(self.stream)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        self.logger.addHandler(handler)
        self.addCleanup(handler.close)
        self.namespace = {
            "logger": self.logger,
            "secret": MARKER,
            "logging": logging,
            "warnings": warnings,
            "traceback": traceback,
            "sys": sys,
        }

    def review(self, expression, action, relative=RELATIVE):
        self.policy["evaluations"][
            (relative, TOOL.dump(TOOL.ast.parse(expression, mode="eval").body))
        ] = action

    def converted(self, source, relative=RELATIVE):
        after, changes = TOOL.transform(source, relative, self.policy)
        second, repeat = TOOL.transform(after, relative, self.policy)
        self.assertEqual(second, after)
        self.assertEqual(repeat, [])
        self.assertEqual(TOOL.nonlog_ast(source), TOOL.nonlog_ast(after))
        return after, changes

    def run_source(self, source):
        after, _ = self.converted(source)
        exec(compile(after, "privacy_fixture.py", "exec"), self.namespace)
        self.assertNotIn(MARKER, self.stream.getvalue())
        return after

    def test_lazy_and_preformatted_messages_are_redacted(self):
        self.run_source(
            "logger.warning('request %s', secret)\nlogger.error(f'failed {secret}')\n"
        )
        self.assertIn("WARNING", self.stream.getvalue())
        self.assertIn("ERROR", self.stream.getvalue())
        self.assertIn("<redacted>", self.stream.getvalue())

    def test_exception_method_keeps_level_without_raw_traceback(self):
        source = (
            "try:\n    raise RuntimeError(secret)\n"
            "except RuntimeError:\n    logger.exception('worker failed')\n"
        )
        after = self.run_source(source)
        self.assertIn("logger.exception", after)
        self.assertIn("exc_info=False", after)
        self.assertIn("ERROR worker failed", self.stream.getvalue())
        self.assertNotIn("Traceback", self.stream.getvalue())

    def test_explicit_exception_and_stack_flags_are_disabled(self):
        self.run_source("logger.error('failure', exc_info=True, stack_info=True)\n")
        self.assertNotIn("Stack", self.stream.getvalue())

    def test_required_evaluation_is_retained_once_even_when_logging_disabled(self):
        called = []
        self.namespace["operation"] = lambda: called.append(1) or MARKER
        self.review("operation()", "preserve")
        self.logger.disabled = True
        self.run_source("logger.info(f'value={operation()}')\n")
        self.assertEqual(called, [1])
        self.assertEqual(self.stream.getvalue(), "")

    def test_conditional_required_evaluation_keeps_the_original_branch(self):
        called = []
        self.namespace.update(
            operation=lambda: called.append("operation") or MARKER,
            otherwise=lambda: called.append("otherwise") or MARKER,
            enabled=False,
        )
        self.review("operation()", "preserve")
        self.run_source(
            "logger.info(f'value={operation() if enabled else otherwise()}')\n"
        )
        self.assertEqual(called, ["otherwise"])

    def test_required_exception_retrieval_does_not_log_the_result(self):
        called = []
        self.namespace["operation"] = lambda: called.append(1) or RuntimeError(MARKER)
        self.review("operation()", "preserve")
        self.run_source("logger.error('failure', exc_info=operation())\n")
        self.assertEqual(called, [1])

    def test_reviewed_diagnostic_evaluation_is_removed(self):
        called = []
        self.namespace["diagnostic"] = lambda: called.append(1) or MARKER
        self.review("diagnostic()", "drop-diagnostic")
        self.run_source("logger.info('diagnostic=%s', diagnostic())\n")
        self.assertEqual(called, [])

    def test_unknown_removed_evaluation_is_rejected(self):
        with self.assertRaisesRegex(TOOL.ReviewRequired, "Unreviewed"):
            TOOL.transform(
                "logger.info('value=%s', mutation())\n", RELATIVE, self.policy
            )

    def test_explicit_logging_level_is_preserved(self):
        called = []
        self.namespace["level"] = lambda: called.append(1) or logging.CRITICAL
        self.run_source("logger.log(level(), 'value=%s', secret)\n")
        self.assertEqual(called, [1])
        self.assertIn("CRITICAL", self.stream.getvalue())

    def test_internal_print_keeps_stream_end_and_flush(self):
        sink = Sink()
        self.namespace["sink"] = sink
        source = "print(f'error {secret}', file=sink, end='', flush=True)\n"
        after, _ = self.converted(source)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(after, self.namespace)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(sink.flushes, 1)
        self.assertFalse(sink.getvalue().endswith("\n"))
        self.assertNotIn(MARKER, sink.getvalue())

    def test_warning_category_and_location_metadata_are_preserved(self):
        source = (
            "warnings.warn(f'warning {secret}', DeprecationWarning, stacklevel=1)\n"
        )
        after, _ = self.converted(source)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            exec(compile(after, "warning-fixture.py", "exec"), self.namespace)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].category, DeprecationWarning)
        self.assertEqual(records[0].filename, "warning-fixture.py")
        self.assertNotIn(MARKER, str(records[0].message))

    def test_traceback_marker_uses_the_original_stream(self):
        sink = io.StringIO()
        self.namespace["sink"] = sink
        source = (
            "try:\n    raise RuntimeError(secret)\n"
            "except RuntimeError:\n    traceback.print_exc(file=sink)\n"
        )
        after, _ = self.converted(source)
        exec(after, self.namespace)
        self.assertIn("Exception details redacted", sink.getvalue())
        self.assertNotIn(MARKER, sink.getvalue())

    def test_json_target_and_event_remain_intact(self):
        captured = []
        targets = [object(), object()]
        self.namespace.update(
            targets=targets,
            payload={"rid": MARKER},
            log_json=lambda target, event, data: captured.append((target, event, data)),
        )
        self.run_source("log_json(targets, 'request.received', payload)\n")
        self.assertEqual(captured, [(targets, "request.received", {"redacted": True})])

    def test_cli_output_and_validate_call_are_unchanged(self):
        for relative in (
            "entrypoints/ollama/smart_router.py",
            "model_loader/expert_pack/validate.py",
            "model_loader/expert_pack/build.py",
        ):
            source = "print(operation(), end='', flush=True)\n"
            after, changes = TOOL.transform(source, relative, self.policy)
            self.assertEqual(after, source)
            self.assertEqual(changes, [])

    def test_pretty_print_business_output_is_excluded_by_function(self):
        source = (
            "def pretty_print(text):\n    print(text)\n"
            "def diagnostic(text):\n    print(text)\n"
        )
        after, _ = self.converted(
            source, "mem_cache/unified_cache/unified_tree_core.py"
        )
        namespace = {}
        exec(after, namespace)
        business, diagnostic = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(business):
            namespace["pretty_print"](MARKER)
        with contextlib.redirect_stdout(diagnostic):
            namespace["diagnostic"](MARKER)
        self.assertIn(MARKER, business.getvalue())
        self.assertNotIn(MARKER, diagnostic.getvalue())

    def test_generic_safe_sinks_are_not_transformed(self):
        for relative in ("utils/log_utils.py", "utils/framework_log_privacy.py"):
            source = "logger.info(message)\n"
            after, changes = TOOL.transform(source, relative, self.policy)
            self.assertEqual(after, source)
            self.assertEqual(changes, [])

    def test_unknown_control_or_unpacked_arguments_require_review(self):
        for source in (
            "logger.info('message', extra=payload)\n",
            "logger.info(*arguments)\n",
            "logger.warning(msg=operation(), stacklevel=2)\n",
            "print(secret, end=secret)\n",
        ):
            with self.subTest(source=source), self.assertRaises(TOOL.ReviewRequired):
                TOOL.transform(source, RELATIVE, self.policy)

    def test_bom_and_unicode_offsets_are_preserved(self):
        source = "\ufeffmessage = '\\u4f60\\u597d'; logger.warning(f'value={secret}')\n"
        after, _ = self.converted(source)
        self.assertTrue(after.startswith("\ufeff"))
        self.assertIn("message =", after)
        compile(after.encode("utf-8"), "bom-fixture.py", "exec")

    def test_verifier_rejects_changed_severity_target_and_dropped_log(self):
        before = "logger.warning(f'value={secret}')\n"
        expected, _ = self.converted(before)
        self.assertTrue(
            VERIFY.migration_matches(before, expected, RELATIVE, self.policy)
        )
        for after in (
            expected.replace("logger.warning", "logger.info"),
            expected.replace("logger.warning", "target.warning"),
            "",
        ):
            with self.subTest(after=after):
                self.assertFalse(
                    VERIFY.migration_matches(before, after, RELATIVE, self.policy)
                )

    def test_verifier_rejects_changed_business_output_and_print_controls(self):
        before = "print(secret, end='', flush=True)\n"
        for relative in (RELATIVE, "entrypoints/ollama/smart_router.py"):
            expected, _ = self.converted(before, relative)
            for after in ("", expected.replace("flush=True", "flush=False")):
                with self.subTest(relative=relative, after=after):
                    self.assertFalse(
                        VERIFY.migration_matches(before, after, relative, self.policy)
                    )

    def test_verifier_rejects_lost_required_evaluation(self):
        self.review("operation()", "preserve")
        before = "logger.info(f'value={operation()}')\n"
        expected, _ = self.converted(before)
        self.assertTrue(
            VERIFY.migration_matches(before, expected, RELATIVE, self.policy)
        )
        self.assertFalse(
            VERIFY.migration_matches(
                before, "logger.info('value=<redacted>')\n", RELATIVE, self.policy
            )
        )

    def test_verifier_allows_only_reviewed_import_removal_and_formatting(self):
        relative = "models/llama.py"
        before = (
            "from sglang.utils import get_exception_traceback\n"
            "import os\nlogger.error(f'error={secret}')\n"
        )
        expected, _ = self.converted(before, relative)
        after = expected.replace(
            "from sglang.utils import get_exception_traceback\n", ""
        ).replace("logger.error(", "logger.error(\n    ")
        self.assertTrue(VERIFY.migration_matches(before, after, relative, self.policy))
        self.assertFalse(
            VERIFY.migration_matches(
                before, after.replace("import os\n", ""), relative, self.policy
            )
        )
        for inserted in (
            "from sglang.utils import get_exception_traceback\n",
            "from sglang.utils import get_exception_traceback as os\n",
        ):
            with self.subTest(inserted=inserted):
                self.assertFalse(
                    VERIFY.migration_matches(
                        before, inserted + after, relative, self.policy
                    )
                )

    def test_source_root_must_be_srt_not_its_parent_repository(self):
        TOOL.validate_source_root(ROOT / "python/sglang/srt")
        for root in (ROOT, ROOT / "python/sglang", ROOT / "python/sglang/srt/utils"):
            with self.subTest(root=root), self.assertRaises(TOOL.ReviewRequired):
                TOOL.validate_source_root(root)


if __name__ == "__main__":
    unittest.main()
