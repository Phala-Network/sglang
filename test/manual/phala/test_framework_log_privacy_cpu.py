"""Source/stdlib checks; final-image HTTP and native framework gates are separate."""

import ast
import copy
import importlib.util
import io
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"
SPEC = importlib.util.spec_from_file_location(
    "framework_log_privacy_source", SRT / "utils/framework_log_privacy.py"
)
PRIVACY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRIVACY)
MARKER = "SYNTHETIC_PRIVATE_FRAMEWORK_MARKER"
FILTER_NAME = "sglang_framework_log_privacy"


def source_tree(relative):
    return ast.parse((SRT / relative).read_text(encoding="utf-8-sig"))


class FrameworkPrivacyTests(unittest.TestCase):
    def render(self, name, message, args=(), exception=None):
        record = logging.LogRecord(
            name, logging.WARNING, __file__, 1, message, args, exception
        )
        record.message = MARKER
        record.color_message = MARKER
        record.stack_info = MARKER if exception else None
        record.exc_text = MARKER if exception else None
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        handler.addFilter(PRIVACY.FrameworkPrivacyFilter())
        handler.handle(record)
        handler.close()
        self.assertEqual(record.levelno, logging.WARNING)
        self.assertNotIn("color_message", record.__dict__)
        self.assertNotIn(MARKER, stream.getvalue())
        return record, stream.getvalue()

    def test_uvicorn_access_keeps_formatter_tuple_and_status(self):
        record, output = self.render(
            "uvicorn.access",
            '%s - "%s %s HTTP/%s" %d',
            (MARKER, "POST", "/" + MARKER, "1.1", 503),
        )
        self.assertEqual(len(record.args), 5)
        self.assertIn("POST", output)
        self.assertIn("503", output)

    def test_bad_access_method_and_status_are_not_interpolated(self):
        record, _ = self.render(
            "uvicorn.access",
            MARKER,
            (MARKER, MARKER, MARKER, MARKER, MARKER),
        )
        self.assertEqual(record.args[1], "OTHER")
        self.assertEqual(record.args[-1], 0)

    def test_granian_access_keeps_method_and_status(self):
        _, output = self.render(
            "granian.access",
            "%(path)s",
            {"path": MARKER, "method": "GET", "status": 200},
        )
        self.assertIn("GET", output)
        self.assertIn("200", output)

    def test_exception_and_context_are_removed(self):
        try:
            raise RuntimeError(MARKER)
        except RuntimeError:
            record, output = self.render(
                "uvicorn.error", MARKER, exception=sys.exc_info()
            )
        self.assertIn("RuntimeError", output)
        self.assertIsNone(record.exc_info)
        self.assertIsNone(record.exc_text)
        self.assertIsNone(record.stack_info)

    def test_customer_controlled_exception_class_name_is_not_logged(self):
        error = type(MARKER, (RuntimeError,), {})
        _, output = self.render("_granian", MARKER, exception=(error, error(), None))
        self.assertIn("type=Exception", output)

    def test_preformatted_and_argument_messages(self):
        for logger in ("uvicorn.error", "uvicorn.asgi", "_granian", "granian"):
            with self.subTest(logger=logger):
                self.render(logger, MARKER)
                self.render(logger, "request %s", (MARKER,))

    def test_static_lifecycle_event_is_preserved(self):
        _, output = self.render("uvicorn.error", "Application startup complete.")
        self.assertIn("Application startup complete.", output)
        self.assertIn("WARNING", output)

    def test_other_loggers_are_not_changed(self):
        record = logging.LogRecord(
            "sglang.test", logging.INFO, __file__, 1, "ok %s", (1,), None
        )
        before = copy.copy(record.__dict__)
        self.assertTrue(PRIVACY.FrameworkPrivacyFilter().filter(record))
        self.assertEqual(record.__dict__, before)

    def test_repeated_configure_preserves_path_filter_before_privacy(self):
        config = {
            "handlers": {
                "access": {"filters": [FILTER_NAME, "path_filter", FILTER_NAME]}
            },
            "loggers": {"uvicorn.access": {"level": "WARNING"}},
        }
        for _ in range(3):
            self.assertIs(PRIVACY.configure_framework_log_privacy(config), config)
            self.assertEqual(
                config["handlers"]["access"]["filters"], ["path_filter", FILTER_NAME]
            )
            self.assertEqual(config["loggers"]["uvicorn.access"]["level"], "WARNING")

    def test_path_exclusion_runs_before_redaction_with_real_handlers(self):
        class PathFilter(logging.Filter):
            def filter(self, record):
                return not record.args[2].startswith("/metrics")

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(PathFilter())
        handler.addFilter(PRIVACY.FrameworkPrivacyFilter())
        try:
            for path in ("/metrics?" + MARKER, "/included/" + MARKER):
                record = logging.LogRecord(
                    "uvicorn.access",
                    logging.INFO,
                    __file__,
                    1,
                    '%s - "%s %s HTTP/%s" %d',
                    (MARKER, "GET", path, "1.1", 200),
                    None,
                )
                handler.handle(record)
                if path.startswith("/metrics"):
                    self.assertEqual(stream.getvalue(), "")
            self.assertIn("GET", stream.getvalue())
            self.assertNotIn(MARKER, stream.getvalue())
        finally:
            handler.close()

    def test_removing_filter_restores_the_marker_leak(self):
        record = logging.LogRecord(
            "uvicorn.error", logging.ERROR, __file__, 1, "request %s", (MARKER,), None
        )
        self.assertIn(MARKER, logging.Formatter().format(record))
        PRIVACY.FrameworkPrivacyFilter().filter(record)
        self.assertNotIn(MARKER, logging.Formatter().format(record))

    def test_granian_configuration_is_a_copy(self):
        log_module = types.ModuleType("granian.log")
        log_module.LOGGING_CONFIG = {"handlers": {"default": {}}}
        with patch.dict(sys.modules, {"granian.log": log_module}):
            config = PRIVACY.configure_granian_log_privacy()
        self.assertIn(FILTER_NAME, config["filters"])
        self.assertEqual(log_module.LOGGING_CONFIG, {"handlers": {"default": {}}})

    def test_common_installer_is_inside_function_after_path_configuration(self):
        tree = source_tree("utils/common.py")
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "set_uvicorn_logging_configs"
        )
        calls = {
            node.func.id: node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertLess(
            calls["_configure_uvicorn_access_log_filter"],
            calls["configure_framework_log_privacy"],
        )
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                self.assertNotEqual(
                    getattr(node.value.func, "id", None),
                    "configure_framework_log_privacy",
                )

    def test_standalone_uvicorn_sites_are_configured(self):
        files = (
            "disaggregation/encoder/http_server.py",
            "disaggregation/encoder/receiver.py",
            "distributed/gated_launch.py",
            "entrypoints/engine_info_bootstrap_server.py",
            "mem_cache/storage/hf3fs/mini_3fs_metadata_server.py",
            "utils/common.py",
        )
        for relative in files:
            with self.subTest(relative=relative):
                tree = source_tree(relative)
                functions = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                for function in functions:
                    calls = [
                        node
                        for node in ast.walk(function)
                        if isinstance(node, ast.Call)
                    ]
                    starts = [
                        node.lineno
                        for node in calls
                        if isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "uvicorn"
                        and node.func.attr in {"run", "Config"}
                    ]
                    if starts:
                        installs = [
                            node.lineno
                            for node in calls
                            if isinstance(node.func, ast.Name)
                            and node.func.id == "configure_framework_log_privacy"
                        ]
                        self.assertTrue(installs)
                        self.assertLess(min(installs), min(starts))

    def test_main_uvicorn_and_granian_share_the_source_integration(self):
        tree = source_tree("entrypoints/http_server.py")
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_setup_and_run_http_server"
        )
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        installer = next(
            node
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "set_uvicorn_logging_configs"
        )
        starts = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "uvicorn"
        ]
        self.assertTrue(starts)
        self.assertTrue(all(installer.lineno < node.lineno for node in starts))
        granian = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_granian_server"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.keyword)
                and node.arg == "log_dictconfig"
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None)
                == "configure_granian_log_privacy"
                for node in ast.walk(granian)
            )
        )


if __name__ == "__main__":
    unittest.main()
