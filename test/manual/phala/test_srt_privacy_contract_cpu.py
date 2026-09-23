"""CPU contracts for the deterministic SRT privacy port.

This test intentionally loads the real RequestLogger, SchedulerStatusLogger,
log_utils, encoder drain function, and NIXL selector source files.  Only the
torch/distributed, SGLang environment, and NIXL routing imports are stubbed so
the contracts stay CPU-only and do not copy the implementation under test.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import dataclasses
import importlib.util
import io
import json
import logging
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python" / "sglang" / "srt"
MARKER = "SRT_SYNTHETIC_PRIVATE_CONTRACT_MARKER"


class _EnvValue:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def is_set(self):
        return self.value is not None


class _EnvStub:
    def __init__(self):
        self.SGLANG_LOG_REQUEST_HEADERS = _EnvValue(["x-contract-safe"])
        self.SGLANG_LOG_REQUEST_EXCEEDED_MS = _EnvValue(0)
        self.SGLANG_LOG_SCHEDULER_STATUS_TARGET = _EnvValue("stdout")
        self.SGLANG_LOG_SCHEDULER_STATUS_INTERVAL = _EnvValue(10.0)


class _HeaderRequest:
    def __init__(self, headers):
        self.headers = headers


@dataclasses.dataclass
class _SyntheticRequest:
    rid: str
    text: str | None
    input_ids: object
    prompt: str


class _Tokenizer:
    def __init__(self):
        self.calls = []

    def decode(self, input_ids, *, skip_special_tokens):
        self.calls.append((list(input_ids), skip_special_tokens))
        return "decoded-input"


class _LoggerSpy:
    def __init__(self):
        self.calls = []

    def error(self, *args, **kwargs):
        self.calls.append(("error", args, kwargs))

    def exception(self, *args, **kwargs):
        self.calls.append(("exception", args, kwargs))


class _RecordingTask(asyncio.Task):
    def __init__(self, coro, *, loop):
        self.exception_calls = 0
        super().__init__(coro, loop=loop)

    def exception(self):
        self.exception_calls += 1
        return super().exception()


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source_tree(relative: str) -> ast.Module:
    return ast.parse((SRT / relative).read_text(encoding="utf-8"), filename=relative)


def _parse_json_line(line: str) -> dict:
    line = re.sub(r"^\[[^\]]+\]\s+", "", line.strip())
    return json.loads(line)


def _flush(loggers):
    for logger in loggers:
        for handler in logger.handlers:
            handler.flush()


def _close_contract_handlers():
    for name, value in list(logging.Logger.manager.loggerDict.items()):
        if not isinstance(value, logging.Logger):
            continue
        if not (
            name.startswith("contract.")
            or name.startswith("sglang.srt.utils.request_logger")
            or name.startswith("sglang.srt.utils.scheduler_status_logger")
        ):
            continue
        for handler in list(value.handlers):
            handler.flush()
            handler.close()
            value.removeHandler(handler)


@contextlib.contextmanager
def _temporary_log_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            yield temp_dir
        finally:
            _close_contract_handlers()


def _extract_function(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class SrtPrivacyContractCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_modules = {}
        module_names = (
            "torch",
            "torch.distributed",
            "sglang",
            "sglang.srt",
            "sglang.srt.environ",
            "sglang.srt.utils",
            "sglang.srt.utils.log_utils",
            "sglang.srt.utils.request_logger",
            "sglang.srt.utils.scheduler_status_logger",
            "sglang.srt.mem_cache",
            "sglang.srt.mem_cache.storage",
            "sglang.srt.mem_cache.storage.nixl",
            "sglang.srt.mem_cache.storage.nixl.nixl_routing",
            "contract_nixl_utils",
        )
        cls._missing = missing = object()
        for name in module_names:
            cls._saved_modules[name] = sys.modules.get(name, missing)

        torch = types.ModuleType("torch")
        torch.__path__ = []
        distributed = types.ModuleType("torch.distributed")
        distributed.is_initialized = lambda: False
        distributed.get_rank = lambda: 0
        torch.distributed = distributed
        sys.modules["torch"] = torch
        sys.modules["torch.distributed"] = distributed

        sglang = types.ModuleType("sglang")
        sglang.__path__ = [str(ROOT / "python" / "sglang")]
        srt = types.ModuleType("sglang.srt")
        srt.__path__ = [str(SRT)]
        environ = types.ModuleType("sglang.srt.environ")
        environ.envs = _EnvStub()
        utils_package = types.ModuleType("sglang.srt.utils")
        utils_package.__path__ = [str(SRT / "utils")]
        mem_cache = types.ModuleType("sglang.srt.mem_cache")
        mem_cache.__path__ = [str(SRT / "mem_cache")]
        storage = types.ModuleType("sglang.srt.mem_cache.storage")
        storage.__path__ = [str(SRT / "mem_cache" / "storage")]
        nixl = types.ModuleType("sglang.srt.mem_cache.storage.nixl")
        nixl.__path__ = [str(SRT / "mem_cache" / "storage" / "nixl")]

        sglang.srt = srt
        srt.environ = environ
        srt.utils = utils_package
        srt.mem_cache = mem_cache
        mem_cache.storage = storage
        storage.nixl = nixl
        sys.modules.update(
            {
                "sglang": sglang,
                "sglang.srt": srt,
                "sglang.srt.environ": environ,
                "sglang.srt.utils": utils_package,
                "sglang.srt.mem_cache": mem_cache,
                "sglang.srt.mem_cache.storage": storage,
                "sglang.srt.mem_cache.storage.nixl": nixl,
            }
        )

        cls.envs = environ.envs
        cls.log_utils = _load_source_module(
            "sglang.srt.utils.log_utils", SRT / "utils" / "log_utils.py"
        )
        cls.request_logger = _load_source_module(
            "sglang.srt.utils.request_logger", SRT / "utils" / "request_logger.py"
        )
        cls.scheduler_module = _load_source_module(
            "sglang.srt.utils.scheduler_status_logger",
            SRT / "utils" / "scheduler_status_logger.py",
        )

    @classmethod
    def tearDownClass(cls):
        _close_contract_handlers()
        for name, previous in cls._saved_modules.items():
            if previous is cls._missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    def setUp(self):
        _close_contract_handlers()
        self.addCleanup(_close_contract_handlers)
        self.envs.SGLANG_LOG_REQUEST_HEADERS.value = ["x-contract-safe"]
        self.envs.SGLANG_LOG_REQUEST_EXCEEDED_MS.value = 0
        self.envs.SGLANG_LOG_SCHEDULER_STATUS_TARGET.value = "stdout"
        self.envs.SGLANG_LOG_SCHEDULER_STATUS_INTERVAL.value = 10.0

    def _new_request_logger(self, *, level=2, log_format="json", targets=None):
        return self.request_logger.RequestLogger(
            log_requests=True,
            log_requests_level=level,
            log_requests_format=log_format,
            log_requests_target=targets,
        )

    def test_log_utils_preserves_multi_target_event_and_logger_levels(self):
        with _temporary_log_dir() as temp_dir:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                loggers = self.log_utils.create_log_targets(
                    targets=["stdout", temp_dir],
                    name_prefix=f"contract.{id(self)}.targets",
                )
                self.assertEqual(
                    [logger.level for logger in loggers], [logging.INFO] * 2
                )
                self.assertTrue(all(not logger.propagate for logger in loggers))
                self.log_utils.log_json(
                    loggers,
                    "contract.event",
                    {"redacted": True, "safe": "ok"},
                )
                _flush(loggers)

            stdout_record = _parse_json_line(stdout.getvalue().splitlines()[0])
            files = list(Path(temp_dir).glob("*.log"))
            self.assertEqual(len(files), 1)
            file_record = _parse_json_line(
                files[0].read_text(encoding="utf-8").splitlines()[0]
            )
            for record in (stdout_record, file_record):
                self.assertEqual(record["event"], "contract.event")
                self.assertTrue(record["redacted"])
                self.assertEqual(record["safe"], "ok")

    def test_request_payloads_redacted_but_events_and_targets_survive(self):
        obj = _SyntheticRequest(
            rid=MARKER,
            text=MARKER,
            input_ids=[1, 2, 3],
            prompt=MARKER,
        )
        request = _HeaderRequest(
            {
                "x-contract-safe": MARKER,
                "authorization": MARKER,
            }
        )
        output = {"meta_info": {"e2e_latency": 1.0}, "text": MARKER, "rid": MARKER}
        openai_obj = {"messages": [{"role": "user", "content": MARKER}]}

        with _temporary_log_dir() as temp_dir:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                logger = self._new_request_logger(
                    level=2,
                    log_format="json",
                    targets=["stdout", temp_dir],
                )
                logger.log_received_request(obj, request=request)
                logger.log_openai_received_request(openai_obj, request=request)
                logger.log_finished_request(obj, output, request=request)
                _flush(logger.targets)

            stdout_records = [
                _parse_json_line(line) for line in stdout.getvalue().splitlines()
            ]
            files = list(Path(temp_dir).glob("*.log"))
            self.assertEqual(len(files), 1)
            file_records = [
                _parse_json_line(line)
                for line in files[0].read_text(encoding="utf-8").splitlines()
            ]
            expected_events = [
                "request.received",
                "request.received.openai",
                "request.finished",
            ]
            self.assertEqual(
                [record["event"] for record in stdout_records], expected_events
            )
            self.assertEqual(
                [record["event"] for record in file_records], expected_events
            )
            for record in stdout_records + file_records:
                self.assertTrue(record["redacted"])
                self.assertNotIn(MARKER, json.dumps(record))
                self.assertNotIn("rid", record)
                self.assertNotIn("obj", record)
                self.assertNotIn("out", record)
                self.assertNotIn("headers", record)

    def test_text_message_is_fixed_and_still_reaches_both_sinks(self):
        obj = _SyntheticRequest(
            rid=MARKER,
            text=MARKER,
            input_ids=[9],
            prompt=MARKER,
        )
        output = {"meta_info": {"e2e_latency": 1.0}, "text": MARKER}
        with _temporary_log_dir() as temp_dir:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                logger = self._new_request_logger(
                    level=2,
                    log_format="text",
                    targets=["stdout", temp_dir],
                )
                logger.log_received_request(obj)
                logger.log_finished_request(obj, output)
                _flush(logger.targets)

            stdout_lines = [
                line for line in stdout.getvalue().splitlines() if line.strip()
            ]
            files = list(Path(temp_dir).glob("*.log"))
            self.assertEqual(len(files), 1)
            file_lines = [
                line
                for line in files[0].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(stdout_lines), 2)
            self.assertEqual(len(file_lines), 2)
            for line in stdout_lines + file_lines:
                message = re.sub(r"^\[[^\]]+\]\s+", "", line)
                self.assertTrue(message.startswith(("Receive:", "Finish:")))
                self.assertIn("<redacted>", message)
                self.assertNotIn(MARKER, line)

    def test_request_logger_levels_and_invalid_level_contract(self):
        for level in range(4):
            logger = self._new_request_logger(level=level, targets=[])
            max_length, skip_names, out_skip_names = logger.metadata
            self.assertIsNotNone(max_length)
            if level in (0, 1):
                self.assertIn("text", skip_names)
                self.assertIn("output_ids", out_skip_names)
            else:
                self.assertIsNone(skip_names)
                self.assertIsNone(out_skip_names)
            _close_contract_handlers()

        disabled = self.request_logger.RequestLogger(
            log_requests=False,
            log_requests_level=2,
            log_requests_format="json",
            log_requests_target=[],
        )
        self.assertEqual(disabled.metadata, (None, None, None))
        _close_contract_handlers()
        with self.assertRaises(ValueError):
            self._new_request_logger(level=4, targets=[])

    def test_input_ids_decode_and_text_assignment_survive_redaction(self):
        obj = _SyntheticRequest(
            rid=MARKER,
            text=None,
            input_ids=[4, 5],
            prompt=MARKER,
        )
        tokenizer = _Tokenizer()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            logger = self._new_request_logger(
                level=2, log_format="json", targets=["stdout"]
            )
            logger.log_received_request(obj, tokenizer=tokenizer)
            _flush(logger.targets)
        self.assertEqual(obj.text, "decoded-input")
        self.assertEqual(tokenizer.calls, [([4, 5], False)])
        record = _parse_json_line(stdout.getvalue().splitlines()[0])
        self.assertEqual(record["event"], "request.received")
        self.assertTrue(record["redacted"])
        self.assertNotIn(MARKER, stdout.getvalue())

    def test_disabled_logging_and_finished_latency_gate_are_preserved(self):
        disabled = self.request_logger.RequestLogger(
            log_requests=False,
            log_requests_level=2,
            log_requests_format="json",
            log_requests_target=["stdout"],
        )
        _close_contract_handlers()
        obj = _SyntheticRequest(MARKER, MARKER, [1], MARKER)
        disabled.log_received_request(obj, tokenizer=_Tokenizer())
        disabled.log_finished_request(obj, None)

        self.envs.SGLANG_LOG_REQUEST_EXCEEDED_MS.value = 500
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            logger = self._new_request_logger(
                level=2, log_format="json", targets=["stdout"]
            )
            output = {"meta_info": {"e2e_latency": 0.1}, "text": MARKER}
            logger.log_finished_request(obj, output)
            _flush(logger.targets)
        self.assertEqual(stdout.getvalue(), "")

        logger.log_exceeded_ms = 0
        logger.log_finished_request(obj, output)
        _flush(logger.targets)
        record = _parse_json_line(stdout.getvalue().splitlines()[0])
        self.assertEqual(record["event"], "request.finished")
        self.assertTrue(record["redacted"])
        self.assertNotIn(MARKER, stdout.getvalue())

    def test_scheduler_interval_last_dump_event_and_rid_privacy(self):
        stdout = io.StringIO()
        running = types.SimpleNamespace(reqs=[types.SimpleNamespace(rid=MARKER)])
        waiting = [types.SimpleNamespace(rid=MARKER)]
        with contextlib.redirect_stdout(stdout):
            logger = self.scheduler_module.SchedulerStatusLogger(
                targets=["stdout"], dump_interval=10.0
            )
            with patch.object(
                self.scheduler_module,
                "time",
                types.SimpleNamespace(
                    time=unittest.mock.Mock(side_effect=[100.0, 105.0, 111.0])
                ),
            ):
                logger.maybe_dump(running, waiting)
                self.assertEqual(logger.last_dump_time, 100.0)
                logger.maybe_dump(running, waiting)
                self.assertEqual(logger.last_dump_time, 100.0)
                logger.maybe_dump(running, waiting)
                self.assertEqual(logger.last_dump_time, 111.0)
            _flush(logger.loggers)

        records = [_parse_json_line(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["event"], "scheduler.status")
            self.assertTrue(record["redacted"])
        self.assertNotIn(MARKER, stdout.getvalue())

    def test_actual_task_drain_retrieves_exception_but_logs_no_raw_exc_info(self):
        spy = _LoggerSpy()
        function = _extract_function(
            SRT / "disaggregation" / "encoder" / "server.py",
            "await_task_completion_on_cancel",
            {"asyncio": asyncio, "logger": spy},
        )

        async def scenario():
            loop = asyncio.get_running_loop()
            gate = asyncio.Event()

            async def worker():
                await gate.wait()
                raise RuntimeError(MARKER)

            inner = _RecordingTask(worker(), loop=loop)
            outer = asyncio.create_task(function(inner, "contract-drain"))
            for _ in range(3):
                await asyncio.sleep(0)
            outer.cancel()
            gate.set()
            with self.assertRaises(asyncio.CancelledError):
                await outer
            return inner.exception_calls

        exception_calls = asyncio.run(scenario())
        self.assertGreaterEqual(exception_calls, 1)
        self.assertTrue(spy.calls)
        method, args, kwargs = spy.calls[-1]
        self.assertIn(method, {"error", "exception"})
        self.assertFalse(kwargs.get("exc_info", True))
        self.assertNotIn(MARKER, repr(args))
        self.assertNotIn(MARKER, repr(kwargs))

    def test_actual_nixl_getter_is_evaluated_with_fake_agent_only(self):
        routing = types.ModuleType("sglang.srt.mem_cache.storage.nixl.nixl_routing")
        routing._BUCKET_MASK = 0xFF
        routing.BUCKET_HEX_CHARS = 2
        routing.route_key = lambda value: value
        sys.modules[routing.__name__] = routing
        nixl = _load_source_module(
            "contract_nixl_utils",
            SRT / "mem_cache" / "storage" / "nixl" / "nixl_utils.py",
        )

        class FakeAgent:
            def __init__(self):
                self.backend_calls = []
                self.get_params_calls = []

            def get_plugin_list(self):
                return ["POSIX"]

            def create_backend(self, name, initparams):
                self.backend_calls.append((name, dict(initparams)))

            def get_backend_params(self, name):
                self.get_params_calls.append(name)
                return {"private": MARKER}

        agent = FakeAgent()
        selection = nixl.NixlBackendSelection(plugin="POSIX")
        captured = io.StringIO()
        handler = logging.StreamHandler(captured)
        old_level = nixl.logger.level
        nixl.logger.setLevel(logging.INFO)
        nixl.logger.addHandler(handler)
        try:
            self.assertTrue(selection.create_backend(agent))
        finally:
            nixl.logger.removeHandler(handler)
            nixl.logger.setLevel(old_level)
            handler.close()
        self.assertEqual(agent.backend_calls, [("POSIX", {})])
        self.assertEqual(agent.get_params_calls, ["POSIX"])
        self.assertEqual(selection.mem_type, "FILE")
        self.assertTrue(captured.getvalue())
        self.assertNotIn(MARKER, captured.getvalue())
        sys.modules.pop("contract_nixl_utils", None)
        sys.modules.pop(routing.__name__, None)

    def test_exception_sites_keep_method_and_explicit_false(self):
        paths = (
            "server_args.py",
            "entrypoints/http_server.py",
            "disaggregation/encoder/server.py",
        )
        found = []
        for relative in paths:
            tree = _source_tree(relative)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "exception"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "logger"
                ):
                    continue
                found.append((relative, node))
                self.assertTrue(
                    any(
                        keyword.arg == "exc_info"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is False
                        for keyword in node.keywords
                    ),
                    f"{relative}:{node.lineno} must keep logger.exception with exc_info=False",
                )
        self.assertTrue(found)

    def test_traceback_calls_use_fixed_marker(self):
        paths = (
            "disaggregation/encoder/grpc_server.py",
            "disaggregation/encoder/runtime.py",
            "disaggregation/encoder/server.py",
        )
        found_marker = False
        for relative in paths:
            tree = _source_tree(relative)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(
                    node.func, ast.Attribute
                ):
                    continue
                if not (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "traceback"
                ):
                    continue
                self.assertNotEqual(node.func.attr, "print_exc")
                if node.func.attr == "print_exception":
                    rendered = ast.unparse(node)
                    self.assertIn("Exception details redacted", rendered)
                    found_marker = True
        self.assertTrue(found_marker)

    def test_cli_business_prints_are_explicitly_outside_log_scope(self):
        paths = (
            "entrypoints/ollama/smart_router.py",
            "entrypoints/http_server_engine.py",
            "checkpoint_engine/update.py",
            "model_loader/expert_pack/build.py",
            "model_loader/expert_pack/prepare_kimi_manifest.py",
            "model_loader/expert_pack/prepare_kimi_pack.py",
            "model_loader/expert_pack/validate.py",
            "debug_utils/dump_comparator.py",
            "debug_utils/dump_loader.py",
            "debug_utils/model_truncator.py",
            "debug_utils/schedule_simulator/entrypoint.py",
            "debug_utils/schedule_simulator/data_source/data_synthesis.py",
            "debug_utils/text_comparator.py",
            "mem_cache/storage/flexkv/verify_outputs.py",
            "utils/model_file_verifier.py",
            "mem_cache/unified_cache/unified_tree_core.py",
        )
        for relative in paths:
            with self.subTest(relative=relative):
                tree = _source_tree(relative)
                print_calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ]
                self.assertTrue(print_calls)
        smart_router = _source_tree("entrypoints/ollama/smart_router.py")
        self.assertTrue(
            any(
                isinstance(node, ast.keyword) and node.arg == "flush"
                for call in ast.walk(smart_router)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "print"
                for node in call.keywords
            )
        )


if __name__ == "__main__":
    unittest.main()
