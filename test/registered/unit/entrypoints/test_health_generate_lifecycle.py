import ast
import asyncio
import copy
import gc
import pathlib
import time
import types
import uuid


SOURCE = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python/sglang/srt/entrypoints/http_server.py"
)
EXPECTED_DISCONNECT_PREFIX = (
    "Request is disconnected from the client side (type 1). "
    "Abort request obj.rid='"
)


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


class _Request:
    url = types.SimpleNamespace(path="/health_generate")


class _Status:
    Starting = object()
    Up = object()
    UnHealthy = object()


class _Flag:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _Logger:
    def __init__(self):
        self.debug_calls = []
        self.exception_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))

    def error(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass


class _RequestInput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Manager:
    def __init__(self, behavior, started):
        self.gracefully_exit = False
        self.server_status = _Status.Up
        self.is_generation = True
        self.last_receive_tstamp = 0.0
        self.rid_to_state = {}
        self.behavior = behavior
        self.started = started
        self.cancel_started = asyncio.Event()
        self.allow_cancel = asyncio.Event()
        self.child_task = None

    async def generate_request(self, request_input, request):
        self.rid_to_state[request_input.rid] = object()
        self.child_task = asyncio.current_task()
        self.started.set()
        if self.behavior == "expected-disconnect":
            raise ValueError(f"{EXPECTED_DISCONNECT_PREFIX}{request_input.rid}'")
        if self.behavior == "unrelated-value-error":
            raise ValueError("backend validation failed")
        if self.behavior == "unexpected-error":
            raise RuntimeError("backend probe failed")
        if self.behavior == "pending":
            await asyncio.Event().wait()
        if self.behavior == "pending-slow-cancel":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancel_started.set()
                await self.allow_cancel.wait()
                raise
        if False:
            yield None


def _load_health_generate(manager, logger, sleep, clock):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(
        candidate
        for candidate in tree.body
        if isinstance(candidate, ast.AsyncFunctionDef)
        and candidate.name == "health_generate"
    )
    node = copy.deepcopy(node)
    node.decorator_list = []
    namespace = {
        "asyncio": types.SimpleNamespace(
            CancelledError=asyncio.CancelledError,
            create_task=asyncio.create_task,
            sleep=sleep,
        ),
        "time": clock,
        "uuid": uuid,
        "Request": _Request,
        "Response": _Response,
        "_global_state": types.SimpleNamespace(tokenizer_manager=manager),
        "ServerStatus": _Status,
        "envs": types.SimpleNamespace(
            SGLANG_DIAG_BYPASS_HEALTH_GENERATE=_Flag(False),
            SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=_Flag(True),
        ),
        "HEALTH_CHECK_RID_PREFIX": "HEALTH_CHECK",
        "GenerateReqInput": _RequestInput,
        "EmbeddingReqInput": _RequestInput,
        "DisaggregationMode": types.SimpleNamespace(NULL=types.SimpleNamespace(value="null")),
        "FAKE_BOOTSTRAP_HOST": "127.0.0.1",
        "get_disagg": lambda: types.SimpleNamespace(disaggregation_mode="null"),
        "logger": logger,
        "HEALTH_CHECK_TIMEOUT": 1,
    }
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["health_generate"]


def _loop_errors(loop, errors):
    loop.set_exception_handler(lambda _loop, context: errors.append(context))


def test_success_retrieves_expected_disconnect_and_cleans_rid():
    async def run():
        started = asyncio.Event()
        manager = _Manager("expected-disconnect", started)
        logger = _Logger()

        async def sleep(_delay):
            await asyncio.sleep(0)
            manager.last_receive_tstamp = 1.0

        clock = types.SimpleNamespace(
            time=lambda: 0.0,
            strftime=time.strftime,
            localtime=time.localtime,
        )
        health_generate = _load_health_generate(manager, logger, sleep, clock)
        errors = []
        _loop_errors(asyncio.get_running_loop(), errors)

        response = await health_generate(_Request())
        await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)

        assert response.status_code == 200
        assert manager.rid_to_state == {}
        assert not errors
        assert logger.debug_calls

    asyncio.run(run())


def test_timeout_logs_unrelated_value_error_and_cleans_rid():
    async def run():
        started = asyncio.Event()
        manager = _Manager("unrelated-value-error", started)
        logger = _Logger()
        calls = 0

        async def sleep(_delay):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)

        def clock_time():
            return 0.0 if calls == 0 else 2.0

        clock = types.SimpleNamespace(
            time=clock_time,
            strftime=time.strftime,
            localtime=time.localtime,
        )
        health_generate = _load_health_generate(manager, logger, sleep, clock)
        errors = []
        _loop_errors(asyncio.get_running_loop(), errors)

        response = await health_generate(_Request())
        await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)

        assert response.status_code == 503
        assert manager.rid_to_state == {}
        assert not errors
        assert logger.exception_calls

    asyncio.run(run())


def test_parent_cancellation_drains_child_and_cleans_rid():
    async def run():
        started = asyncio.Event()
        manager = _Manager("pending", started)
        logger = _Logger()

        async def sleep(_delay):
            await asyncio.sleep(0.01)

        clock = types.SimpleNamespace(
            time=lambda: 0.0,
            strftime=time.strftime,
            localtime=time.localtime,
        )
        health_generate = _load_health_generate(manager, logger, sleep, clock)
        task = asyncio.create_task(health_generate(_Request()))
        await asyncio.wait_for(started.wait(), timeout=1)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert manager.rid_to_state == {}
        assert not logger.exception_calls

    asyncio.run(run())


def test_parent_cancellation_during_drain_still_cleans_rid():
    async def run():
        started = asyncio.Event()
        manager = _Manager("pending-slow-cancel", started)
        logger = _Logger()

        async def sleep(_delay):
            await asyncio.sleep(0.01)

        clock = types.SimpleNamespace(
            time=lambda: 0.0,
            strftime=time.strftime,
            localtime=time.localtime,
        )
        health_generate = _load_health_generate(manager, logger, sleep, clock)
        health_task = asyncio.create_task(health_generate(_Request()))
        await asyncio.wait_for(started.wait(), timeout=1)

        health_task.cancel()
        await asyncio.wait_for(manager.cancel_started.wait(), timeout=1)
        health_task.cancel()
        try:
            await health_task
        except asyncio.CancelledError:
            pass

        assert manager.rid_to_state == {}
        manager.allow_cancel.set()
        await asyncio.wait_for(
            asyncio.gather(manager.child_task, return_exceptions=True), timeout=1
        )

    asyncio.run(run())
