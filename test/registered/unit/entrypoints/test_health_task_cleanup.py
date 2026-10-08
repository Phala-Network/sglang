"""Execute the actual health endpoint with CPU-only dependency fixtures."""
import ast
import asyncio
from contextlib import aclosing
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/entrypoints/http_server.py"


class HealthTaskTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = 100.0
        self.closed = False
        self.child = None
        self.mode = "pending"
        self.heartbeat = True
        self.rid = "HEALTH_CHECK_fixed"
        self.manager = NS(
            gracefully_exit=False, server_status="Up", is_generation=True,
            last_receive_tstamp=0, rid_to_state={}, abort_request=Mock(),
        )

        async def generate(obj, request):
            self.child = asyncio.current_task()
            self.manager.rid_to_state[obj.rid] = object()
            try:
                if self.mode.startswith("disconnect"):
                    kind = self.mode[-1]
                    raise ValueError(
                        "Request is disconnected from the client side "
                        f"(type {kind}). Abort request obj.rid={obj.rid!r}"
                    )
                if self.mode == "wrong_rid":
                    raise ValueError(
                        "Request is disconnected from the client side "
                        "(type 1). Abort request obj.rid='CUSTOMER_1'"
                    )
                if self.mode == "error":
                    raise RuntimeError("real engine failure")
                if self.mode == "value_error":
                    raise ValueError("invalid model input")
                if self.mode == "response":
                    yield {}
                else:
                    await asyncio.Event().wait()
            finally:
                self.closed = True

        async def sleep(_):
            await asyncio.sleep(0)
            self.clock += 1
            if self.heartbeat:
                self.manager.last_receive_tstamp = self.clock
            if self.mode == "cancel_handler":
                raise asyncio.CancelledError

        self.manager.generate_request = generate
        flag = lambda value: NS(get=lambda: value)
        self.ns = {
            "asyncio": NS(create_task=asyncio.create_task, sleep=sleep,
                          CancelledError=asyncio.CancelledError),
            "aclosing": aclosing,
            "_global_state": NS(tokenizer_manager=self.manager),
            "ServerStatus": NS(Starting="Starting", Up="Up", UnHealthy="UnHealthy"),
            "Response": lambda **kw: NS(**kw), "Request": object,
            "GenerateReqInput": lambda **kw: NS(**kw),
            "EmbeddingReqInput": lambda **kw: NS(**kw),
            "get_disagg": lambda: NS(disaggregation_mode="null"),
            "DisaggregationMode": NS(NULL=NS(value="null")),
            "envs": NS(SGLANG_DIAG_BYPASS_HEALTH_GENERATE=flag(False),
                       SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=flag(True)),
            "uuid": NS(uuid4=lambda: NS(hex="fixed")),
            "HEALTH_CHECK_RID_PREFIX": "HEALTH_CHECK",
            "HEALTH_CHECK_TIMEOUT": 2,
            "time": NS(time=lambda: self.clock, localtime=lambda t: t,
                       strftime=lambda *args: "fixture"),
            "logger": Mock(),
        }
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        node = next(n for n in tree.body
                    if isinstance(n, ast.AsyncFunctionDef) and n.name == "health_generate")
        node.decorator_list = []
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), self.ns)

    async def run_endpoint(self):
        return await self.ns["health_generate"](NS(url=NS(path="/health")))

    def assert_clean(self):
        self.assertTrue(self.closed)
        self.assertTrue(self.child.done())
        self.assertNotIn(self.rid, self.manager.rid_to_state)
        self.manager.abort_request.assert_called_once_with(self.rid)

    async def test_heartbeat_cancels_and_awaits_pending_probe(self):
        self.assertEqual((await self.run_endpoint()).status_code, 200)
        self.assert_clean()

    async def test_timeout_cleans_pending_probe(self):
        self.heartbeat = False
        self.assertEqual((await self.run_endpoint()).status_code, 503)
        self.assertEqual(self.manager.server_status, "UnHealthy")
        self.assert_clean()

    async def test_handler_cancellation_is_propagated_and_cleans_child(self):
        self.mode = "cancel_handler"
        with self.assertRaises(asyncio.CancelledError):
            await self.run_endpoint()
        self.assert_clean()

    async def test_completed_generator_is_closed(self):
        self.mode = "response"
        self.assertEqual((await self.run_endpoint()).status_code, 200)
        self.assert_clean()

    async def test_known_disconnects_are_retrieved(self):
        for mode in ("disconnect1", "disconnect3"):
            self.setUp()
            self.mode = mode
            self.assertEqual((await self.run_endpoint()).status_code, 200)
            self.assert_clean()

    async def test_disconnect_without_heartbeat_does_not_mark_healthy(self):
        self.mode, self.heartbeat = "disconnect1", False
        self.assertEqual((await self.run_endpoint()).status_code, 503)
        self.assert_clean()

    async def test_unrelated_exceptions_are_not_swallowed(self):
        for mode, exception in (("error", RuntimeError), ("value_error", ValueError),
                                ("wrong_rid", ValueError), ("disconnect2", ValueError)):
            self.setUp()
            self.mode = mode
            with self.assertRaises(exception):
                await self.run_endpoint()
            self.assert_clean()

    async def test_starting_server_creates_no_child(self):
        self.manager.server_status = "Starting"
        self.assertEqual((await self.run_endpoint()).status_code, 503)
        self.assertIsNone(self.child)
        self.manager.abort_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
