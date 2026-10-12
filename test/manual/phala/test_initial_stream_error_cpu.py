"""Pre-header status and ownership using real ASGI responses, no engine import."""

import ast
import asyncio
import json
import unittest
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional
from unittest.mock import Mock

import anyio
from fastapi.responses import ORJSONResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"


def execute(path, names, ns, owner=None):
    body = ast.parse(path.read_text(encoding="utf-8")).body
    if owner:
        body = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    nodes = [n for n in body if getattr(n, "name", None) in names]
    assert {n.name for n in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)


class TestInitialStreamError(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ns = dict(
            asyncio=asyncio,
            json=json,
            anyio=anyio,
            AsyncExitStack=AsyncExitStack,
            StreamingResponse=StreamingResponse,
            ORJSONResponse=ORJSONResponse,
            Optional=Optional,
            Literal=Literal,
            BaseModel=BaseModel,
        )
        execute(SRT / "entrypoints/openai/protocol.py", {"ErrorResponse"}, self.ns)
        execute(
            SRT / "entrypoints/request_disconnect.py",
            {"response_disconnect_watched", "await_response_or_disconnect"},
            self.ns,
        )
        path = SRT / "entrypoints/openai/serving_base.py"
        execute(path, {"GenerationStreamingResponse"}, self.ns)
        execute(
            path,
            {"create_error_response", "_streaming_response_before_headers"},
            self.ns,
            "OpenAIServingBase",
        )
        self.abort = Mock(return_value=None)
        self.serving = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(create_abort_task=self.abort)
        )
        self.serving.create_error_response = lambda **kw: self.ns[
            "create_error_response"
        ](self.serving, **kw)
        self.closed = []

    def generator(self, chunks, fail_after=False):
        async def generate():
            try:
                for chunk in chunks:
                    yield chunk
                if fail_after:
                    raise AssertionError(
                        "must close instead of requesting another event"
                    )
            finally:
                await asyncio.sleep(0)
                self.closed.append(True)

        return generate()

    async def response(self, generator, request=None):
        return await self.ns["_streaming_response_before_headers"](
            self.serving, generator, None, request
        )

    async def test_initial_status_and_generator_close(self):
        for status in [400, 422, 429, 500, 503]:
            with self.subTest(status=status):
                event = (
                    "data: "
                    + json.dumps(
                        {
                            "error": {
                                "message": "bad schema",
                                "type": "SchemaError",
                                "param": "schema",
                                "code": status,
                            }
                        }
                    )
                    + "\n\n"
                )
                response = await self.response(self.generator([event], fail_after=True))
                self.assertIsInstance(response, ORJSONResponse)
                self.assertEqual(response.status_code, status)
                body = json.loads(response.body)
                self.assertEqual(
                    (body["message"], body["type"], body["param"]),
                    ("bad schema", "SchemaError", "schema"),
                )
        self.assertEqual(len(self.closed), 5)
        self.abort.assert_not_called()

    async def test_missing_code_defaults_to_400(self):
        response = await self.response(self.generator(['data: {"error":{}}\n\n']))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.closed, [True])

    async def test_later_error_preserved_on_real_asgi(self):
        role = 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        error = 'data: {"error":{"message":"late","code":500}}\n\n'
        response = await self.response(self.generator([role, error]))
        sent = []

        async def send(message):
            sent.append(message)

        async def receive():
            await asyncio.Event().wait()

        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        self.assertEqual(sent[0]["status"], 200)
        self.assertEqual(
            b"".join(m.get("body", b"") for m in sent), (role + error).encode()
        )
        self.assertEqual(self.closed, [True])
        self.abort.assert_called_once()

    async def test_nonerror_invalid_json_and_invalid_status_remain_streams(self):
        for chunk in [
            "data: [DONE]\n\n",
            "data: []\n\n",
            'data: {"error":"text"}\n\n',
            'data: {"error":{"code":200}}\n\n',
            'data: {"error":{"code":"500"}}\n\n',
            'data: {"error":{"code":600}}\n\n',
            b"data: bytes\n\n",
        ]:
            response = await self.response(self.generator([chunk]))
            self.assertIsInstance(response, StreamingResponse)
            self.assertEqual([part async for part in response.body_iterator], [chunk])

    async def test_raised_value_error_closes_generator(self):
        async def generate():
            try:
                raise ValueError("invalid grammar")
                yield
            finally:
                self.closed.append(True)

        # Production ValueError path passes the message positionally.
        self.serving.create_error_response = lambda *a, **kw: self.ns[
            "create_error_response"
        ](self.serving, *a, **kw)
        response = await self.response(generate())
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.closed, [True])

    async def test_send_failure_keeps_owned_generator_cleanup(self):
        response = await self.response(self.generator(["first", "second"]))

        async def send(_):
            raise OSError("socket closed")

        async def receive():
            await asyncio.Event().wait()

        with self.assertRaises(Exception):
            await response(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
            )
        self.assertEqual(self.closed, [True])

    async def test_ablation_without_initial_error_promotion_returns_200(self):
        path = SRT / "entrypoints/openai/serving_base.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef)
            and n.name == "_streaming_response_before_headers"
        )
        method.body = [
            n
            for n in method.body
            if not (isinstance(n, ast.If) and "first_chunk" in ast.unparse(n.test))
        ]
        ns = dict(self.ns)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        event = 'data: {"error":{"code":503}}\n\n'
        response = await ns["_streaming_response_before_headers"](
            self.serving, self.generator([event]), None, None
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual([part async for part in response.body_iterator], [event])


if __name__ == "__main__":
    unittest.main()
