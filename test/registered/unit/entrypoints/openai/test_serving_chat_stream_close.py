"""Cancellation regressions for OpenAI chat SSE responses."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from starlette.requests import ClientDisconnect

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat  # noqa: E402


class TestStreamingResponseDisconnectCleanup(unittest.TestCase):
    def _make_serving(self, closed: asyncio.Event):
        serving = object.__new__(OpenAIServingChat)

        async def generate(*_args):
            try:
                yield "data: first\n\n"
                yield "data: second\n\n"
            finally:
                closed.set()

        serving._generate_chat_stream = generate
        serving.tokenizer_manager = Mock()
        serving.tokenizer_manager.create_abort_task.return_value = None
        return serving

    def test_asgi_24_send_oserror_closes_sglang_generator(self):
        async def drive():
            closed = asyncio.Event()
            serving = self._make_serving(closed)
            response = await serving._handle_streaming_request(
                SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
            )

            async def receive():
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.body":
                    raise OSError("client socket closed")

            scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
            with self.assertRaises(ClientDisconnect):
                await response(scope, receive, send)
            self.assertTrue(closed.is_set())

        asyncio.run(drive())

    def test_response_start_oserror_closes_prefetched_generator(self):
        async def drive():
            closed = asyncio.Event()
            serving = self._make_serving(closed)
            response = await serving._handle_streaming_request(
                SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
            )

            async def receive():
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.start":
                    raise OSError("client socket closed before body")

            scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
            with self.assertRaises(ClientDisconnect):
                await response(scope, receive, send)
            self.assertTrue(closed.is_set())

        asyncio.run(drive())

    def test_send_cancelled_error_closes_sglang_generator_and_propagates(self):
        async def drive():
            closed = asyncio.Event()
            serving = self._make_serving(closed)
            response = await serving._handle_streaming_request(
                SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
            )

            async def receive():
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.body":
                    raise asyncio.CancelledError()

            scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
            with self.assertRaises(asyncio.CancelledError):
                await response(scope, receive, send)
            self.assertTrue(closed.is_set())

        asyncio.run(drive())

    def test_asgi_23_disconnect_while_send_waits_closes_generator(self):
        async def drive():
            closed = asyncio.Event()
            body_send_started = asyncio.Event()
            serving = self._make_serving(closed)
            response = await serving._handle_streaming_request(
                SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
            )

            async def receive():
                await body_send_started.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.body":
                    body_send_started.set()
                    await asyncio.Event().wait()

            scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
            await response(scope, receive, send)
            self.assertTrue(closed.is_set())

        asyncio.run(drive())

    def test_asgi_23_immediate_disconnect_closes_prefetched_generator(self):
        async def drive():
            closed = asyncio.Event()
            serving = self._make_serving(closed)
            response = await serving._handle_streaming_request(
                SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
            )

            async def receive():
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.start":
                    await asyncio.Event().wait()

            scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
            await response(scope, receive, send)
            self.assertTrue(closed.is_set())

        asyncio.run(drive())

    def test_normal_stream_completion_still_closes_generator(self):
        async def drive():
            closed = asyncio.Event()
            serving = self._make_serving(closed)
            response = await serving._handle_streaming_request(
                SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
            )
            messages = []

            async def receive():
                return {"type": "http.disconnect"}

            async def send(message):
                messages.append(message)

            scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
            await response(scope, receive, send)

            bodies = [
                message["body"]
                for message in messages
                if message["type"] == "http.response.body" and message["body"]
            ]
            self.assertEqual(bodies, [b"data: first\n\n", b"data: second\n\n"])
            self.assertTrue(closed.is_set())

        asyncio.run(drive())


if __name__ == "__main__":
    unittest.main(verbosity=2)
