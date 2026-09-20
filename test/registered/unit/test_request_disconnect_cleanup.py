"""HTTP cancellation/ownership tests without an inference engine."""

import asyncio
import json
import unittest
from types import SimpleNamespace

from fastapi import HTTPException

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.entrypoints.openai.serving_base import GenerationStreamingResponse
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.entrypoints.request_disconnect import (
    await_response_or_disconnect,
    response_disconnect_watched,
)


class Request:
    def __init__(self):
        self.state = SimpleNamespace()
        self.messages = asyncio.Queue()
        self.receiving = asyncio.Event()
        self.receive_cancelled = False

    async def receive(self):
        self.receiving.set()
        try:
            return await self.messages.get()
        except asyncio.CancelledError:
            self.receive_cancelled = True
            raise


class TestDisconnectCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_cancels_and_joins_pending_work(self):
        request = Request()
        closed = asyncio.Event()

        async def work():
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                closed.set()

        task = asyncio.create_task(await_response_or_disconnect(work(), request))
        await request.receiving.wait()
        self.assertTrue(response_disconnect_watched(request))
        await request.messages.put({"type": "http.disconnect"})
        with self.assertRaisesRegex(ValueError, "disconnected"):
            await task
        self.assertTrue(closed.is_set())
        self.assertFalse(response_disconnect_watched(request))

    async def test_caller_cancellation_closes_work_and_monitor(self):
        request = Request()
        closed = asyncio.Event()

        async def work():
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        task = asyncio.create_task(await_response_or_disconnect(work(), request))
        await request.receiving.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())
        self.assertTrue(request.receive_cancelled)
        self.assertFalse(response_disconnect_watched(request))

    async def test_work_failure_preserves_exception_and_releases_listener(self):
        request = Request()

        async def work():
            await request.receiving.wait()
            raise RuntimeError("engine failed")

        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            await await_response_or_disconnect(work(), request)
        self.assertTrue(request.receive_cancelled)
        self.assertFalse(response_disconnect_watched(request))

    async def test_background_does_not_consume_receive(self):
        request = Request()

        async def work():
            return 7

        self.assertEqual(
            await await_response_or_disconnect(work(), request, background=True), 7
        )
        self.assertFalse(request.receiving.is_set())

    async def test_nested_waiter_uses_existing_listener(self):
        request = Request()
        request.state.sglang_response_disconnect_watched = True

        async def work():
            return 9

        self.assertEqual(await await_response_or_disconnect(work(), request), 9)
        self.assertFalse(request.receiving.is_set())
        self.assertTrue(response_disconnect_watched(request))


class TestStreamingOwnership(unittest.IsolatedAsyncioTestCase):
    scope = {"type": "http", "asgi": {"spec_version": "2.4"}}

    async def receive(self):
        await asyncio.Event().wait()

    async def test_send_failure_closes_suspended_event_and_generation(self):
        closed = []

        async def generation():
            try:
                yield "token"
            finally:
                await asyncio.sleep(0)
                closed.append("generation")

        source = generation()

        async def events():
            try:
                async for token in source:
                    yield token
            finally:
                await asyncio.sleep(0)
                closed.append("events")

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("socket closed")

        response = GenerationStreamingResponse(events(), generation=source)
        with self.assertRaises(Exception):
            await response(self.scope, self.receive, send)
        self.assertEqual(closed, ["events", "generation"])

    async def test_header_send_failure_closes_prestarted_generation(self):
        closed = asyncio.Event()

        async def generation():
            try:
                yield "first"
                yield "next"
            finally:
                closed.set()

        source = generation()
        first = await source.__anext__()

        async def events():
            yield first
            async for item in source:
                yield item

        async def send(message):
            raise OSError("header send failed")

        response = GenerationStreamingResponse(events(), generation=source)
        with self.assertRaises(Exception):
            await response(self.scope, self.receive, send)
        self.assertTrue(closed.is_set())

    async def test_cancel_during_send_closes_suspended_generator(self):
        closed = asyncio.Event()
        sending = asyncio.Event()

        async def events():
            try:
                yield "first"
            finally:
                await asyncio.sleep(0)
                closed.set()

        async def send(message):
            if message["type"] == "http.response.body":
                sending.set()
                await asyncio.Event().wait()

        response = GenerationStreamingResponse(events())
        task = asyncio.create_task(response(self.scope, self.receive, send))
        await sending.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())


class TestResponsesOwnership(unittest.IsolatedAsyncioTestCase):
    async def test_non_harmony_send_failure_closes_inner_events_and_generation(self):
        serving = object.__new__(OpenAIServingResponses)
        request = ResponsesRequest(
            model="model", input="hello", stream=True, store=False
        )
        inner_events = []
        generation_closed = asyncio.Event()
        sent_bodies = []

        # Retain the actual inner generator so GC cannot hide a missing close.
        # Both Responses generator methods below run their production code.
        real_inner = serving._responses_stream_generator_non_harmony

        def capture_inner(*args, **kwargs):
            events = real_inner(*args, **kwargs)
            inner_events.append(events)
            return events

        serving._responses_stream_generator_non_harmony = capture_inner

        async def generation():
            try:
                yield "engine fixture"
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                generation_closed.set()

        source = generation()
        # response.created precedes engine consumption; prime the engine fixture
        # so its finally block is observable even at this earliest send failure.
        await source.__anext__()
        events = serving.responses_stream_generator_non_harmony(
            request, None, source, "model", None, None,
            require_reasoning=False,
        )

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body":
                sent_bodies.append(message["body"])
                raise OSError("socket closed after response.created")

        response = GenerationStreamingResponse(events, generation=source)
        try:
            with self.assertRaises(Exception):
                await response(
                    {"type": "http", "asgi": {"spec_version": "2.4"}},
                    receive,
                    send,
                )
            self.assertEqual(len(sent_bodies), 1)
            self.assertIn(b"event: response.created\n", sent_bodies[0])
            self.assertEqual(len(inner_events), 1)
            self.assertIsNone(inner_events[0].ag_frame)
            self.assertIsNone(events.ag_frame)
            self.assertIsNone(source.ag_frame)
            self.assertTrue(generation_closed.is_set())
        finally:
            # Keep failure cleanup separate from the assertions above.
            await events.aclose()
            for inner in inner_events:
                await inner.aclose()
            await source.aclose()

    async def test_full_response_keeps_http_error_and_closes_generation(self):
        serving = object.__new__(OpenAIServingResponses)
        closed = asyncio.Event()

        async def generation():
            try:
                raise HTTPException(503, "engine unavailable")
                yield
            finally:
                closed.set()

        response = await serving.responses_full_generator(
            SimpleNamespace(background=False), None, generation(), None,
            "model", None, None, require_reasoning=False,
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(response.body)["error"]["code"], 503)
        self.assertTrue(closed.is_set())

    async def test_full_response_cancellation_propagates_after_cleanup(self):
        serving = object.__new__(OpenAIServingResponses)
        started = asyncio.Event()
        closed = asyncio.Event()

        async def generation():
            try:
                started.set()
                await asyncio.Event().wait()
                yield
            finally:
                closed.set()

        task = asyncio.create_task(serving.responses_full_generator(
            SimpleNamespace(background=False), None, generation(), None,
            "model", None, None, require_reasoning=False,
        ))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())


if __name__ == "__main__":
    unittest.main()
