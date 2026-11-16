"""HTTP cancellation/ownership tests without an inference engine."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints.openai.serving_base import GenerationStreamingResponse
from sglang.srt.managers.tokenizer_manager import TokenizerManager
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


class TestTokenizerDisconnectReceiveOwnership(unittest.IsolatedAsyncioTestCase):
    async def test_real_waiter_does_not_compete_with_preheader_monitor(self):
        # Run both branches of the actual TokenizerManager waiter: a timeout
        # with no output, and a partial non-streaming output that cannot finish.
        for has_partial_output in (False, True):
            with self.subTest(has_partial_output=has_partial_output):

                class OwnedRequest(Request):
                    def __init__(self):
                        super().__init__()
                        self.readers = 0
                        self.max_readers = 0
                        self.poll_calls = 0

                    async def receive(self):
                        self.readers += 1
                        self.max_readers = max(self.max_readers, self.readers)
                        try:
                            if self.readers > 1:
                                raise AssertionError("competing ASGI receive consumers")
                            return await super().receive()
                        finally:
                            self.readers -= 1

                    async def is_disconnected(self):
                        self.poll_calls += 1
                        return (await self.receive())["type"] == "http.disconnect"

                class StateEvent(asyncio.Event):
                    def __init__(self):
                        super().__init__()
                        self.waits = 0
                        self.second_wait = asyncio.Event()

                    async def wait(self):
                        self.waits += 1
                        if self.waits >= 2:
                            self.second_wait.set()
                        return await super().wait()

                request = OwnedRequest()
                event = StateEvent()
                state = SimpleNamespace(
                    event=event,
                    finished=False,
                    out_list=(
                        [{"text": "partial", "meta_info": {}}]
                        if has_partial_output
                        else []
                    ),
                )
                if has_partial_output:
                    event.set()
                obj = SimpleNamespace(
                    stream=False, background=False, rid="fixture-owner"
                )
                manager = object.__new__(TokenizerManager)
                manager.incremental_streaming_output = False
                aborted = []
                manager.abort_request = aborted.append
                iterator = manager._stream_one_response(obj, state, request)
                joined = []

                async def work():
                    try:
                        return await iterator.__anext__()
                    finally:
                        joined.append(True)

                with patch(
                    "sglang.srt.managers.tokenizer_manager._REQUEST_STATE_WAIT_TIMEOUT",
                    0.005,
                ):
                    task = asyncio.create_task(
                        await_response_or_disconnect(work(), request)
                    )
                    try:
                        await asyncio.wait_for(event.second_wait.wait(), 2)
                        await asyncio.wait_for(request.receiving.wait(), 2)
                        self.assertTrue(response_disconnect_watched(request))
                        await request.messages.put({"type": "http.disconnect"})
                        with self.assertRaisesRegex(
                            ValueError, "before the response was ready"
                        ):
                            await asyncio.wait_for(task, 2)
                    finally:
                        if not task.done():
                            task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        await iterator.aclose()
                self.assertEqual(request.poll_calls, 0)
                self.assertEqual(request.max_readers, 1)
                self.assertEqual(aborted, [])
                self.assertEqual(joined, [True])
                self.assertFalse(response_disconnect_watched(request))


if __name__ == "__main__":
    unittest.main()
