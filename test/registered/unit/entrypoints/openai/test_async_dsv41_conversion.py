"""CPU-only routing, cancellation and serialization checks for async V4.1 chat."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import asyncio
import contextvars
import threading
import time
import unittest
from contextlib import suppress
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.async_dynamic_batch_tokenizer import AsyncDynamicbatchTokenizer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class FakeTokenizer:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.thread_ids = set()

    def __call__(self, prompts, **kwargs):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.thread_ids.add(threading.get_ident())
        try:
            time.sleep(0.015)
            encode = lambda text: [ord(char) for char in text]
            return {
                "input_ids": [encode(text) for text in prompts]
                if isinstance(prompts, list)
                else encode(prompts)
            }
        finally:
            with self.lock:
                self.active -= 1


class AsyncConversionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tokenizer = FakeTokenizer()
        self.batcher = AsyncDynamicbatchTokenizer(self.tokenizer)

    async def asyncTearDown(self):
        task = self.batcher._batcher_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self.batcher._executor.shutdown(wait=True, cancel_futures=True)

    def chat(self, spec="dsv41", enabled=True):
        chat = object.__new__(OpenAIServingChat)
        chat.chat_encoding_spec = spec
        chat.tokenizer_manager = SimpleNamespace(
            async_dynamic_batch_tokenizer=self.batcher if enabled else None,
            request_logger=SimpleNamespace(log_requests=False),
        )
        return chat

    async def test_opt_in_uses_worker_and_preserves_request_context(self):
        context = contextvars.ContextVar("test_request_context", default="missing")
        context.set("request-17")
        main_thread = threading.get_ident()
        request, raw = SimpleNamespace(input_ids=None), object()
        chat = self.chat()
        seen = []

        def convert(req, raw_req):
            seen.append((req, raw_req, threading.get_ident(), context.get()))
            return {"ids": [3, 4, 5]}, req

        chat._convert_to_internal_request = convert
        result = await chat._convert_to_internal_request_async(request, raw)
        self.assertEqual(result, ({"ids": [3, 4, 5]}, request))
        self.assertIs(seen[0][0], request)
        self.assertIs(seen[0][1], raw)
        self.assertNotEqual(seen[0][2], main_thread)
        self.assertEqual(seen[0][3], "request-17")

    async def test_other_encoders_and_disabled_or_pretokenized_inputs_stay_inline(self):
        main_thread = threading.get_ident()
        for spec, enabled, ids in [
            (None, True, None),
            ("dsv4", True, None),
            ("dsv41", False, None),
            ("dsv41", True, [8, 9]),
        ]:
            with self.subTest(spec=spec, enabled=enabled, ids=ids):
                chat = self.chat(spec, enabled)
                chat._convert_to_internal_request = lambda req, raw: (
                    threading.get_ident(),
                    req,
                )
                request = SimpleNamespace(input_ids=ids)
                result = await chat._convert_to_internal_request_async(request, None)
                self.assertEqual(result, (main_thread, request))

    async def test_encode_and_conversion_share_one_serialized_worker(self):
        results = await asyncio.gather(
            self.batcher.encode("alpha"),
            self.batcher.run_sync(self.tokenizer, "beta"),
            self.batcher.encode("gamma"),
            self.batcher.run_sync(self.tokenizer, "delta"),
        )
        self.assertEqual(
            [result["input_ids"] for result in results],
            [
                [ord(char) for char in text]
                for text in ["alpha", "beta", "gamma", "delta"]
            ],
        )
        self.assertEqual(self.tokenizer.peak, 1)
        self.assertEqual(len(self.tokenizer.thread_ids), 1)

    async def test_event_loop_remains_responsive_during_conversion(self):
        chat = self.chat()
        started, release = threading.Event(), threading.Event()

        def convert(req, raw):
            started.set()
            self.assertTrue(release.wait(2))
            return "converted", req

        chat._convert_to_internal_request = convert
        pending = asyncio.create_task(
            chat._convert_to_internal_request_async(
                SimpleNamespace(input_ids=None), None
            )
        )
        try:
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.005)
            self.assertTrue(started.is_set())
            self.assertFalse(pending.done())
            await asyncio.sleep(0.01)
            self.assertFalse(pending.done())
        finally:
            release.set()
            await pending

    async def test_queued_cancellation_does_not_execute_conversion(self):
        started, release = threading.Event(), threading.Event()
        calls = []

        def blocking():
            started.set()
            release.wait(2)

        first = asyncio.create_task(self.batcher.run_sync(blocking))
        try:
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.005)
            self.assertTrue(started.is_set())
            second = asyncio.create_task(
                self.batcher.run_sync(calls.append, "must-not-run")
            )
            await asyncio.sleep(0.01)
            second.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await second
        finally:
            release.set()
            await first
        await self.batcher.run_sync(lambda: None)
        self.assertEqual(calls, [])

    async def test_handler_cancellation_does_not_submit_inference(self):
        chat = self.chat()
        started, release = threading.Event(), threading.Event()
        submitted = []
        chat._validate_request = lambda request: None

        def convert(req, raw):
            started.set()
            release.wait(2)
            return SimpleNamespace(), req

        async def stream(*args):
            submitted.append(True)

        chat._convert_to_internal_request = convert
        chat._handle_streaming_request = stream
        task = asyncio.create_task(
            chat.handle_request(SimpleNamespace(input_ids=None, stream=True), None)
        )
        try:
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.005)
            self.assertTrue(started.is_set())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            release.set()
        await self.batcher.run_sync(lambda: None)
        self.assertEqual(submitted, [])

    async def test_handler_keeps_value_error_status(self):
        chat = self.chat()
        chat._validate_request = lambda request: None
        chat.create_error_response = lambda **kwargs: kwargs

        def invalid(req, raw):
            raise ValueError("invalid-fixture")

        chat._convert_to_internal_request = invalid
        result = await chat.handle_request(
            SimpleNamespace(input_ids=None, stream=True), None
        )
        self.assertEqual(result["status_code"], 400)
        self.assertEqual(result["message"], "invalid-fixture")

    async def test_base_default_keeps_synchronous_conversion(self):
        main_thread = threading.get_ident()
        target = SimpleNamespace(
            _convert_to_internal_request=lambda req, raw: (threading.get_ident(), req)
        )
        request = object()
        result = await OpenAIServingBase._convert_to_internal_request_async(
            target, request, None
        )
        self.assertEqual(result, (main_thread, request))


if __name__ == "__main__":
    unittest.main()
