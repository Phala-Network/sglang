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
