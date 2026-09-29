"""CPU smoke for the installed cold shared-read bypass path."""

import dataclasses
import queue
import types
import unittest
from unittest.mock import patch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.cold_shared_read import ColdSharedReadTrace
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


class _FakeStore:
    def __init__(self):
        self.calls = 0

    def batch_get_into(self, keys, pointers, sizes):
        self.calls += 1
        return [-1] * len(keys)


class ColdSharedReadInstalledTest(unittest.TestCase):
    def test_request_flag_single_and_batch(self):
        single = GenerateReqInput(text="cold", cold_shared_read_bypass=True)
        single.normalize_batch_and_arguments()
        self.assertTrue(single.cold_shared_read_bypass)

        ordinary = GenerateReqInput(text="ordinary")
        ordinary.normalize_batch_and_arguments()
        self.assertFalse(ordinary.cold_shared_read_bypass)

        batch = GenerateReqInput(
            text=["one", "two"],
            extra_key=["one", "two"],
            cold_shared_read_bypass=True,
        )
        batch.normalize_batch_and_arguments()
        self.assertTrue(batch[0].cold_shared_read_bypass)
        self.assertTrue(batch[1].cold_shared_read_bypass)

        invalid = GenerateReqInput(text="bad", cold_shared_read_bypass="true")
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            invalid.normalize_batch_and_arguments()

    def test_tokenizer_ipc_flag_propagation(self):
        class FakeSamplingParams:
            def __init__(self, **kwargs):
                pass

            def normalize(self, tokenizer):
                pass

            def verify(self, vocab_size):
                pass

        state = types.SimpleNamespace(
            time_stats=types.SimpleNamespace(set_tokenize_finish_time=lambda: None)
        )
        manager = types.SimpleNamespace(
            preferred_sampling_params=None,
            sampling_params_class=FakeSamplingParams,
            tokenizer=None,
            model_config=types.SimpleNamespace(vocab_size=128),
            rid_to_state={"request": state},
        )
        request = GenerateReqInput(
            rid="request",
            input_ids=[1, 2],
            bootstrap_room=1,
            cold_shared_read_bypass=True,
        )
        request.normalize_batch_and_arguments()
        tokenized = TokenizerManager._create_tokenized_object(
            manager, request, None, [1, 2]
        )
        self.assertTrue(tokenized.cold_shared_read_bypass)

    def test_scheduler_and_radix_guards(self):
        trace = ColdSharedReadTrace("rid", "fresh", None, 0)
        handle = CacheRequestHandle("rid", 0, cold_shared_read_trace=trace)
        retry = types.SimpleNamespace(cancelled=[], cancel=lambda rid: retry.cancelled.append(rid))
        cache = types.SimpleNamespace(
            enable_storage=True,
            cache_controller=object(),
            storage_prefetch_retries=retry,
        )
        UnifiedRadixCache.prefetch_from_storage(cache, handle, None, [1, 2])
        self.assertEqual(retry.cancelled, ["rid"])

        calls = []
        request = types.SimpleNamespace(
            cold_shared_read_bypass=True,
            init_next_round_input=lambda tree, cow_mamba: calls.append("local_match"),
        )
        scheduler = types.SimpleNamespace(enable_hicache_storage=True, tree_cache=object())
        Scheduler._prefetch_kvcache(scheduler, request)
        self.assertEqual(calls, ["local_match"])
        trace.terminal("finished")

    def test_async_link_and_actual_mooncake_get_miss(self):
        trace = ColdSharedReadTrace("rid", "fresh", None, 0)
        handle = CacheRequestHandle("rid", 0, cold_shared_read_trace=trace)
        controller = types.SimpleNamespace(prefetch_queue=queue.Queue())
        operation = HybridCacheController.prefetch(controller, handle, [1, 2])
        self.assertIs(operation.cold_shared_read_trace, trace)
        self.assertIs(controller.prefetch_queue.get_nowait(), operation)

        backend = object.__new__(MooncakeStore)
        backend.store = _FakeStore()
        with patch.object(MooncakeStore, "_uses_multi_buffer", return_value=False):
            self.assertEqual(
                backend._get_batch_zero_copy_impl(["miss"], [0], [8], trace),
                [-1],
            )
        self.assertEqual(backend.store.calls, 1)
        trace.terminal("abort")
        self.assertFalse(trace._ended)
        trace.operation_end()
        self.assertTrue(trace._ended)
        self.assertEqual(trace._get_calls, 1)
        self.assertEqual(trace._get_keys, 1)
        self.assertEqual(trace._returned_bytes, 0)

        with patch.object(MooncakeStore, "_uses_multi_buffer", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "after terminal summary"):
                backend._get_batch_zero_copy_impl(["late"], [0], [8], trace)
        self.assertEqual(backend.store.calls, 1)
        advanced = dataclasses.replace(handle, attempt_id=1)
        self.assertIs(advanced.cold_shared_read_trace, trace)


if __name__ == "__main__":
    unittest.main()
