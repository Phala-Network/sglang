"""CPU lifecycle regressions for optional checkpoints under real slot pressure.

Tree, slot allocator, requests, donation and cleanup are production code. Only
device selection is overridden; pending backup transport is not exercised.
"""

import unittest
from array import array
from unittest.mock import patch

import torch
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.mem_cache_utils import finish_req

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMambaCheckpointPressure(unittest.TestCase):
    def fixture(self, extra_buffer=True, size=40):
        cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.MAMBA),
            enable_mamba_extra_buffer=extra_buffer,
            mamba_cache_size=size,
            page_size=64 if extra_buffer else 1,
            kv_size=4096,
            max_context_len=256,
            num_layers=2,
            full_attention_layer_ids=(0,),
        )
        with patch("sglang.test.kits.unified_radix_cache.fixtures.get_device", return_value="cpu"):
            return build_fixture(cfg)

    def request(self, cache, pool, index):
        req = Req(
            rid=str(index),
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        self.assertIsNotNone(pool.alloc([req]))
        req.last_node = cache.root_node_handle()
        req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
        return req

    def extend(self, req, allocator, pool, length):
        previous = len(req.prefix_indices)
        indices = allocator.alloc(length - previous)
        self.assertIsNotNone(indices)
        pool.write((req.kv.req_pool_idx, slice(previous, length)), indices)
        tokens = array("q", range(int(req.rid) * 1000, int(req.rid) * 1000 + length))
        req.origin_input_ids = tokens
        req.full_untruncated_fill_ids = tokens
        req.set_extend_range(previous, length)
        req.kv.kv_committed_len = length
        req.kv.kv_allocated_len = length
        req.kv.mamba_last_track_seqlen = length

    def finish(self, cache, pool, req, length):
        finish_req(cache, req, length)
        pool.free(req)

    def _cohort(self, pending_backup):
        cache, allocator, pool = self.fixture()
        requests = []
        for index in range(8):
            req = self.request(cache, pool, index)
            self.extend(req, allocator, pool, 64)
            cache.cache_unfinished_req(req, chunked=True)
            requests.append(req)
        # 24 live/request track slots + 8 current locked radix checkpoints.
        self.assertEqual(pool.mamba_allocator.available_size(), 8)
        pins = []
        for index in range(8, 16):
            key = RadixKey(array("q", range(index * 1000, index * 1000 + 64)))
            result = cache.insert(
                InsertParams(
                    key=key,
                    value=allocator.alloc(64),
                    mamba_value=pool.mamba_allocator.alloc(1),
                )
            )
            node = result.last_device_node
            receipt = cache.inc_lock_ref(node).to_dec_params()
            pins.append((node, receipt))
            if pending_backup:
                # Use real backup ownership/lock accounting; no synthetic CUDA
                # event or transport wait is needed to reproduce pinned slots.
                cache._track_write_through_node(node, receipt, [node])
        self.assertEqual(pool.mamba_allocator.available_size(), 0)
        self.assertEqual(cache.mamba_evictable_size(), 0)
        return cache, allocator, pool, requests, pins

    def _run_cohort(self, pending_backup):
        with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(False):
            cache, allocator, pool, requests, pins = self._cohort(pending_backup)
            state = pool.mamba_pool.mamba_cache
            state.temporal.fill_(7)
            for conv in state.conv:
                conv.fill_(11)
            temporal = state.temporal.clone()
            convs = [conv.clone() for conv in state.conv]
            for req in requests:
                node, receipt = req.last_node, req.lock_receipt
                working = req.kv.mamba_pool_idx.clone()
                buffers = req.kv.mamba_ping_pong_track_buffer.clone()
                self.extend(req, allocator, pool, 128)
                row = pool.req_to_token[req.kv.req_pool_idx, :128].clone()
                cache.cache_unfinished_req(req, chunked=True)
                self.assertEqual(req.last_node, node)
                self.assertEqual(req.lock_receipt, receipt)
                self.assertEqual(req.kv.cache_protected_len, 64)
                self.assertIsNone(req.kv.mamba_last_track_seqlen)
                torch.testing.assert_close(req.prefix_indices, row.to(torch.int64))
                torch.testing.assert_close(req.kv.mamba_pool_idx, working)
                torch.testing.assert_close(req.kv.mamba_ping_pong_track_buffer, buffers)
                torch.testing.assert_close(
                    pool.req_index_to_mamba_ping_pong_track_buffer_mapping[
                        req.kv.req_pool_idx
                    ],
                    buffers,
                )
                self.assertEqual(pool.mamba_allocator.available_size(), 0)
            torch.testing.assert_close(state.temporal, temporal)
            for actual, expected in zip(state.conv, convs):
                torch.testing.assert_close(actual, expected)
            self.assertEqual(cache.mamba_evictable_size(), 0)

            # Finish immediately after a declined checkpoint: untracked state
            # must not be inserted against a stale key, or leaked/double-freed.
            self.finish(cache, pool, requests.pop(), 128)
            for node, receipt in pins:
                if pending_backup:
                    cache._finish_write_through_ack(node)
                else:
                    cache.dec_lock_ref(node, receipt)
            # Remaining requests cache their next checkpoint when space returns.
            for req in requests:
                self.extend(req, allocator, pool, 192)
                cache.cache_unfinished_req(req, chunked=True)
                self.assertEqual(req.kv.cache_protected_len, 192)
                self.finish(cache, pool, req, 192)
            cache.evict(EvictParams(num_tokens=4096, mamba_num=40))
            self.assertEqual(pool.mamba_allocator.available_size(), 40)
            self.assertEqual(len(torch.unique(pool.mamba_allocator.free_slots)), 40)
            self.assertEqual(allocator.available_size(), 4096)
            self.assertFalse(cache.ongoing_write_through)

    def test_40_slots_8_requests_all_locked_without_pending_backups(self):
        self._run_cohort(False)

    def test_40_slots_8_requests_with_pending_write_through_backups(self):
        self._run_cohort(True)

    def test_empty_prefix_skip_and_normal_donation(self):
        for extra_buffer in (False, True):
            with self.subTest(extra_buffer=extra_buffer):
                cache, allocator, pool = self.fixture(extra_buffer)
                req = self.request(cache, pool, 0)
                self.extend(req, allocator, pool, 64)
                held = pool.mamba_allocator.alloc(pool.mamba_allocator.available_size())
                cache.cache_unfinished_req(req, chunked=True)
                self.assertEqual(req.kv.cache_protected_len, 0)
                self.assertEqual(len(req.prefix_indices), 64)
                pool.mamba_allocator.free(held)
                self.extend(req, allocator, pool, 128)
                cache.cache_unfinished_req(req, chunked=True)
                self.assertEqual(req.kv.cache_protected_len, 128)
                self.finish(cache, pool, req, 128)
                cache.evict(EvictParams(num_tokens=4096, mamba_num=40))
                self.assertEqual(pool.mamba_allocator.available_size(), 40)
                self.assertEqual(allocator.available_size(), 4096)

    def test_mandatory_allocation_still_asserts(self):
        cache, _, pool = self.fixture()
        pool.mamba_allocator.alloc(40)
        with self.assertRaisesRegex(AssertionError, "Can not alloc mamba cache"):
            cache.components[ComponentType.MAMBA]._alloc_mamba_slot()

    def test_finished_checkpoint_needs_no_replacement_slot(self):
        cache, allocator, pool = self.fixture()
        req = self.request(cache, pool, 0)
        self.extend(req, allocator, pool, 64)
        held = pool.mamba_allocator.alloc(pool.mamba_allocator.available_size())
        self.finish(cache, pool, req, 64)
        # Two request-owned slots freed; the tracked slot belongs to the tree.
        self.assertEqual(pool.mamba_allocator.available_size(), 2)
        self.assertEqual(cache.mamba_evictable_size(), 1)
        pool.mamba_allocator.free(held)
        cache.evict(EvictParams(num_tokens=4096, mamba_num=40))
        self.assertEqual(pool.mamba_allocator.available_size(), 40)


if __name__ == "__main__":
    unittest.main()
