"""Scheduler side of SGLANG_CHUNKED_PREFILL_SHORT_RESERVE: a continuing chunked
prefill leaves part of the chunk budget only if a waiting request may fit."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    DecLockRefResult,
    IncLockRefResult,
)
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

CHUNK = 8192
PAGE = 64


class _BatchBuilt(Exception):
    pass


def _make_req(rid, num_tokens, prefix_len=0):
    req = MagicMock(spec=Req)
    # A spec=Req mock gets a per-instance class; derive extend_len like Req.
    type(req).extend_len = Req.extend_len
    req.rid = rid
    req.cache_request_handle = CacheRequestHandle(rid, 0)
    req.priority = 0
    req.origin_input_ids = list(range(num_tokens))
    req.output_ids = []
    req.full_untruncated_fill_ids = req.origin_input_ids
    req.prefix_len = prefix_len
    req.sampling_params = SimpleNamespace(max_new_tokens=8, ignore_eos=False)
    req.time_stats = MagicMock()
    req.retracted_stain = False
    req.host_hit_length = 0
    req.swa_host_hit_length = 0
    req.storage_hit_length = 0
    req.storage_hit_start = None
    req.host_hit_is_storage = False
    req.host_loaded_length = 0
    req.materialized_host_hit_len.return_value = 0
    req.fulfilled_storage_hit_len.return_value = 0
    req.needs_host_load_back.return_value = False
    req.last_node = MagicMock()
    req.lora_id = None
    req.token_indices_to_pool = None
    req.beam_group = None
    req.inflight_middle_chunks = 0
    req.kv = SimpleNamespace(
        req_pool_idx=None,
        mamba_cow_src_index=None,
        mamba_needs_clear=False,
        holds_mamba=False,
        holds_kv=False,
    )
    req.extend_end = None
    return req


def _make_scheduler(chunked_req, waiting_queue):
    tree_cache = MagicMock()
    tree_cache.evictable_size.return_value = 0
    tree_cache.supports_mamba.return_value = False
    tree_cache.disable = False
    tree_cache.inc_lock_ref.return_value = IncLockRefResult()
    tree_cache.dec_lock_ref.return_value = DecLockRefResult()
    tree_cache.buffer_pipeline = None
    allocator = MagicMock()
    allocator.available_size.return_value = 1_000_000
    allocator.page_size = PAGE
    allocator.request_slot_reserve.return_value = 0
    allocator.create_prefill_budget.side_effect = lambda tree_cache, **kwargs: (
        PrefillBudget(allocator, tree_cache, **kwargs)
    )

    s = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    s.grammar_manager = MagicMock()
    s.grammar_manager.has_waiting_grammars.return_value = False
    s.enable_priority_preemption = False
    s.is_hybrid_swa = False
    s.chunked_req = chunked_req
    s.waiting_queue = list(waiting_queue)
    s.min_free_slots_delayer = None
    s.get_num_allocatable_reqs = lambda *args, **kwargs: 100
    s.policy = MagicMock()
    s.policy.shortest_prefill_chunk_limit.return_value = None
    s.processed_tokens_counter = 0
    s.chunked_prefill_size = CHUNK
    s.dynamic_chunk_sizer = None
    s.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
    )
    s.page_size = PAGE
    s.tree_cache = tree_cache
    s.token_to_kv_pool_allocator = allocator
    s.new_token_ratio_tracker = SimpleNamespace(current=1.0)
    s.max_prefill_tokens = 2 * CHUNK
    s.is_mixed_chunk = False
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs = 0
    s.governor = None
    s.max_running_requests = 100
    s.dllm_config = None
    s.enable_lora = False
    s.req_to_token_pool = SimpleNamespace()
    s.disaggregation_mode = DisaggregationMode.NULL
    s.enable_hicache_storage = False
    s.enable_hierarchical_cache = False
    s.enable_unified_cache_external_linker = False
    s.truncation_align_size = None
    s.model_config = None
    s.enable_overlap = False
    s.spec_algorithm = None
    s.enable_lmcache = False
    return s


class TestChunkedPrefillShortReserve(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _prefill(self, waiting_queue, reserve=None):
        """Run one prefill pass with a long prompt one chunk in; return the
        scheduler and the (rid, extend length) of each request in the batch."""
        long_prompt = _make_req("long-prompt", 5 * CHUNK, prefix_len=CHUNK)
        s = _make_scheduler(long_prompt, waiting_queue)
        running_batch = MagicMock()
        running_batch.reqs = []
        running_batch.batch_is_full = False
        built = []

        def init_new(can_run_list, *args, **kwargs):
            built.extend((r.rid, r.extend_len) for r in can_run_list)
            raise _BatchBuilt

        with (
            (
                nullcontext()
                if reserve is None
                else envs.SGLANG_CHUNKED_PREFILL_SHORT_RESERVE.override(reserve)
            ),
            patch.object(
                scheduler_module.ScheduleBatch, "init_new", side_effect=init_new
            ),
            self.assertRaises(_BatchBuilt),
        ):
            s._get_new_batch_prefill_raw(None, running_batch)
        self.assertIs(s.chunked_req, long_prompt)
        return s, built

    def test_reserve_unset_keeps_whole_chunk(self):
        short = _make_req("short", 1000)
        s, built = self._prefill([short])
        self.assertEqual(built, [("long-prompt", CHUNK)])
        self.assertEqual(s.waiting_queue, [short])

    def test_only_long_waiting_reqs_keep_whole_chunk(self):
        waiting = [_make_req("long-1", 5000), _make_req("long-2", 3000)]
        s, built = self._prefill(waiting, reserve=2048)
        self.assertEqual(built, [("long-prompt", CHUNK)])
        self.assertEqual(s.waiting_queue, waiting)

    def test_short_waiting_req_joins_with_reserve(self):
        short, longer = _make_req("short", 1000), _make_req("longer", 5000)
        s, built = self._prefill([short, longer], reserve=2048)
        self.assertEqual(built, [("long-prompt", CHUNK - 2048), ("short", 1000)])
        self.assertEqual(s.waiting_queue, [longer])

    def test_short_behind_long_head_keeps_whole_chunk(self):
        longer, short = _make_req("longer", 5000), _make_req("short", 1000)
        s, built = self._prefill([longer, short], reserve=2048)
        self.assertEqual(built, [("long-prompt", CHUNK)])
        self.assertEqual(s.waiting_queue, [longer, short])

    def test_cached_prefix_counts_toward_fit(self):
        cached = _make_req("cached", 6000, prefix_len=5000)
        s, built = self._prefill([cached], reserve=2048)
        self.assertEqual(built, [("long-prompt", CHUNK - 2048), ("cached", 1000)])
        self.assertEqual(s.waiting_queue, [])


if __name__ == "__main__":
    unittest.main()
