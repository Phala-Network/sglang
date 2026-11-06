"""CPU source-method regression for the K3.1 admission candidate.

This intentionally avoids importing the CUDA-serving dependency graph. It compiles
the actual, unedited AST methods/classes from the selected source tree, and supplies
explicit bounded allocator/cache collaborators. No policy method is reimplemented.
It does not validate CUDA, tensor copies, a real radix tree or wall-clock GPU TPS.
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import enum
import hashlib
import json
import logging
import statistics
import sys
import time
import types
import unittest
from array import array
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

PATHS = {
    "scheduler": "python/sglang/srt/managers/scheduler.py",
    "policy": "python/sglang/srt/managers/schedule_policy.py",
    "batch": "python/sglang/srt/managers/schedule_batch.py",
    "prefix": "python/sglang/srt/mem_cache/base_prefix_cache.py",
    "mamba": "python/sglang/srt/mem_cache/unified_cache/components/mamba.py",
    "budget": "python/sglang/srt/mem_cache/prefill_budget.py",
}


def source_namespace(root: Path, label: str):
    """Keep all executed policy bodies verbatim; omit only unused imports/bases."""
    module = types.ModuleType(f"admission_source_{label}")
    sys.modules[module.__name__] = module
    scope = module.__dict__
    scope.update(
        dataclasses=dataclasses,
        dataclass=dataclasses.dataclass,
        Enum=enum.Enum,
        auto=enum.auto,
        contextmanager=contextmanager,
        nullcontext=nullcontext,
        CLIP_MAX_NEW_TOKENS=4096,
        TEST_RETRACT=False,
        TEST_RETRACT_NO_PREFILL_BS=0,
        _IS_HIP=False,
        PREFILL_TILE_BUDGET=0,
        PREFILL_TILE_BUDGET_MODE="compact",
        logger=logging.getLogger(label),
        get_schedule=lambda: NS(prefill_max_requests=None, schedule_policy="hrrn"),
        get_exec=lambda: NS(features=NS(enable_encoder_swa_bounded_replay=False)),
        _use_exact_chunk_fill=lambda: False,
        mamba_prefill_chunk_grid=lambda: 1,
        page_interleave_shard_size=lambda allocator: 1,
        set_time_batch=lambda *args, **kwargs: None,
        PrefillStats=NS(from_adder=lambda *args, **kwargs: NS()),
        ScheduleBatch=NS(init_new=lambda reqs, *args, **kwargs: FixtureBatch(reqs)),
    )
    for name in (
        "SWATokenToKVPoolAllocator",
        "DeepSeekV4HiSparseTokenToKVPoolAllocator",
        "UnifiedMambaTokenToKVPoolAllocator",
        "UnifiedMambaSWATokenToKVPoolAllocator",
        "UnifiedRadixCache",
    ):
        scope[name] = type(name, (), {})

    def load(key, names, methods=None):
        path = root / PATHS[key]
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        chosen = []
        for node in tree.body:
            if getattr(node, "name", None) not in names:
                continue
            node = copy.deepcopy(node)
            if methods is not None:
                node.bases = []
                node.keywords = []
                node.decorator_list = []
                node.body = [
                    n for n in node.body if getattr(n, "name", None) in methods
                ]
            chosen.append(node)
        if {n.name for n in chosen} != set(names):
            raise AssertionError(f"Source API changed: {path}: {names}")
        body = [
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            )
        ]
        body.extend(chosen)
        fragment = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        exec(compile(fragment, str(path), "exec"), scope)

    load("policy", {"AddReqResult", "_PrefillAdmission", "PrefillAdder"})
    load(
        "policy",
        {"SchedulePolicy"},
        {"_uncached_len", "_sort_by_hrrn", "_compute_prefix_matches"},
    )
    load("prefix", {"EvictParams", "IncLockRefResult", "DecLockRefParams"})
    prefix_tree = ast.parse((root / PATHS["prefix"]).read_text(encoding="utf-8-sig"))
    if any(getattr(n, "name", None) == "MambaCowAllocError" for n in prefix_tree.body):
        load("prefix", {"MambaCowAllocError"})
    load("batch", {"ReqKvInfo"})
    load("batch", {"Req"}, {"seqlen", "refresh_fill_ids"})
    load("budget", {"PrefillBudget"})
    load(
        "scheduler",
        {"Scheduler"},
        {"_get_new_batch_prefill_raw", "_revert_unadmitted_mamba"},
    )
    load("mamba", {"MambaComponent"}, {"finalize_match_result_in_cache"})
    scope["hashes"] = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in PATHS.values()
    }
    return NS(**scope)


class Slot(int):
    def unsqueeze(self, axis):
        return [self]


class SlotAllocator:
    """Finite logical slots with strict double-free and alloc-group checks."""

    def __init__(self, free_slots=1):
        self.free_slots = [Slot(i + 1) for i in range(free_slots)]
        self.held = set()
        self.alloc_calls = 0
        self.free_calls = 0
        self.group_depth = 0
        self.group_begins = 0
        self.group_ends = 0
        self.next_slot = free_slots + 1

    def alloc(self, n):
        assert n == 1
        self.alloc_calls += 1
        if not self.free_slots:
            return None
        slot = self.free_slots.pop()
        assert slot not in self.held
        self.held.add(slot)
        return [slot]

    def free(self, slots):
        self.free_calls += 1
        for slot in slots:
            assert slot in self.held, f"Double/free of unowned slot {slot}"
            self.held.remove(slot)
            self.free_slots.append(slot)

    def recover_one(self):
        self.free_slots.append(Slot(self.next_slot))
        self.next_slot += 1

    def schedulable_available_size(self):
        return len(self.free_slots)

    def alloc_group_begin(self, count):
        assert self.group_depth == 0
        self.group_depth = 1
        self.group_begins += 1

    def alloc_group_end(self):
        assert self.group_depth == 1
        self.group_depth = 0
        self.group_ends += 1


class TokenAllocator:
    def __init__(self, api, free_tokens):
        self.api = api
        self.free_tokens = free_tokens
        self.page_size = 1

    def create_prefill_budget(self, cache, **kwargs):
        return self.api.PrefillBudget(self, cache, **kwargs)

    def available_size(self):
        return self.free_tokens

    def request_slot_reserve(self, **kwargs):
        return 0


class FixtureCache:
    """Fixed prefix hit, explicit capacity recovery and exact lock ownership."""

    def __init__(self, api, *, mamba=True, free_slots=1, prefix_hit=True):
        self.api = api
        self.req_to_token_pool = NS(mamba_allocator=SlotAllocator(free_slots))
        self.mamba = mamba
        self.disable = False
        self.buffer_pipeline = None
        self.storage_prefetch_retries = None
        self.lock_depth = 0
        self.lock_calls = 0
        self.unlock_calls = 0
        self.evict_calls = 0
        self.match_calls = 0
        self.evict_recovers_one = False
        self.prefix_hit = prefix_hit
        self.node = NS(id=7)
        self.component = api.MambaComponent.__new__(api.MambaComponent)
        self.component.cache = self
        self.component.component_type = "mamba"
        self.component.tree_core = NS(
            get_component_device_value=lambda node, kind: (
                [Slot(999)] if self.prefix_hit else None
            )
        )

    def supports_mamba(self):
        return self.mamba

    def supports_prefix_sharing(self):
        return True

    def full_evictable_size(self):
        return 0

    def evictable_size(self):
        return 0

    def mamba_evictable_size(self):
        return int(self.evict_recovers_one)

    def inc_lock_ref(self, node):
        self.lock_depth += 1
        self.lock_calls += 1
        return self.api.IncLockRefResult()

    def dec_lock_ref(self, node, receipt=None):
        assert self.lock_depth > 0
        self.lock_depth -= 1
        self.unlock_calls += 1

    def lock(self, node):
        self.inc_lock_ref(node)
        return node

    def unlock(self, lock):
        self.dec_lock_ref(lock)

    def evict_for_alloc(self, params):
        assert self.lock_depth >= 1
        assert params.mamba_num == 1
        self.evict_calls += 1
        if self.evict_recovers_one:
            self.req_to_token_pool.mamba_allocator.recover_one()
            self.evict_recovers_one = False

    def finish_storage_prefetch_admission(self, *args, **kwargs):
        return None


class FixtureReq:
    def __init__(self, api, cache):
        self.api = api
        self.rid = "synthetic-admission"
        self.cache_request_handle = NS(rid=self.rid, attempt_id=0)
        self.beam_group = None
        self.session = None
        self.lora_id = None
        self.token_indices_to_pool = None
        self.kv = api.ReqKvInfo()
        self.output_ids = []
        self.origin_input_ids = list(range(119))
        self.full_untruncated_fill_ids = list(range(119))
        self.sampling_params = NS(ignore_eos=False, max_new_tokens=128)
        self.prefix_len = 0
        self.host_hit_length = 0
        self.swa_host_hit_length = 0
        self.auxiliary_swa_host_hit_length = 0
        self.mamba_host_hit_length = 0
        self.host_loaded_length = 0
        self.storage_hit_length = 0
        self.storage_hit_start = 0
        self.host_hit_is_storage = False
        self.retracted_stain = False
        self.num_matched_prefix_tokens = 0
        self.arrival_processed_tokens = 0
        self.match_calls = 0
        self.last_node = cache.node
        self.lock = None

    def init_next_round_input(self, cache):
        self.match_calls += 1
        cache.match_calls += 1
        if cache.supports_mamba():
            cache.component.finalize_match_result_in_cache(
                NS(cow_mamba=True, req=self), NS(best_match_node=cache.node)
            )

    def needs_host_load_back(self):
        return False

    def fulfilled_storage_hit_len(self, prefix_len):
        return 0

    @property
    def seqlen(self):
        # Exercise the real Req contract, rather than a mirrored span formula.
        return self.api.Req.seqlen.fget(self)


class FixtureBatch:
    def __init__(self, reqs, *, full=False):
        self.reqs = list(reqs)
        self.batch_is_full = full
        self.return_logprob = False
        self.input_embeds = None

    def is_empty(self):
        return not self.reqs

    def batch_size(self):
        return len(self.reqs)

    def prepare_for_extend(self):
        return None


def fixture(api, *, hybrid=True, mamba=True, free_tokens=100, free_slots=1):
    cache = FixtureCache(api, mamba=mamba, free_slots=free_slots)
    req = FixtureReq(api, cache)
    req.api = api
    s = api.Scheduler.__new__(api.Scheduler)
    s.grammar_manager = NS(has_waiting_grammars=lambda: False)
    s.enable_priority_preemption = False
    s.is_hybrid_swa = False
    s.is_hybrid_ssm = hybrid
    s.tree_cache = cache
    s.waiting_queue = [req]
    s.chunked_req = None
    s.min_free_slots_delayer = None
    s.get_num_allocatable_reqs = lambda *args, **kwargs: 8
    s.disagg_prefill_bootstrap_queue = None
    s.priority_calls = 0

    def calc_priority(*args, **kwargs):
        s.priority_calls += 1

    s.policy = NS(calc_priority=calc_priority)
    s.processed_tokens_counter = 0
    s.chunked_prefill_size = 4096
    s.dynamic_chunk_sizer = None
    s.pp_min_chunked_prefill_size = None
    s.tp_worker = NS(model_runner=NS(attn_backend=NS(), prefill_aware_swa=False))
    s.get_prefill_adder_class = lambda: api.PrefillAdder
    s.page_size = 1
    s.token_to_kv_pool_allocator = TokenAllocator(api, free_tokens)
    s.new_token_ratio_tracker = NS(current=1.0)
    s.max_prefill_tokens = 4096
    s.is_mixed_chunk = False
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs = 8
    s.max_running_requests = 8
    s.dllm_config = None
    s.enable_lora = False
    s.enable_hicache_storage = False
    s.enable_lmcache = False
    s.enable_hierarchical_cache = False
    s.enable_unified_cache_external_linker = False
    s.req_to_token_pool = cache.req_to_token_pool
    s.truncation_align_size = None
    s.model_config = None
    s.enable_overlap = False
    s.spec_algorithm = None
    s.hisparse_coordinator = None
    s.enable_priority_scheduling = False
    s.load_inquirer = NS(_get_num_pending_tokens=lambda **kwargs: 0)
    running_req = NS(
        sampling_params=NS(ignore_eos=False, max_new_tokens=0), output_ids=[]
    )
    running = FixtureBatch([running_req])
    return s, req, running


def step(s, running):
    return s._get_new_batch_prefill_raw(
        prefill_delayer_single_pass=None, running_batch=running
    )[0]


def release_fixture(s, req):
    if req.lock is not None:
        s.tree_cache.unlock(req.lock)
        req.lock = None
    s._revert_unadmitted_mamba(req)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        # When pytest imports this registered test, use real engine imports.
        # CLI --source-root is a separate, explicitly labeled source-method gate.
        self.import_patches = ExitStack()
        self.addCleanup(self.import_patches.close)
        if REAL_IMPORTS:
            for target, value in (
                (
                    "sglang.srt.managers.scheduler.get_schedule",
                    lambda: NS(prefill_max_requests=None),
                ),
                (
                    "sglang.srt.managers.scheduler.ScheduleBatch",
                    NS(init_new=lambda reqs, *args, **kwargs: FixtureBatch(reqs)),
                ),
                (
                    "sglang.srt.managers.scheduler.PrefillStats",
                    NS(from_adder=lambda *args, **kwargs: NS()),
                ),
                (
                    "sglang.srt.managers.scheduler.set_time_batch",
                    lambda *args, **kwargs: None,
                ),
                ("sglang.srt.managers.scheduler.TEST_RETRACT", False),
                (
                    "sglang.srt.managers.schedule_policy.mamba_prefill_chunk_grid",
                    lambda: 1,
                ),
                (
                    "sglang.srt.managers.schedule_policy._use_exact_chunk_fill",
                    lambda: False,
                ),
                (
                    "sglang.srt.managers.schedule_policy.page_interleave_shard_size",
                    lambda allocator: 1,
                ),
                ("sglang.srt.managers.schedule_policy._IS_HIP", False),
                (
                    "sglang.srt.managers.schedule_policy.get_exec",
                    lambda: NS(features=NS(enable_encoder_swa_bounded_replay=False)),
                ),
            ):
                self.import_patches.enter_context(patch(target, value))

    def test_baseline_reproduces_no_token_latch_after_kv_recovery(self):
        if BASELINE is None:
            self.skipTest(
                "Before-source comparison belongs to the explicit source-method gate"
            )
        s, req, running = fixture(BASELINE)
        self.assertIsNone(step(s, running))
        self.assertTrue(running.batch_is_full)
        s.token_to_kv_pool_allocator.free_tokens = 754000
        for _ in range(3):
            self.assertIsNone(step(s, running))
        self.assertEqual(s.tree_cache.match_calls, 1)
        self.assertEqual(s.waiting_queue, [req])
        self.assertEqual(s.tree_cache.lock_depth, 0)
        self.assertFalse(req.kv.holds_mamba)

    def test_candidate_retries_and_admits_after_kv_recovery(self):
        s, req, running = fixture(CANDIDATE)
        self.assertIsNone(step(s, running))
        self.assertTrue(running.batch_is_full)
        s.token_to_kv_pool_allocator.free_tokens = 754000
        batch = step(s, running)
        self.assertIsNotNone(batch, "Recovered KV capacity must reopen SSM admission")
        self.assertEqual(batch.reqs, [req])
        self.assertEqual(s.waiting_queue, [])
        self.assertEqual(s.tree_cache.match_calls, 2)
        self.assertTrue(req.kv.holds_mamba)
        self.assertIsNotNone(req.kv.mamba_cow_src_index)
        self.assertEqual(s.tree_cache.lock_depth, 1)
        release_fixture(s, req)
        self.assertEqual(s.tree_cache.lock_depth, 0)
        self.assertFalse(req.kv.holds_mamba)
        self.assertEqual(s.req_to_token_pool.mamba_allocator.held, set())

    def test_repeated_no_token_cleans_cow_and_balances_groups_and_locks(self):
        s, req, running = fixture(CANDIDATE)
        for _ in range(100):
            self.assertIsNone(step(s, running))
            self.assertEqual(s.waiting_queue, [req])
            self.assertFalse(req.kv.holds_mamba)
            self.assertIsNone(req.kv.mamba_cow_src_index)
            self.assertFalse(req.kv.mamba_needs_clear)
            self.assertEqual(s.tree_cache.lock_depth, 0)
        alloc = s.req_to_token_pool.mamba_allocator
        self.assertEqual(alloc.alloc_calls, 100)
        self.assertEqual(alloc.free_calls, 100)
        self.assertEqual(alloc.group_begins, alloc.group_ends)
        self.assertEqual(alloc.group_depth, 0)
        self.assertEqual(len(alloc.free_slots), 1)

    def test_baseline_cow_exhaustion_is_assertion(self):
        if BASELINE is None:
            self.skipTest(
                "Before-source comparison belongs to the explicit source-method gate"
            )
        s, req, _ = fixture(BASELINE, free_slots=0)
        with self.assertRaisesRegex(AssertionError, "Can not alloc mamba cache"):
            req.init_next_round_input(s.tree_cache)
        self.assertEqual(s.tree_cache.lock_depth, 0)
        self.assertFalse(req.kv.holds_mamba)

    def test_candidate_cow_exhaustion_queues_then_recovers(self):
        s, req, running = fixture(CANDIDATE, free_tokens=754000, free_slots=0)
        for _ in range(100):
            self.assertIsNone(step(s, running))
            self.assertEqual(s.waiting_queue, [req])
            self.assertEqual(s.tree_cache.lock_depth, 0)
            self.assertFalse(req.kv.holds_mamba)
            self.assertIsNone(req.kv.mamba_cow_src_index)
        alloc = s.req_to_token_pool.mamba_allocator
        self.assertEqual(alloc.group_depth, 0)
        self.assertEqual(alloc.group_begins, alloc.group_ends)
        self.assertEqual(s.tree_cache.evict_calls, 100)
        alloc.recover_one()
        self.assertEqual(step(s, running).reqs, [req])
        release_fixture(s, req)
        self.assertEqual(s.tree_cache.lock_depth, 0)
        self.assertEqual(len(alloc.free_slots), 1)

    def test_cow_eviction_success_balances_temporary_pin_and_cleanup(self):
        s, req, _ = fixture(CANDIDATE, free_slots=0)
        s.tree_cache.evict_recovers_one = True
        req.init_next_round_input(s.tree_cache)
        self.assertTrue(req.kv.holds_mamba)
        self.assertEqual(s.tree_cache.evict_calls, 1)
        self.assertEqual(s.tree_cache.lock_depth, 0)
        self.assertEqual(s.tree_cache.lock_calls, s.tree_cache.unlock_calls)
        release_fixture(s, req)
        alloc = s.req_to_token_pool.mamba_allocator
        self.assertEqual(len(alloc.free_slots), 1)
        self.assertEqual(alloc.held, set())

    def test_non_ssm_retains_latch_and_success_behavior(self):
        for api in (BASELINE, CANDIDATE) if BASELINE else (CANDIDATE,):
            s, req, running = fixture(api, hybrid=False, mamba=False)
            self.assertIsNone(step(s, running))
            s.token_to_kv_pool_allocator.free_tokens = 754000
            self.assertIsNone(step(s, running))
            self.assertEqual(s.tree_cache.match_calls, 1)
            running.batch_is_full = False
            self.assertEqual(step(s, running).reqs, [req])
            self.assertFalse(req.kv.holds_mamba)
            release_fixture(s, req)
            self.assertEqual(s.tree_cache.lock_depth, 0)

    def test_latch_configuration_gate_keeps_existing_empty_swa_priority_paths(self):
        for hybrid, mamba, swa, priority, empty, expected in (
            (True, True, False, False, False, False),
            (True, False, False, False, False, True),
            (False, True, False, False, False, True),
            (False, False, True, False, False, False),
            (False, False, False, True, False, False),
            (False, False, False, False, True, False),
        ):
            with self.subTest(
                hybrid=hybrid, mamba=mamba, swa=swa, priority=priority, empty=empty
            ):
                s, _, running = fixture(CANDIDATE, hybrid=hybrid, mamba=mamba)
                s.waiting_queue = []
                s.is_hybrid_swa = swa
                s.enable_priority_preemption = priority
                running.batch_is_full = True
                if empty:
                    running.reqs = []
                self.assertIsNone(step(s, running))
                self.assertEqual(running.batch_is_full, expected)

    def test_session_owned_slot_is_not_freed_on_refusal(self):
        s, req, running = fixture(CANDIDATE)
        req.session = NS(streaming=True)
        slot = s.req_to_token_pool.mamba_allocator.alloc(1)[0]
        req.kv.mamba_pool_idx = slot
        for _ in range(3):
            self.assertIsNone(step(s, running))
            self.assertEqual(req.kv.mamba_pool_idx, slot)
            self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertEqual(s.req_to_token_pool.mamba_allocator.free_calls, 0)
        s.req_to_token_pool.mamba_allocator.free([slot])

    def test_unrelated_runtime_error_is_not_silenced(self):
        s, req, running = fixture(CANDIDATE)

        def unrelated(cache):
            raise RuntimeError("synthetic non-capacity bug")

        req.init_next_round_input = unrelated
        with self.assertRaisesRegex(RuntimeError, "non-capacity bug"):
            step(s, running)

    def test_855058_replay_still_fails_756736_budget_before_chunk8192(self):
        s, replay, running = fixture(CANDIDATE, free_tokens=754048)
        replay.full_untruncated_fill_ids = range(855058)
        replay.origin_input_ids = range(118)
        replay.output_ids = range(854940)
        replay.sampling_params.max_new_tokens = 1048576
        replay.retracted_stain = True
        s.tree_cache.full_evictable_size = lambda: 2688
        s.chunked_prefill_size = 8192
        fresh = FixtureReq(CANDIDATE, s.tree_cache)
        s.waiting_queue.append(fresh)
        for _ in range(2):
            self.assertIsNone(step(s, running))
        self.assertEqual(s.waiting_queue, [replay, fresh])
        self.assertEqual(replay.match_calls, 2)
        self.assertEqual(fresh.match_calls, 0)
        self.assertFalse(replay.kv.holds_mamba)
        self.assertEqual(s.tree_cache.lock_depth, 0)

    def test_hrrn_uncached_work_counts_replay_output_and_cached_span(self):
        s, replay, _ = fixture(CANDIDATE)
        replay.origin_input_ids = range(118)
        replay.output_ids = range(854940)
        for matched, expected in (
            (0, 855058),
            (118, 854940),
            (2688, 852370),
            (855058, 0),
        ):
            with self.subTest(matched=matched):
                replay.num_matched_prefix_tokens = matched
                self.assertEqual(
                    CANDIDATE.SchedulePolicy._uncached_len(replay), expected
                )
        fresh = FixtureReq(CANDIDATE, s.tree_cache)
        fresh.api = CANDIDATE
        for matched in (0, 90, 119):
            fresh.num_matched_prefix_tokens = matched
            self.assertEqual(
                CANDIDATE.SchedulePolicy._uncached_len(fresh), max(0, 119 - matched)
            )

    def test_hrrn_orders_replay_by_full_cost_even_when_fill_snapshot_is_stale(self):
        s, replay, _ = fixture(CANDIDATE)
        replay.rid = "a-replay"
        replay.origin_input_ids = range(118)
        replay.output_ids = range(854940)
        replay.num_matched_prefix_tokens = 118
        # Policy matching happens before init_next_round_input refreshes this.
        self.assertEqual(len(replay.full_untruncated_fill_ids), 119)
        fresh = FixtureReq(CANDIDATE, s.tree_cache)
        fresh.api = CANDIDATE
        fresh.rid = "b-fresh"
        fresh.num_matched_prefix_tokens = 90
        queue = [replay, fresh]
        CANDIDATE.SchedulePolicy._sort_by_hrrn(queue, set(), processed_tokens=1000)
        self.assertEqual([req.rid for req in queue], ["b-fresh", "a-replay"])
        if BASELINE is not None:
            queue = [replay, fresh]
            BASELINE.SchedulePolicy._sort_by_hrrn(queue, set(), processed_tokens=1000)
            self.assertEqual([req.rid for req in queue], ["a-replay", "b-fresh"])

    def test_hrrn_fix_alone_does_not_open_latch_but_combined_admits_short(self):
        s, replay, running = fixture(CANDIDATE)
        replay.rid = "a-replay"
        replay.origin_input_ids = range(118)
        replay.output_ids = range(854940)
        replay.num_matched_prefix_tokens = 118
        replay.full_untruncated_fill_ids = range(855058)
        fresh = FixtureReq(CANDIDATE, s.tree_cache)
        fresh.api = CANDIDATE
        fresh.rid = "b-fresh"
        s.waiting_queue.append(fresh)
        # First refuse with old/equal order, establishing the full latch.
        self.assertIsNone(step(s, running))
        self.assertTrue(running.batch_is_full)
        s.token_to_kv_pool_allocator.free_tokens = 754048
        s.tree_cache.full_evictable_size = lambda: 2688

        def hrrn_priority(queue, *args, **kwargs):
            CANDIDATE.SchedulePolicy._sort_by_hrrn(queue, set(), processed_tokens=1000)

        s.policy.calc_priority = hrrn_priority
        if BASELINE is not None:
            # Old admission method still exits before the corrected policy runs.
            self.assertIsNone(
                BASELINE.Scheduler._get_new_batch_prefill_raw(s, None, running)[0]
            )
            self.assertEqual(s.waiting_queue, [replay, fresh])
        batch = step(s, running)
        self.assertIsNotNone(batch)
        self.assertEqual(batch.reqs, [fresh])
        self.assertEqual(s.waiting_queue, [replay])
        release_fixture(s, fresh)
        self.assertEqual(s.tree_cache.lock_depth, 0)


def benchmark(api, n=3000, *, exhausted_cow=False):
    samples = []
    counts = None
    for _ in range(3):
        s, _, running = fixture(api, free_slots=0 if exhausted_cow else 1)
        start = time.perf_counter_ns()
        for _ in range(n):
            step(s, running)
        elapsed = time.perf_counter_ns() - start
        alloc = s.req_to_token_pool.mamba_allocator
        samples.append(elapsed / n / 1000)
        counts = {
            "rounds": n,
            "priority_calls": s.priority_calls,
            "prefix_matches": s.tree_cache.match_calls,
            "alloc_calls": alloc.alloc_calls,
            "free_calls": alloc.free_calls,
            "eviction_calls": s.tree_cache.evict_calls,
        }
    return dict(
        median_us_per_round=statistics.median(samples), samples_us=samples, **counts
    )


def long_fill_copy_benchmark(api, n=100):
    """Actual prefix-policy concat and Req refresh; primitive match is a stub."""
    req = NS(
        origin_input_ids=array("q", range(118)),
        output_ids=array("q", [0]) * 854940,
        full_untruncated_fill_ids=array("q"),
        extra_key=None,
        cache_salt=None,
        prefix_len=2688,
    )
    start = time.perf_counter_ns()
    api.Req.refresh_fill_ids(req)
    first_refresh_us = (time.perf_counter_ns() - start) / 1000
    snapshot_id = id(req.full_untruncated_fill_ids)
    start = time.perf_counter_ns()
    for _ in range(n):
        api.Req.refresh_fill_ids(req)
    stable_refresh_us = (time.perf_counter_ns() - start) / n / 1000
    assert id(req.full_untruncated_fill_ids) == snapshot_id
    policy = api.SchedulePolicy.__new__(api.SchedulePolicy)
    policy.waiting_queue_radix_tree = NS(reset=lambda: None)
    policy.tree_cache = NS()
    match_calls = 0

    def mock_primitive_match(*args, **kwargs):
        nonlocal match_calls
        match_calls += 1

    globals_ = api.SchedulePolicy._compute_prefix_matches.__globals__
    with patch.dict(
        globals_,
        {
            "match_kv_cache": mock_primitive_match,
            "IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD": 32,
        },
    ):
        start = time.perf_counter_ns()
        for _ in range(n):
            policy._compute_prefix_matches([req], None)
        policy_100_ms = (time.perf_counter_ns() - start) / 1e6
    return {
        "rounds": n,
        "fill_tokens": 855058,
        "array_itemsize": req.output_ids.itemsize,
        "copied_bytes_per_policy_pass": 855058 * req.output_ids.itemsize,
        "first_refresh_us": first_refresh_us,
        "stable_refresh_us_per_call": stable_refresh_us,
        "stable_refresh_preserved_array_identity": True,
        "policy_100_rounds_ms": policy_100_ms,
        "policy_us_per_round": policy_100_ms * 1000 / n,
        "primitive_match_calls": match_calls,
        "boundary": "Real _compute_prefix_matches array concat; primitive match stub; prefix_len=2688 bypasses simulated-tree insertion; no real radix/Rust/tree transfer cost",
    }


REAL_IMPORTS = __name__ != "__main__"
BASELINE = None
CANDIDATE = None

if REAL_IMPORTS:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import maybe_stub_sgl_kernel

    maybe_stub_sgl_kernel()

    from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
    from sglang.srt.managers.schedule_policy import PrefillAdder, SchedulePolicy
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.base_prefix_cache import IncLockRefResult
    from sglang.srt.mem_cache.prefill_budget import PrefillBudget
    from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent

    register_cpu_ci(est_time=3, suite="base-a-test-cpu")
    CANDIDATE = NS(
        Scheduler=Scheduler,
        Req=Req,
        ReqKvInfo=ReqKvInfo,
        PrefillAdder=PrefillAdder,
        SchedulePolicy=SchedulePolicy,
        PrefillBudget=PrefillBudget,
        IncLockRefResult=IncLockRefResult,
        MambaComponent=MambaComponent,
    )
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--case", default="all")
    args = parser.parse_args()
    CANDIDATE = source_namespace(args.source_root, "candidate")
    BASELINE = source_namespace(args.baseline_root, "baseline")
    suite = (
        unittest.defaultTestLoader.loadTestsFromTestCase(AdmissionTests)
        if args.case == "all"
        else unittest.defaultTestLoader.loadTestsFromName(args.case, AdmissionTests)
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "python": sys.version,
        "baseline_root": str(args.baseline_root.resolve()),
        "candidate_root": str(args.source_root.resolve()),
        "tests_run": result.testsRun,
        "case": args.case,
        "success": result.wasSuccessful(),
        "failures": [(str(test), detail) for test, detail in result.failures],
        "errors": [(str(test), detail) for test, detail in result.errors],
        "source_hashes": CANDIDATE.hashes,
        "baseline_hashes": BASELINE.hashes,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "boundary": "AST-executed real source methods; finite mock slots/cache; no torch/GPU/real radix tree/full engine import",
    }
    if result.wasSuccessful() and args.case == "all":
        report["microbench"] = {
            "baseline_latched": benchmark(BASELINE),
            "candidate_repeated_budget_no_token": benchmark(CANDIDATE),
            "candidate_repeated_cow_no_slot": benchmark(CANDIDATE, exhausted_cow=True),
            "long_fill_copy": long_fill_copy_benchmark(CANDIDATE),
        }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"source_hashes", "baseline_hashes", "failures", "errors"}
            },
            indent=2,
        )
    )
    sys.exit(not result.wasSuccessful())
