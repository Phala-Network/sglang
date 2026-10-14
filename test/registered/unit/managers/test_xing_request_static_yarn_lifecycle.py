"""Actual Scheduler/Req budget preservation, local rejection and retraction."""

from array import array
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest

from test_request_static_yarn_admission_reject import make_case, handle
from sglang.srt.configs.request_static_yarn import RequestStaticYarnConfig, admit_request_static_yarn
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator


def xing_config():
    return RequestStaticYarnConfig.from_hf_config(SimpleNamespace(
        model_type="xing4_0", rope_theta=10000,
        rope_scaling={"rope_type": "yarn", "factor": 128,
            "request_static_factors": [64, 128], "original_max_position_embeddings": 4096,
            "beta_fast": 32, "beta_slow": 1, "mscale": 1, "mscale_all_dim": 1},
    ))


def intake(prompt, output, factor=None, pool_capacity=524416):
    scheduler, recv, req, _ = make_case(output=output, factor=factor)
    req.origin_input_ids = array("q", (i % 128 for i in range(prompt)))
    req.origin_input_ids_unpadded = req.origin_input_ids
    recv.input_ids = req.origin_input_ids
    scheduler.model_config.context_len = 524288
    scheduler.model_config.request_static_yarn_config = xing_config()
    scheduler.max_req_len = 524287  # Legacy worker limit would clip boundary.
    scheduler.max_req_input_len = 524288
    scheduler.max_new_tokens_limit = None
    scheduler.max_total_num_tokens = pool_capacity
    scheduler.kv_shard_widening = 1
    scheduler.page_size = 64
    scheduler.sliding_window_size = None
    scheduler.chunked_prefill_size = 4096
    allocator = SimpleNamespace(page_size=64)
    allocator.max_new_tokens_for_memory = MethodType(
        BaseTokenToKVPoolAllocator.max_new_tokens_for_memory, allocator
    )
    scheduler.token_to_kv_pool_allocator = allocator
    scheduler.init_req_max_new_tokens = MethodType(Scheduler.init_req_max_new_tokens, scheduler)
    return scheduler, recv, req


@pytest.mark.parametrize("prompt,output,factor", [
    (261888, 256, 64), (261889, 256, 128), (524032, 256, 128),
])
def test_real_intake_public_boundaries_preserve_exact_original_output(prompt, output, factor):
    scheduler, recv, req = intake(prompt, output)
    handle(scheduler, recv, req)
    assert req.rope_profile.factor == factor
    assert req.rope_profile.budget == prompt + output
    assert req.rope_profile.reserved_tokens == (-(-prompt // 64) * 64 - prompt + 65)
    assert req.sampling_params.max_new_tokens == output
    assert scheduler.waiting_queue == [req]
    assert not req.finished()
    scheduler.output_streamer.stream_output.assert_not_called()


def test_raw_overflow_rejects_before_allocator_cache_and_short_request_recovers():
    scheduler, recv, rejected = intake(524032, 257)
    handle(scheduler, recv, rejected)
    assert rejected.finished()
    assert "server context" in rejected.finished_reason.message
    assert rejected.rope_profile is None
    assert rejected.sampling_params.max_new_tokens == 257
    assert scheduler.waiting_queue == []
    rejected.init_next_round_input.assert_not_called()
    _, legal_recv, legal = intake(16, 4)
    handle(scheduler, legal_recv, legal)
    assert legal.rope_profile.factor == 64
    assert scheduler.waiting_queue == [legal]
    assert not legal.finished()


def test_insufficient_empty_pool_and_output_limit_reject_without_clipping():
    # Headroom is outside public 524288. A pool of exactly 524288 cannot
    # accommodate page64's independent spare page and allocator sentinel.
    for output_limit in (None, 255):
        scheduler, recv, req = intake(524032, 256, pool_capacity=(524288 if output_limit is None else 524416))
        scheduler.max_new_tokens_limit = output_limit
        handle(scheduler, recv, req)
        assert req.finished() and "does not clip" in req.finished_reason.message
        assert req.sampling_params.max_new_tokens == 256
        assert req.rope_profile.budget == 524288
        assert scheduler.waiting_queue == []
        scheduler.grammar_manager.process_req_with_grammar.assert_not_called()
        req.init_next_round_input.assert_not_called()
        scheduler.tree_cache.finish.assert_called_once()


def test_null_budget_fills_public_context_and_real_retraction_keeps_profile():
    scheduler, recv, req = intake(32, None)
    handle(scheduler, recv, req)
    assert req.rope_profile.factor == 128
    assert req.rope_profile.budget == 524288
    assert req.sampling_params.max_new_tokens == 524256
    profile, namespace = req.rope_profile, req.extra_key
    req.reset_for_retract()
    req.sampling_params.custom_params = {"yarn_factor": 64}
    admit_request_static_yarn(req, xing_config(), 524288)
    assert req.rope_profile is profile and req.extra_key == namespace
    assert req.is_retracted and req.retraction_count == 1


def test_real_worker_public_limits_do_not_consume_internal_kv_headroom():
    runner = SimpleNamespace(
        effective_logical_max_total_num_tokens=524160,
        logical_max_total_num_tokens=524160, token_to_kv_pool_allocator=object(),
        max_running_requests=8, forward_stream=None,
        req_to_token_pool=SimpleNamespace(size=8, max_context_len=524288),
        token_to_kv_pool=SimpleNamespace(size=524160),
    )
    worker = SimpleNamespace(
        model_config=SimpleNamespace(context_len=524288, request_static_yarn_config=xing_config()),
        model_runner=runner, dllm_algorithm=None, random_seed=1, device="cuda",
    )
    with patch("sglang.srt.managers.tp_worker.page_interleave_shard_size", return_value=1), patch(
        "sglang.srt.managers.tp_worker.get_schedule",
        return_value=SimpleNamespace(max_prefill_tokens=8192, max_queued_requests=None),
    ):
        info = TpModelWorker.get_worker_info(worker)
        assert info[4:6] == (524288, 524288)
        worker.model_config.request_static_yarn_config = None
        assert TpModelWorker.get_worker_info(worker)[4:6] == (524159, 524154)
