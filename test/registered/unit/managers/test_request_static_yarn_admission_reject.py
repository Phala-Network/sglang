"""Real Scheduler/Req intake rejection lifecycle; no inference or GPU claim."""

from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.configs.request_static_yarn import RequestStaticYarnConfig
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.managers.scheduler import Scheduler, _MultimodalInputProcessingError
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.srt.sampling.sampling_params import SamplingParams


def yarn_config():
    return RequestStaticYarnConfig.from_hf_config(
        SimpleNamespace(
            rope_parameters={
                "rope_type": "yarn",
                "factor": 4.0,
                "request_static_factors": [1, 2, 4],
                "original_max_position_embeddings": 262144,
                "mrope_section": [11, 11, 10],
            }
        )
    )


def make_case(*, output, factor=None, mm=False, stream=False, logprob=False):
    params = SamplingParams(
        temperature=0,
        max_new_tokens=output,
        custom_params={"yarn_factor": factor} if factor is not None else None,
    )
    params.normalize(None)
    ids = array("q", range(23))
    recv = TokenizedGenerateReqInput(
        rid="intake-fixture",
        input_text="",
        input_ids=ids,
        input_embeds=None,
        mm_inputs=object() if mm else None,
        token_type_ids=None,
        sampling_params=params,
        return_logprob=logprob,
        logprob_start_len=0 if logprob else -1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=stream,
        bootstrap_port=1,
        extra_key="caller-namespace",
    )
    req = Req(
        recv.rid,
        recv.input_text,
        recv.input_ids,
        params,
        return_logprob=logprob,
        stream=stream,
        disagg_mode=DisaggregationMode.NULL,
        extra_key=recv.extra_key,
    )
    req.time_stats = MagicMock()
    req.init_next_round_input = MagicMock(
        side_effect=AssertionError("rejected request reached cache lookup")
    )

    scheduler = object.__new__(Scheduler)
    scheduler.enable_session_radix_cache = False
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.model_config = SimpleNamespace(
        hf_eos_token_id={0},
        vocab_size=128,
        context_len=1000000,
        request_static_yarn_config=yarn_config(),
    )
    scheduler.tokenizer = None
    scheduler.dllm_config = None
    scheduler.max_req_input_len = 999999
    scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
    scheduler.validate_dllm_request = MagicMock(return_value=None)
    scheduler.spec_algorithm = SimpleNamespace(
        is_dflash_family=lambda: False,
        is_uno=lambda: False,
        is_none=lambda: True,
    )
    scheduler.init_req_max_new_tokens = MagicMock()
    scheduler.waiting_queue = []
    scheduler.enable_priority_scheduling = False
    scheduler.abort_on_priority_when_disabled = True
    scheduler.max_queued_requests = None
    scheduler.enable_hicache_storage = False
    scheduler.enable_lmcache = False
    scheduler.processed_tokens_counter = 0
    scheduler._add_request_to_queue = MagicMock(wraps=scheduler._add_request_to_queue)
    scheduler.tree_cache = MagicMock()
    scheduler.tree_cache.cache_controller = SimpleNamespace()
    scheduler.beam_coordinator = MagicMock()
    scheduler.grammar_manager = MagicMock()
    scheduler.grammar_manager.process_req_with_grammar.return_value = False
    scheduler.output_streamer = MagicMock()
    scheduler.pad_input_ids_func = None
    scheduler._maybe_compute_mrope_positions = MagicMock()

    proxy = None
    if mm:
        # Reuse the deferred-transport lifecycle fixture: no CUDA allocation.
        proxy = object.__new__(CudaIpcTensorTransportProxy)
        proxy.total_consumer_count = 1
        proxy.acknowledge_consumption = MagicMock()
        mm_inputs = MultimodalInputs(
            mm_items=[MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)],
            mrope_positions=torch.arange(95, dtype=torch.int64).expand(3, -1),
        )
        scheduler._get_multimodal_inputs = MagicMock(return_value=mm_inputs)

        def pad(_recv, actual_req, _mm):
            actual_req.origin_input_ids = array("q", range(95))
            return True

        scheduler._try_apply_padded_mm_input_ids = MagicMock(side_effect=pad)
    return scheduler, recv, req, proxy


def handle(scheduler, recv, req, **kwargs):
    with (
        patch("sglang.srt.managers.scheduler.Req", return_value=req),
        patch(
            "sglang.srt.managers.scheduler.get_serving",
            return_value=SimpleNamespace(allow_auto_truncate=False),
        ),
        patch(
            "sglang.srt.managers.scheduler.get_device",
            return_value=SimpleNamespace(mlx_enable_sampling=False),
        ),
        patch(
            "sglang.srt.managers.scheduler.get_parallel",
            return_value=SimpleNamespace(pp_rank=0),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.get_parallel",
            return_value=SimpleNamespace(tp_rank=1),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.get_spec",
            return_value=SimpleNamespace(speculative_algorithm="EAGLE"),
        ),
    ):
        scheduler.handle_generate_request(recv, **kwargs)


def assert_local_rejection(
    scheduler, req, message, *, queued_validator=False, internal_error=False
):
    assert scheduler.waiting_queue == []
    if queued_validator:
        scheduler._add_request_to_queue.assert_called_once_with(req)
        scheduler.init_req_max_new_tokens.assert_called_once_with(req)
    else:
        scheduler._add_request_to_queue.assert_not_called()
        scheduler.init_req_max_new_tokens.assert_not_called()
    assert req.rope_profile is None
    assert req.extra_key == "caller-namespace"
    assert req.finished()
    assert isinstance(req.finished_reason, FINISH_ABORT)
    assert req.finished_reason.status_code == (
        HTTPStatus.INTERNAL_SERVER_ERROR if internal_error else HTTPStatus.BAD_REQUEST
    )
    assert req.finished_reason.err_type == (
        "InternalServerError" if internal_error else "BadRequestError"
    )
    assert message in req.finished_reason.message
    assert req.to_finish is None
    assert not req.kv.holds_kv and not req.kv.holds_mamba
    scheduler.grammar_manager.process_req_with_grammar.assert_not_called()
    req.init_next_round_input.assert_not_called()
    scheduler.tree_cache.match_prefix.assert_not_called()
    scheduler.tree_cache.finish.assert_called_once()
    assert scheduler.tree_cache.finish.call_args.args[0] == req.cache_request_handle
    scheduler.beam_coordinator.retire_group.assert_called_once_with(req)
    req.time_stats.trace_ctx.abort.assert_called_once()
    scheduler.output_streamer.stream_output.assert_called_once_with([req], False)


@pytest.mark.parametrize(
    "stream,logprob", [(False, False), (True, False), (False, True), (True, True)]
)
def test_mm_expansion_undercoverage_is_request_local(stream, logprob):
    # Exact observed boundary: 23 text IDs -> 95 MM IDs; 95 + 262050 = 262145.
    scheduler, recv, req, proxy = make_case(
        output=262050, factor=1, mm=True, stream=stream, logprob=logprob
    )
    handle(scheduler, recv, req)
    assert_local_rejection(scheduler, req, "selected YaRN profile capacity")
    proxy.acknowledge_consumption.assert_called_once_with(1)
    assert req.multimodal_inputs is None


@pytest.mark.parametrize(
    "output,factor,message",
    [
        (1000000, None, "server context length"),
        (524266, 2, "selected YaRN profile capacity"),
        (1, 3, "configured factor"),
    ],
)
def test_native_policy_rejections_are_request_local(output, factor, message):
    scheduler, recv, req, _ = make_case(output=output, factor=factor)
    handle(scheduler, recv, req)
    assert_local_rejection(scheduler, req, message)


def test_explicit_factor_on_disabled_feature_is_request_local():
    scheduler, recv, req, _ = make_case(output=1, factor=1)
    scheduler.model_config.request_static_yarn_config = None
    handle(scheduler, recv, req)
    assert_local_rejection(scheduler, req, "requires request-static YaRN")


def test_legal_request_after_rejection_keeps_normal_path():
    scheduler, recv, rejected, _ = make_case(output=262050, factor=1, mm=True)
    handle(scheduler, recv, rejected)
    assert_local_rejection(scheduler, rejected, "selected YaRN profile capacity")
    _, legal_recv, legal, _ = make_case(output=1, factor=2)
    handle(scheduler, legal_recv, legal)
    assert legal.rope_profile.factor == 2
    assert legal.rope_profile.budget == 24
    assert not legal.finished()
    assert scheduler.waiting_queue == [legal]
    scheduler._add_request_to_queue.assert_called_once_with(legal)
    scheduler.init_req_max_new_tokens.assert_called_once_with(legal)
    scheduler.grammar_manager.process_req_with_grammar.assert_called_once_with(legal)
    # The accepted request did not emit a second rejection completion.
    scheduler.output_streamer.stream_output.assert_called_once_with([rejected], False)


def test_auto_selection_still_uses_expanded_prompt_before_clipping():
    scheduler, recv, req, proxy = make_case(output=262050, mm=True)
    handle(scheduler, recv, req)
    assert req.rope_profile.factor == 2
    assert req.rope_profile.budget == 262145
    assert req.sampling_params.max_new_tokens == 262050
    assert not req.finished()
    assert scheduler.waiting_queue == [req]
    scheduler.init_req_max_new_tokens.assert_called_once_with(req)
    scheduler.output_streamer.stream_output.assert_not_called()
    scheduler.tree_cache.finish.assert_not_called()
    proxy.acknowledge_consumption.assert_not_called()


@pytest.mark.parametrize("failure", ["expanded-length", "processing", "fanout"])
def test_pre_admission_mm_abort_never_enters_prefill(failure):
    scheduler, recv, req, proxy = make_case(output=1, factor=1, mm=True)
    if failure == "expanded-length":
        scheduler.max_req_input_len = 95
        handle(scheduler, recv, req)
        message = "Multimodal prompt is too long"
        proxy.acknowledge_consumption.assert_called_once_with(1)
    elif failure == "processing":
        scheduler._get_multimodal_inputs.side_effect = _MultimodalInputProcessingError(
            "fixture MM processing failure"
        )
        handle(scheduler, recv, req)
        message = "fixture MM processing failure"
    else:
        handle(scheduler, recv, req, mm_input_error="fixture MM fanout failure")
        message = "fixture MM fanout failure"
    assert_local_rejection(
        scheduler,
        req,
        message,
        queued_validator=True,
        internal_error=failure != "expanded-length",
    )
    assert req.multimodal_inputs is None


def test_feature_disabled_keeps_existing_mm_abort_queue_path():
    scheduler, recv, req, proxy = make_case(output=1, mm=True)
    scheduler.model_config.request_static_yarn_config = None
    scheduler.max_req_input_len = 95
    handle(scheduler, recv, req)
    assert scheduler.waiting_queue == [req]
    assert not req.finished()
    assert isinstance(req.to_finish, FINISH_ABORT)
    proxy.acknowledge_consumption.assert_called_once_with(1)
    scheduler.output_streamer.stream_output.assert_not_called()
    scheduler.tree_cache.finish.assert_not_called()


@pytest.mark.parametrize("state", ["retracted", "holds-kv"])
def test_owned_or_retracted_abort_keeps_existing_queue_path(state):
    scheduler, _, req, _ = make_case(output=1)
    req.rope_profile = scheduler.model_config.request_static_yarn_config.select(24)
    req.to_finish = FINISH_ABORT("existing in-flight abort")
    if state == "retracted":
        req.is_retracted = True
    else:
        req.kv.req_pool_idx = 0
    scheduler._add_request_to_queue(req)
    assert scheduler.waiting_queue == [req]
    scheduler.output_streamer.stream_output.assert_not_called()
    scheduler.tree_cache.finish.assert_not_called()
