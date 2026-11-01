from __future__ import annotations

import logging
import time
from concurrent import futures
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.constrained.base_grammar_backend import (
    InvalidGrammarObject,
    create_grammar_backend,
)
from sglang.srt.constrained.reasoner_grammar_backend import (
    KIMIK3_REQUIRED_NATIVE_BOUNDARY,
    ReasonerGrammarObject,
)
from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import matches_abort_rid
from sglang.srt.runtime_context import get_parallel, get_serving
from sglang.srt.sampling.sampling_params import (
    get_request_reasoning_end_token_ids,
    get_request_reasoning_in_grammar,
)

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class GrammarManager:
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.server_args = scheduler.server_args
        self.grammar_queue: List[Req] = []
        if not get_serving().skip_tokenizer_init:
            self.grammar_backend = create_grammar_backend(
                self.server_args,
                scheduler.tokenizer,
                scheduler.model_config.vocab_size,
                scheduler.model_config.hf_eos_token_id,
                think_end_ids=scheduler.model_config.think_end_ids,
            )
        else:
            self.grammar_backend = None

        self._enable_strict_thinking = (
            self.grammar_backend.enable_strict_thinking
            if self.grammar_backend is not None
            else False
        )

        self.grammar_sync_group = scheduler.dp_tp_cpu_group
        self.grammar_sync_size = scheduler.dp_tp_group.world_size
        self.grammar_sync_entry = scheduler.dp_tp_group.first_rank
        self.is_grammar_sync_entry = scheduler.dp_tp_group.is_first_rank
        self.pp_rank = get_parallel().pp_rank
        self.pp_size = get_parallel().pp_size
        self.pp_group = scheduler.pp_group
        self.grammar_pp_sync_work_list = []
        self._synced_grammar_errors = {}

        self.SGLANG_GRAMMAR_POLL_INTERVAL = envs.SGLANG_GRAMMAR_POLL_INTERVAL.get()
        self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = (
            envs.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS.get()
        )

    def __len__(self):
        return len(self.grammar_queue)

    def clear(self):
        if self.grammar_backend:
            self.grammar_backend.reset()

    def has_waiting_grammars(self) -> bool:
        return len(self.grammar_queue) > 0

    def _drain_pp_sync_work(self):
        for p2p_work in self.grammar_pp_sync_work_list:
            p2p_work.work.wait()
        self.grammar_pp_sync_work_list.clear()

    def _pp_sync_ready_failed(
        self,
        ready_req_idxs: set[int],
        failed_req_idxs: set[int],
    ) -> tuple[set[int], set[int]]:
        """Keep PP's existing one-way, per-microbatch admission decision."""
        if self.pp_size <= 1 or self.pp_group is None:
            return ready_req_idxs, failed_req_idxs
        self._drain_pp_sync_work()
        data = (ready_req_idxs, failed_req_idxs)
        if self.pp_rank > 0:
            data = self.pp_group.recv_object(
                src=self.pp_rank - 1,
                tag=P2PTag.GRAMMAR_PP_SYNC,
            )
        if self.pp_rank + 1 < self.pp_size:
            self.grammar_pp_sync_work_list.extend(
                self.pp_group.send_object(
                    data,
                    dst=self.pp_rank + 1,
                    async_send=True,
                    tag=P2PTag.GRAMMAR_PP_SYNC,
                )
            )
        return data

    @staticmethod
    def _merge_grammar_errors(gathered):
        merged = {}
        for _, _, errors in gathered:
            for idx, error in errors.items():
                previous = merged.get(idx)
                if previous is None or error[1] > previous[1]:
                    merged[idx] = error
        return merged

    @staticmethod
    def _future_grammar_error(future):
        try:
            value = future.result()
        except futures.CancelledError:
            value = InvalidGrammarObject(
                "Grammar compilation was cancelled; retry later",
                status_code=503,
                error_type="service_unavailable_error",
            )
        except Exception as exc:
            logger.error("Grammar compiler raised %s", type(exc).__name__)
            value = InvalidGrammarObject(
                f"Grammar compilation failed ({type(exc).__name__})",
                status_code=500,
                error_type="internal_server_error",
            )
        if value is None:
            value = InvalidGrammarObject(
                "Grammar compiler returned no grammar",
                status_code=500,
                error_type="internal_server_error",
            )
        return value

    @staticmethod
    def _abort_invalid_grammar(req, key_type, value):
        req.set_finish_with_abort(
            f"Failed to compile {key_type} grammar: {value.error_message}",
            status_code=value.status_code,
            err_type=value.error_type,
        )

    def _release_request_future(self, req):
        future = getattr(req, "_grammar_future", None)
        if future is None and isinstance(req.grammar, futures.Future):
            future = req.grammar
        if future is not None:
            future.cancel()
            self.grammar_backend.release_future(future)
        if hasattr(req, "_grammar_future"):
            del req._grammar_future

    def abort_requests(self, recv_req: AbortReq):
        for req in self.grammar_queue:
            if recv_req.abort_all or matches_abort_rid(req.rid, recv_req.rid):
                logger.debug(f"Abort grammar queue request. {req.rid=}")
                self._release_request_future(req)
                req.set_finish_with_abort("Aborted by AbortReq.")

    def _get_request_thinking_bounds(self, req: Req) -> tuple[int | None, int | None]:
        custom_params = req.sampling_params.custom_params
        if not isinstance(custom_params, dict):
            return None, None
        min_thinking_budget = custom_params.get("min_thinking_budget")
        thinking_budget = custom_params.get("thinking_budget")
        return (
            min_thinking_budget if isinstance(min_thinking_budget, int) else None,
            thinking_budget if isinstance(thinking_budget, int) else None,
        )

    def _apply_request_reasoning_config(self, req: Req) -> None:
        custom_params = req.sampling_params.custom_params
        native_k3 = (
            isinstance(custom_params, dict)
            and custom_params.get(KIMIK3_REQUIRED_NATIVE_BOUNDARY) is True
        )
        if native_k3 and not isinstance(req.grammar, InvalidGrammarObject):
            try:
                if (
                    not isinstance(req.grammar, ReasonerGrammarObject)
                    or get_serving().reasoning_parser != "kimi_k3"
                    or req.sampling_params.structural_tag is None
                ):
                    raise ValueError(
                        "K3 native boundary requires its reasoning/tool grammar"
                    )
                tokenizer = self.scheduler.tokenizer
                model_eos = self.scheduler.model_config.hf_eos_token_id
                stop_ids = (
                    {model_eos} if isinstance(model_eos, int) else set(model_eos or [])
                )
                if tokenizer.eos_token_id is not None:
                    stop_ids.add(tokenizer.eos_token_id)
                stop_ids.update(
                    getattr(tokenizer, "additional_stop_token_ids", None) or []
                )
                req.grammar.enable_kimik3_required_native_boundary(
                    tokenizer, list(stop_ids)
                )
            except (ValueError, RuntimeError) as exc:
                req.set_finish_with_abort(
                    f"Unable to configure K3 native boundary: {exc}"
                )
                return
        if not isinstance(req.grammar, ReasonerGrammarObject):
            return
        think_end_ids = get_request_reasoning_end_token_ids(
            req.sampling_params.custom_params,
            allowed_sequences=getattr(
                self.scheduler.model_config,
                "request_selectable_think_end_id_sequences",
                None,
            ),
        )
        if think_end_ids is not None:
            req.grammar.set_request_think_end_ids(think_end_ids)
        if get_request_reasoning_in_grammar(req.sampling_params.custom_params):
            req.grammar.set_reasoning_in_grammar()
        min_thinking_budget, thinking_budget = self._get_request_thinking_bounds(req)
        if min_thinking_budget is not None:
            req.grammar.min_think_tokens = min_thinking_budget
        if thinking_budget is not None:
            req.grammar.max_think_tokens = thinking_budget

    def _strict_thinking_grammar_disabled(self, req: Req) -> bool:
        custom_params = req.sampling_params.custom_params
        return (
            isinstance(custom_params, dict)
            and custom_params.get("disable_strict_thinking_grammar") is True
        )

    def process_req_with_grammar(self, req: Req) -> bool:
        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            if self.grammar_backend is None:
                error_msg = "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported when the server is launched with --grammar-backend none"
                req.set_finish_with_abort(error_msg)
            else:
                if req.sampling_params.json_schema is not None:
                    key = ("json", req.sampling_params.json_schema)
                elif req.sampling_params.regex is not None:
                    key = ("regex", req.sampling_params.regex)
                elif req.sampling_params.ebnf is not None:
                    key_type = (
                        "full_assistant_ebnf"
                        if req.sampling_params.ebnf_full_assistant
                        else "ebnf"
                    )
                    key = (key_type, req.sampling_params.ebnf)
                elif req.sampling_params.structural_tag is not None:
                    key = ("structural_tag", req.sampling_params.structural_tag)
                key = self.grammar_backend.get_request_grammar_key(
                    key,
                    input_ids=req.origin_input_ids,
                    thinking_budget=self._get_request_thinking_bounds(req)[1],
                )

                value, cache_hit = self.grammar_backend.get_cached_or_future_value(
                    key, req.require_reasoning
                )
                if cache_hit and self.pp_size > 1:
                    req.grammar = value
                    if isinstance(value, InvalidGrammarObject):
                        req.set_finish_with_abort(
                            f"Failed to compile {key[0]} grammar: {value.error_message}"
                        )
                    else:
                        self._apply_request_reasoning_config(req)
                    return False
                if cache_hit:
                    future = futures.Future()
                    future.set_result(value)
                    future._grammar_from_cache = True
                    value = future
                req.grammar = value
                req._grammar_future = value
                req.grammar_key = key
                # LRU eviction, TTL, cancellation and compilation timing can
                # differ by rank. Every grammar request enters the same ordered
                # queue, including hot copies, before the existing batched sync.
                add_to_grammar_queue = True
        elif (
            self._enable_strict_thinking
            and not self._strict_thinking_grammar_disabled(req)
        ):
            grammar_obj = self.grammar_backend.init_strict_reasoning_grammar(
                req.require_reasoning
            )
            if grammar_obj is not None:
                req.grammar = grammar_obj
                self._apply_request_reasoning_config(req)

        if add_to_grammar_queue:
            self.grammar_queue.append(req)

        return add_to_grammar_queue

    def _get_ready_local_state(self):
        """Poll locally and combine this stage's DP/TP compiler state."""
        assert self.grammar_backend
        ready_req_idxs: set[int] = set()
        failed_req_idxs: set[int] = set()
        grammar_errors = {}

        # Poll for ready requests
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < self.SGLANG_GRAMMAR_POLL_INTERVAL:
            for i, req in enumerate(self.grammar_queue):
                if i in ready_req_idxs:
                    continue

                if req.finished() or req.grammar is None:  # It is aborted by AbortReq
                    ready_req_idxs.add(i)
                    continue

                assert isinstance(req.grammar, futures.Future), f"{req=}"
                if req.grammar.done():
                    ready_req_idxs.add(i)
                    value = self._future_grammar_error(req.grammar)
                    if isinstance(value, InvalidGrammarObject):
                        grammar_errors[i] = (
                            value.error_message,
                            value.status_code,
                            value.error_type,
                        )

            if len(ready_req_idxs) == len(self.grammar_queue):
                break

            # Sleep a bit to avoid busy waiting
            time.sleep(self.SGLANG_GRAMMAR_POLL_INTERVAL / 10)

        # Check failed requests
        for i, req in enumerate(self.grammar_queue):
            if i not in ready_req_idxs:
                # Count local waits on every stage; a slow stage can time out.
                self.grammar_queue[i].grammar_wait_ct += 1
                if (
                    self.grammar_queue[i].grammar_wait_ct
                    >= self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS
                ):
                    # Timeout after max poll iterations
                    # The actual waiting time is SGLANG_GRAMMAR_MAX_POLL_ITERATIONS * max(SGLANG_GRAMMAR_POLL_INTERVAL, GPU_forward_batch_latency)
                    failed_req_idxs.add(i)

        # Sync readiness, timeouts and typed errors across this stage's DP/TP.
        if self.grammar_sync_size == 1:
            synced_ready_req_idxs = ready_req_idxs
            synced_failed_req_idxs = failed_req_idxs
        else:
            all_gather_output = [None] * self.grammar_sync_size
            torch.distributed.all_gather_object(
                all_gather_output,
                (ready_req_idxs, failed_req_idxs, grammar_errors),
                group=self.grammar_sync_group,
            )
            synced_ready_req_idxs = set.intersection(*[x[0] for x in all_gather_output])
            synced_failed_req_idxs = set.union(*[x[1] for x in all_gather_output])
            # All ranks make the same failure decision even when one rank
            # rejects local compiler capacity while another compiles or hits.
            grammar_errors = self._merge_grammar_errors(all_gather_output)

        return synced_ready_req_idxs, synced_failed_req_idxs, grammar_errors

    def get_ready_grammar_requests(self) -> List[Req]:
        """PP1 uses bounded admission; PP retains PP0's previous decision path."""
        if self.pp_size <= 1 or self.pp_rank == 0:
            ready, failed, errors = self._get_ready_local_state()
        else:
            ready, failed, errors = set(), set(), {}
        self._synced_grammar_errors = errors if self.pp_size <= 1 else {}
        (
            synced_ready_req_idxs,
            synced_failed_req_idxs,
        ) = self._pp_sync_ready_failed(ready, failed)

        # Return ready requests
        return_reqs: List[Req] = []
        for i in sorted(synced_ready_req_idxs):
            req = self.grammar_queue[i]
            return_reqs.append(req)
            if req.finished() or req.grammar is None:  # It is aborted by AbortReq
                self._release_request_future(req)
                continue

            assert isinstance(req.grammar, futures.Future) and req.grammar_key
            future = req.grammar
            req.grammar = self._future_grammar_error(future)
            if error := self._synced_grammar_errors.get(i):
                req.grammar = InvalidGrammarObject(
                    error[0], status_code=error[1], error_type=error[2]
                )
            try:
                if not getattr(future, "_grammar_from_cache", False):
                    self.grammar_backend.set_cache(
                        req.grammar_key,
                        req.grammar.copy(),
                        req.require_reasoning,
                        future=future,
                    )
            finally:
                self._release_request_future(req)
            if isinstance(req.grammar, InvalidGrammarObject):
                if self.pp_size > 1:
                    req.set_finish_with_abort(
                        f"Failed to compile {req.grammar_key[0]} grammar: {req.grammar.error_message}"
                    )
                else:
                    self._abort_invalid_grammar(req, req.grammar_key[0], req.grammar)
            else:
                self._apply_request_reasoning_config(req)

        # Return failed requests
        for i in sorted(synced_failed_req_idxs):
            req = self.grammar_queue[i]
            return_reqs.append(req)

            if req.finished() or req.grammar is None:
                self._release_request_future(req)
                continue

            assert isinstance(req.grammar, futures.Future) and req.grammar_key
            future = req.grammar
            future.cancel()
            self.grammar_backend.set_cache(
                req.grammar_key,
                InvalidGrammarObject(
                    "Grammar preprocessing timed out",
                    status_code=504,
                    error_type="timeout_error",
                ),
                req.require_reasoning,
                future=future,
            )
            self._release_request_future(req)
            if self.pp_size > 1:
                req.set_finish_with_abort("Grammar preprocessing timed out")
            else:
                req.set_finish_with_abort(
                    "Grammar preprocessing timed out",
                    status_code=504,
                    err_type="timeout_error",
                )

        # Remove finished requests from grammar_queue
        self.grammar_queue = [
            req
            for i, req in enumerate(self.grammar_queue)
            if i not in synced_ready_req_idxs and i not in synced_failed_req_idxs
        ]
        return return_reqs
