# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The base class of a backend for grammar-guided constrained decoding."""

import json
import logging
import math
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_parallel,
    get_resources,
    get_serving,
)
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

GRAMMAR_BACKEND_REGISTRY = {}


def grammar_cache_budget_enabled() -> bool:
    """The bounded cache/admission change is supported for PP1 only."""
    try:
        return get_parallel().pp_size == 1
    except ValueError:
        # Direct CPU use can precede publish(); serving publishes PP placement
        # before constructing any grammar backend.
        return True


@dataclass
class GrammarStats:
    compilation_time: Optional[float] = None
    schema_count: Optional[int] = None
    ebnf_size: Optional[int] = None
    is_cache_hit: bool = False
    is_grammar_aborted: bool = False
    tree_traversal_time: List[float] = field(default_factory=list)
    dispatch_type: Optional[str] = None
    num_timeout: int = 0


class GrammarRow(NamedTuple):
    """Grammar and destination row for a batched vocab-mask fill."""

    row: int
    grammar: "BaseGrammarObject"


class BaseGrammarObject:
    def __init__(self):
        self._finished = False
        self.grammar_stats = None
        self.current_token = None

    def maybe_init_reasoning(self, reasoning: bool):
        pass

    def accept_token(self, token: int) -> None:
        """
        Accept a token in the grammar.
        """
        raise NotImplementedError()

    def rollback(self, k: int):
        raise NotImplementedError()

    def is_terminated(self):
        return False

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        raise NotImplementedError()

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        raise NotImplementedError()

    @staticmethod
    def fill_vocab_mask_batched(
        entries: List[GrammarRow], vocab_mask: torch.Tensor
    ) -> None:
        """Fill listed rows, leaving unlisted rows untouched."""
        for entry in entries:
            entry.grammar.fill_vocab_mask(vocab_mask, entry.row)

    @staticmethod
    def reset_vocab_mask(vocab_mask: torch.Tensor) -> None:
        """Restore a reusable mask to the backend's unconstrained state."""
        raise NotImplementedError()

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        raise NotImplementedError()

    def copy(self) -> "BaseGrammarObject":
        return self

    def cache_size_bytes(self) -> int:
        """Approximate retained bytes; native backends include compiled contexts."""
        return sys.getsizeof(self)

    @property
    def finished(self):
        return self._finished

    @finished.setter
    def finished(self, finished):
        self._finished = finished

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        """
        Try to jump forward in the grammar.

        Returns:
            A jump forward helper which may be used in `jump_forward_str_state`.
            None if the jump forward is not possible.
        """
        raise NotImplementedError()

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        """
        Jump forward for the grammar.

        Returns:
            A tuple of the jump forward string and the next state of the grammar
            (which can be used in `jump_and_retokenize` if needed).
        """
        raise NotImplementedError()

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ) -> None:
        """
        Jump forward occurs, and update the grammar state if needed.
        """
        raise NotImplementedError()


class GrammarMask(NamedTuple):
    """A filled vocab_mask plus the backend that applies it.

    The grammar is any one of the batch's -- a handle, not per-request state.
    """

    grammar: BaseGrammarObject
    vocab_mask: torch.Tensor

    def apply(self, logits: torch.Tensor) -> None:
        self.grammar.apply_vocab_mask(logits=logits, vocab_mask=self.vocab_mask)


def _grammar_key_contains_nul(key_type: str, key_string: str) -> bool:
    """A NUL in a spec segfaults xgrammar's regex converter, which a JSON schema
    also reaches through `pattern` (possibly escaped). Drop once the upstream fix
    https://github.com/mlc-ai/xgrammar/pull/850 is in our pinned version.
    """
    if "\x00" in key_string:
        return True
    if key_type not in ("json", "structural_tag"):
        return False
    try:
        decoded = json.loads(key_string)
    except ValueError:
        # Malformed JSON: the backend's own parse reports it as a normal error.
        return False

    stack = [decoded]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            if "\x00" in node:
                return True
        elif isinstance(node, dict):
            stack.extend(node.keys())
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return False


class InvalidGrammarObject(BaseGrammarObject):
    """Represents a grammar that failed to compile, carrying the original error message."""

    def __init__(
        self,
        error_message: str = "Unknown grammar error",
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
    ):
        super().__init__()
        self.error_message = error_message
        self.status_code = status_code
        self.error_type = error_type

    def cache_size_bytes(self) -> int:
        return sys.getsizeof(self) + sys.getsizeof(self.error_message)

    def __repr__(self):
        return f"InvalidGrammarObject(error_message={self.error_message!r})"


class BaseGrammarBackend:
    _enable_strict_thinking: bool = False

    def __init__(self):
        self.use_cache_budget = grammar_cache_budget_enabled()
        if not self.use_cache_budget:
            # Preserve the existing PP cache/compiler ownership and scheduling.
            self.executor = ThreadPoolExecutor()
            self.cache = {}
            return
        self.cache: Dict[tuple, BaseGrammarObject] = OrderedDict()
        self._cache_weights = {}
        self._invalid_expirations = OrderedDict()
        self._cache_bytes = 0
        self._invalid_cache_bytes = 0
        self._cache_generation = 0
        self._cache_lock = threading.RLock()
        self._pending = {}
        self._inflight = {}
        self._pending_bytes = 0
        self._budget_observations = {
            "cache_lookup_hits": 0,
            "cache_lookup_misses": 0,
            "capacity_rejections": 0,
            "compiles_started": 0,
            "compiles_finished": 0,
            "compile_queue_seconds": 0.0,
            "compile_wall_seconds": 0.0,
        }
        self.cache_max_bytes = envs.SGLANG_GRAMMAR_CACHE_MAX_BYTES.get()
        self.cache_max_entries = envs.SGLANG_GRAMMAR_CACHE_MAX_ENTRIES.get()
        self.invalid_cache_max_bytes = envs.SGLANG_GRAMMAR_INVALID_CACHE_MAX_BYTES.get()
        self.invalid_cache_max_entries = (
            envs.SGLANG_GRAMMAR_INVALID_CACHE_MAX_ENTRIES.get()
        )
        self.invalid_cache_ttl = envs.SGLANG_GRAMMAR_INVALID_CACHE_TTL.get()
        self.max_pending_compiles = envs.SGLANG_GRAMMAR_MAX_PENDING_COMPILES.get()
        self.max_pending_bytes = envs.SGLANG_GRAMMAR_MAX_PENDING_BYTES.get()
        workers = envs.SGLANG_GRAMMAR_COMPILER_WORKERS.get()
        if workers < 1 or self.max_pending_compiles < 1 or self.max_pending_bytes < 1:
            raise ValueError(
                "Grammar compiler worker and pending limits must be positive"
            )
        if min(
            self.cache_max_bytes,
            self.cache_max_entries,
            self.invalid_cache_max_bytes,
            self.invalid_cache_max_entries,
            self.invalid_cache_ttl,
        ) < 0 or not math.isfinite(self.invalid_cache_ttl):
            raise ValueError("Grammar cache limits must be finite and non-negative")
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="grammar"
        )

    @staticmethod
    def _key_size_bytes(key: tuple) -> int:
        return sys.getsizeof(key) + sum(sys.getsizeof(part) for part in key)

    def cache_key(self, key: tuple, require_reasoning: bool) -> tuple:
        return key

    @property
    def cache_size_bytes(self) -> Optional[int]:
        # The preserved PP cache has no byte accounting.
        return self._cache_bytes if self.use_cache_budget else None

    def budget_snapshot(self) -> dict:
        """Fixed-size, nonblocking diagnostics; never traverse keys or ctxs.

        Cache bytes are the existing approximate admission charge, not RSS.
        Pending bytes cover input keys, not compiled results or live matchers.
        Counters last for the backend lifetime, including cache resets.
        """
        if not self.use_cache_budget:
            return {"available": False, "reason": "budget_disabled"}
        if not self._cache_lock.acquire(blocking=False):
            return {"available": False, "reason": "busy"}
        try:
            return {
                "available": True,
                "cache_generation": self._cache_generation,
                "cache_entries": len(self.cache),
                "approx_cache_bytes": self._cache_bytes,
                "cache_limit_bytes": self.cache_max_bytes,
                "pending_requests": len(self._pending),
                "pending_input_bytes": self._pending_bytes,
                "inflight_compiles": len(self._inflight),
                **self._budget_observations,
            }
        finally:
            self._cache_lock.release()

    def _remove_cache_entry(self, key):
        value = self.cache.pop(key, None)
        weight = self._cache_weights.pop(key, 0)
        self._cache_bytes -= weight
        if isinstance(value, InvalidGrammarObject):
            self._invalid_cache_bytes -= weight
            self._invalid_expirations.pop(key, None)

    def release_future(self, future):
        """Drop admission ownership; running work keeps its slot until completion."""
        if not self.use_cache_budget:
            return
        with self._cache_lock:
            entry = self._pending.get(future)
            if entry is None:
                return
            source = getattr(future, "_grammar_source", future)
            if source.done():
                self._pending.pop(future)
                self._pending_bytes -= entry[0]
            else:
                self._pending[future] = (entry[0], True)

    def _compile_context(self):
        """Reserve backend compilation admission before checking live waiters."""
        return nullcontext()

    def _run_compile(self, flight_key, key, require_reasoning):
        with self._cache_lock:
            _, waiters = self._inflight[flight_key]
            if all(
                waiter.cancelled() or self._pending.get(waiter, (0, True))[1]
                for waiter in waiters
            ):
                del self._inflight[flight_key]
                return None
        started = None
        try:
            with self._compile_context():
                with self._cache_lock:
                    source, waiters = self._inflight[flight_key]
                    if all(
                        waiter.cancelled() or self._pending.get(waiter, (0, True))[1]
                        for waiter in waiters
                    ):
                        del self._inflight[flight_key]
                        return None
                    self._budget_observations["compiles_started"] += 1
                    started = time.perf_counter()
                    self._budget_observations["compile_queue_seconds"] += (
                        started - source._grammar_submitted_at
                    )
                return self._init_value_dispatch(key, require_reasoning)
        finally:
            if started is not None:
                elapsed = time.perf_counter() - started
                with self._cache_lock:
                    self._budget_observations["compiles_finished"] += 1
                    self._budget_observations["compile_wall_seconds"] += elapsed

    def _complete_compile(self, flight_key, source):
        with self._cache_lock:
            waiters = source._grammar_waiters
            del source._grammar_waiters
            flight = self._inflight.get(flight_key)
            if flight is not None and flight[0] is source:
                self._inflight.pop(flight_key)
            for future in waiters:
                if future.cancelled():
                    self._release_abandoned_compile(future)
                    continue
                if not future.running() and not future.set_running_or_notify_cancel():
                    self._release_abandoned_compile(future)
                    continue
                try:
                    value = source.result()
                    # Each request owns its matcher; only compiled ctx is shared.
                    copied = value.copy() if value is not None else None
                    if copied is not None and copied.grammar_stats is not None:
                        copied.grammar_stats.is_cache_hit = False
                    future.set_result(copied)
                except Exception as exc:
                    future.set_exception(exc)
                self._release_abandoned_compile(future)

    def _release_abandoned_compile(self, future):
        with self._cache_lock:
            entry = self._pending.get(future)
            if entry is not None and entry[1]:
                self._pending.pop(future)
                self._pending_bytes -= entry[0]

    def initialize_vocab_mask_buffer(
        self,
        name: str,
        vocab_size: int,
        max_rows: int,
        device,
    ) -> Optional[torch.Tensor]:
        """Initialize a reusable mask buffer when supported by the backend."""
        return None

    def _not_supported(self, key_type: str, key_string: str) -> BaseGrammarObject:
        logger.warning("Skip unsupported grammar type %s", key_type)
        return InvalidGrammarObject()

    @property
    def enable_strict_thinking(self):
        return self._enable_strict_thinking

    def get_request_grammar_key(
        self,
        key: Tuple[str, str],
        *,
        input_ids: Sequence[int],
        thinking_budget: Optional[int],
    ) -> Tuple[str, str]:
        """Cache key for this request's grammar; may depend on how the prompt ends
        and on the request's thinking budget."""
        return key

    def reserve_framing_tokens(self, token_ids: Sequence[int]) -> None:
        """Let ``token_ids`` match only token-level framing, never grammar text."""

    def compile_framed_grammar(
        self, grammar: BaseGrammarObject, frame: Callable[[dict], dict]
    ) -> Optional[BaseGrammarObject]:
        """Compile ``frame(content)``, an XGrammar structural tag embedding ``grammar``
        as content; None if the backend cannot embed it."""
        return None

    @property
    def is_support_token_filter(self):
        return False

    def set_token_filter(
        self, vocab_mask, token_ids, batch_idx, is_allowed=True, reset_vocab_mask=True
    ):
        """Set or clear specific tokens in the vocab mask. No-op by default."""
        pass

    def init_strict_reasoning_grammar(self, reasoning: bool):
        """Create a grammar object for strict token filtering only. Returns None by default."""
        return None

    def dispatch_fallback(self, key_type: str, key_string: str) -> BaseGrammarObject:
        """
        This function should not be reached in any case.
        """
        raise ValueError(f"Invalid key_type: {key_type}={key_string}")

    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("json", key_string)

    def dispatch_regex(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("regex", key_string)

    def dispatch_ebnf(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("ebnf", key_string)

    def dispatch_structural_tag(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("structural_tag", key_string)

    def _init_value_dispatch(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> BaseGrammarObject:
        s = time.perf_counter()
        key_type, key_string = key
        if _grammar_key_contains_nul(key_type, key_string):
            logger.error(f"Rejecting {key_type} grammar containing a NUL byte")
            return InvalidGrammarObject(
                f"Invalid {key_type}: NUL bytes (\\u0000) are not allowed"
            )
        if key_type == "json":
            grammar = self.dispatch_json(key_string)
        elif key_type == "regex":
            grammar = self.dispatch_regex(key_string)
        elif key_type in ("ebnf", "full_assistant_ebnf"):
            grammar = self.dispatch_ebnf(key_string)
        elif key_type == "structural_tag":
            grammar = self.dispatch_structural_tag(key_string)
        else:
            grammar = self.dispatch_fallback(key_type, key_string)

        if grammar is not None and grammar.grammar_stats is not None:
            grammar.grammar_stats.compilation_time = time.perf_counter() - s
        return grammar

    def get_cached_or_future_value(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> Tuple[BaseGrammarObject | Future[BaseGrammarObject], bool]:
        if not self.use_cache_budget:
            value = self.cache.get(key)
            if value:
                copied_value = value.copy()
                copied_value.maybe_init_reasoning(require_reasoning)
                return copied_value, True
            return (
                self.executor.submit(self._init_value_dispatch, key, require_reasoning),
                False,
            )
        cache_key = self.cache_key(key, require_reasoning)
        with self._cache_lock:
            value = self.cache.get(cache_key)
            if isinstance(value, InvalidGrammarObject) and (
                time.monotonic() >= self._invalid_expirations.get(cache_key, 0)
            ):
                self._remove_cache_entry(cache_key)
                value = None
            if value is not None:
                self.cache.move_to_end(cache_key)
                self._budget_observations["cache_lookup_hits"] += 1
        if value is not None:
            copied_value = value.copy()
            copied_value.maybe_init_reasoning(require_reasoning)
            return copied_value, True
        with self._cache_lock:
            self._budget_observations["cache_lookup_misses"] += 1
            weight = self._key_size_bytes(cache_key)
            if (
                len(self._pending) >= self.max_pending_compiles
                or self._pending_bytes + weight > self.max_pending_bytes
            ):
                self._budget_observations["capacity_rejections"] += 1
                future = Future()
                future.set_result(
                    InvalidGrammarObject(
                        "Grammar compilation capacity exhausted; retry later",
                        status_code=503,
                        error_type="service_unavailable_error",
                    )
                )
            else:
                future = Future()
                flight_key = (self._cache_generation, cache_key)
                flight = self._inflight.get(flight_key)
                is_new = flight is None
                if is_new:
                    submitted_at = time.perf_counter()
                    source = self.executor.submit(
                        self._run_compile, flight_key, key[:2], require_reasoning
                    )
                    source._grammar_submitted_at = submitted_at
                    flight = self._inflight[flight_key] = (source, [])
                    source._grammar_waiters = flight[1]
                source, waiters = flight
                waiters.append(future)
                future._grammar_source = source
                future._grammar_flight_key = flight_key
                if source.running():
                    future.set_running_or_notify_cancel()
                self._pending[future] = (weight, False)
                self._pending_bytes += weight
                if is_new:
                    source.add_done_callback(
                        lambda completed, token=flight_key: self._complete_compile(
                            token, completed
                        )
                    )
            future._grammar_cache_generation = self._cache_generation
        return future, False

    def set_cache(
        self,
        key: tuple,
        value: BaseGrammarObject,
        require_reasoning: Optional[bool] = None,
        *,
        future: Optional[Future] = None,
    ):
        if not self.use_cache_budget:
            self.cache[key] = value
            return
        if value is None:
            return
        if require_reasoning is None:
            require_reasoning = getattr(value, "_kimik3_json_reasoning", False)
        key = self.cache_key(key, require_reasoning)
        retained = value.cache_size_bytes()
        if not isinstance(retained, int) or retained < 0:
            retained = sys.getsizeof(value)
        weight = self._key_size_bytes(key) + retained
        invalid = isinstance(value, InvalidGrammarObject)
        with self._cache_lock:
            if (
                future is not None
                and getattr(future, "_grammar_cache_generation", self._cache_generation)
                != self._cache_generation
            ):
                return
            self._remove_cache_entry(key)
            if (
                not self.cache_max_entries
                or weight > self.cache_max_bytes
                or (
                    invalid
                    and (
                        value.status_code != 400
                        or not self.invalid_cache_max_entries
                        or not self.invalid_cache_ttl
                        or weight > self.invalid_cache_max_bytes
                    )
                )
            ):
                return
            self.cache[key] = value
            self._cache_weights[key] = weight
            self._cache_bytes += weight
            if invalid:
                self._invalid_cache_bytes += weight
                self._invalid_expirations[key] = (
                    time.monotonic() + self.invalid_cache_ttl
                )
                while (
                    len(self._invalid_expirations) > self.invalid_cache_max_entries
                    or self._invalid_cache_bytes > self.invalid_cache_max_bytes
                ):
                    self._remove_cache_entry(next(iter(self._invalid_expirations)))
            while (
                len(self.cache) > self.cache_max_entries
                or self._cache_bytes > self.cache_max_bytes
            ):
                self._remove_cache_entry(next(iter(self.cache)))

    def reset(self):
        if not self.use_cache_budget:
            self.cache.clear()
            return
        with self._cache_lock:
            self._cache_generation += 1
            self.cache.clear()
            self._cache_weights.clear()
            self._invalid_expirations.clear()
            self._cache_bytes = self._invalid_cache_bytes = 0
            for future in list(self._pending):
                self.release_future(future)
                future.cancel()


def register_vocab_mask_buffer(
    name: str, vocab_mask: torch.Tensor, max_rows: int
) -> torch.Tensor:
    """Register a fixed-capacity mask buffer, preserving an equivalent one."""
    if max_rows <= 0:
        raise ValueError(f"Grammar mask max_rows must be positive, got {max_rows}")
    if vocab_mask.ndim == 0 or vocab_mask.shape[0] != max_rows:
        raise ValueError(
            f"Grammar mask buffer {name!r} must have {max_rows} rows, "
            f"got shape {tuple(vocab_mask.shape)}"
        )

    buffers = get_resources().buffers
    existing = buffers.get(name)
    if existing is not None:
        if (
            existing.shape != vocab_mask.shape
            or existing.dtype != vocab_mask.dtype
            or existing.device != vocab_mask.device
        ):
            raise RuntimeError(
                f"Grammar mask buffer {name!r} was already initialized as "
                f"{tuple(existing.shape)}, {existing.dtype}, {existing.device}; "
                f"new buffer is {tuple(vocab_mask.shape)}, {vocab_mask.dtype}, "
                f"{vocab_mask.device}"
            )
        return existing

    buffers[name] = vocab_mask
    return vocab_mask


def get_vocab_mask_buffer(name: str, rows: int) -> Optional[torch.Tensor]:
    """Return the active rows of a registered mask buffer, if available."""
    vocab_mask = get_resources().buffers.get(name)
    if vocab_mask is None:
        return None
    if rows > vocab_mask.shape[0]:
        raise ValueError(
            f"Grammar batch needs {rows} mask rows, exceeding initialized "
            f"capacity {vocab_mask.shape[0]} for {name!r}"
        )
    return vocab_mask[:rows]


def register_grammar_backend(name, init_func):
    GRAMMAR_BACKEND_REGISTRY[name] = init_func


def create_grammar_backend(
    server_args: ServerArgs,
    tokenizer,
    vocab_size: int,
    eos_token_ids: Optional[set] = None,
    think_end_ids: Optional[List[int]] = None,
) -> Optional[BaseGrammarBackend]:
    name = get_exec().kernel.grammar_backend
    if not grammar_cache_budget_enabled():
        logger.info(
            "Grammar cache/admission budgets apply to PP1; PP uses its existing cache path"
        )

    # Custom grammar backend has the highest priority
    if name in GRAMMAR_BACKEND_REGISTRY:
        return GRAMMAR_BACKEND_REGISTRY[name](
            server_args, tokenizer, vocab_size, eos_token_ids
        )

    # Default grammar backends
    if name == "outlines":
        from sglang.srt.constrained.outlines_backend import OutlinesGrammarBackend

        grammar_backend = OutlinesGrammarBackend(
            tokenizer,
            whitespace_pattern=get_serving().constrained_json_whitespace_pattern,
        )
    elif name == "xgrammar":
        from sglang.srt.constrained.xgrammar_backend import (
            TokenizerNotSupportedError,
            XGrammarGrammarBackend,
        )

        # Convert Set[int] to List[int] if needed
        eos_list = list(eos_token_ids) if eos_token_ids else None

        try:
            grammar_backend = XGrammarGrammarBackend(
                tokenizer,
                vocab_size=vocab_size,
                model_eos_token_ids=eos_list,
                any_whitespace=not get_serving().constrained_json_disable_any_whitespace,
                max_whitespace_cnt=get_serving().constrained_json_max_whitespace_cnt,
            )
        except TokenizerNotSupportedError as e:
            if get_serving().enable_strict_thinking:
                raise ValueError(
                    f"--enable-strict-thinking requires a grammar backend with "
                    f"token filtering support, but XGrammar failed to initialize: "
                    f"{e}. Cannot fall back to grammar_backend='none' with strict "
                    f"thinking enabled."
                ) from e
            logger.warning(
                f"Grammar backend disabled because tokenizer is not supported by XGrammar: {e}. "
                "Falling back to grammar_backend='none'. "
                "Structured outputs (JSON schema, regex, EBNF) will not be available."
            )
            get_context().override("grammar.import_fallback", grammar_backend="none")
            return None
    elif name == "llguidance":
        from sglang.srt.constrained.llguidance_backend import GuidanceBackend

        grammar_backend = GuidanceBackend(
            tokenizer=tokenizer,
            any_whitespace=not get_serving().constrained_json_disable_any_whitespace,
            whitespace_pattern=get_serving().constrained_json_whitespace_pattern,
            n_vocab=vocab_size,
            eos_token_ids=eos_token_ids,
        )
    elif name == "none":
        if get_serving().enable_strict_thinking:
            raise ValueError(
                "--enable-strict-thinking requires a grammar backend that supports "
                "token filtering, but grammar_backend='none' was specified. Use "
                "--grammar-backend xgrammar or another backend that supports token "
                "filtering."
            )
        return None
    else:
        raise ValueError(f"Invalid grammar backend: {name}")

    if get_serving().reasoning_parser and think_end_ids:
        from sglang.srt.constrained.reasoner_grammar_backend import (
            ReasonerGrammarBackend,
        )

        reasoning_parser = ReasoningParser(
            model_type=get_serving().reasoning_parser,
            stream_reasoning=False,
            tokenizer=tokenizer,
        )

        grammar_backend = ReasonerGrammarBackend(
            grammar_backend,
            reasoning_parser,
            tokenizer,
            enable_strict_thinking=get_serving().enable_strict_thinking,
        )

    return grammar_backend
