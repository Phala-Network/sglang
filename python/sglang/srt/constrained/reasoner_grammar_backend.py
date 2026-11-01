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
"""The baseclass of a backend for reasoner grammar-guided constrained decoding."""

import json
import logging
import sys
from collections import OrderedDict
from concurrent.futures import Future
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sglang.srt.environ import envs
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.parser.response_template import ResponseTemplateReasoningDetector
from sglang.srt.utils.token_sequence_matcher import TokenSequenceMatcher

from .base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)

logger = logging.getLogger(__name__)

_OPEN_ANSWER_VARIANT = "open_answer"
_THINK_BUDGET_VARIANT = "think_budget="
# Arbitrary bound; budgets are a per-request knob, so only recent variants stay cached.
_BUDGET_VARIANT_CACHE_SIZE = 32

# Set by the chat renderer, never inferred from a user-provided grammar.
KIMIK3_REQUIRED_NATIVE_BOUNDARY = "__sglang_kimik3_required_native_boundary"


class ReasonerGrammarObject(BaseGrammarObject):
    """Defers grammar constraints until the reasoning end sequence is complete."""

    def __init__(
        self,
        grammar: Optional[BaseGrammarObject],
        think_end_ids: Sequence[int],
        think_excluded_token_ids: Optional[List[int]] = None,
        min_think_tokens: int = -1,
        max_think_tokens: int = -1,
        enable_token_filter: bool = False,
        token_filter_fn=None,
        allocate_vocab_mask_fn=None,
        move_vocab_mask_fn=None,
        apply_vocab_mask_fn=None,
        think_starts: Sequence[str] = (),
        decode: Optional[Callable[[List[int]], str]] = None,
    ):
        super().__init__()
        self.grammar = grammar
        self.think_end_ids = tuple(think_end_ids)
        # Decoded openers match every tokenization the grammar admits; without any,
        # the reasoning starts at the first token.
        self.think_starts = tuple(think_starts)
        self.decode = decode
        self._think_end_matcher = TokenSequenceMatcher(self.think_end_ids)
        self.think_excluded_token_ids = think_excluded_token_ids
        self.min_think_tokens = min_think_tokens
        self.max_think_tokens = max_think_tokens
        self.enable_token_filter = enable_token_filter
        self.token_filter_fn = token_filter_fn
        self.allocate_vocab_mask_fn = allocate_vocab_mask_fn
        self.move_vocab_mask_fn = move_vocab_mask_fn
        self.apply_vocab_mask_fn = apply_vocab_mask_fn

        self.tokens_in_think = -1
        self.tokens_after_end = -1
        self._matched_think_end_tokens = 0
        self._thinking_match_history: List[int] = []
        # The grammar also covers output that skips reasoning; reasoning opened by an
        # opener stays bounded by the budget and strict thinking filter.
        self.reasoning_in_grammar = False
        self._opening: List[int] = []
        self._opening_text = ""
        self._generated = 0
        self._kimik3_token_bytes = None
        self._kimik3_boundary_matchers = ()
        self._kimik3_boundary_next_tokens = ()
        self._kimik3_boundary_matches = (0, 0)
        self._kimik3_thinking_history = []
        self._kimik3_json_reasoning = None
        self._kimik3_json_grammars = None

    def maybe_init_reasoning(self, reasoning: bool):
        if self._kimik3_json_grammars is not None:
            template = self._kimik3_json_grammars[bool(reasoning)]
            if (
                self.grammar.ctx is not template.ctx
                or self.current_token is not None
                or self.grammar.accepted_tokens
                or self.grammar.finished
            ):
                self.grammar = template.copy()
            self.current_token = None
        if self._kimik3_json_reasoning is not None:
            if self._kimik3_json_reasoning != bool(reasoning):
                raise ValueError(
                    "Kimi K3 JSON grammar thinking mode does not match its cache key"
                )
            if (
                self.current_token is not None
                or self.grammar.accepted_tokens
                or self.grammar.finished
            ):
                self.grammar = self.grammar.copy()
            self.current_token = None
        self._matched_think_end_tokens = 0
        self._thinking_match_history.clear()
        self._kimik3_boundary_matches = (0,) * len(self._kimik3_boundary_matchers)
        self._kimik3_thinking_history.clear()
        if reasoning:
            self.tokens_in_think = 0
            self.tokens_after_end = -1
        else:
            self.tokens_in_think = -1
            self.tokens_after_end = 0

    def enable_kimik3_required_native_boundary(self, tokenizer, stop_token_ids):
        """Keep generated required/strict-auto K3 constraints active while thinking."""
        if self.current_token is not None or self._thinking_match_history:
            raise ValueError("K3 native boundary must be configured before decoding")
        from .xgrammar_backend import XGrammarGrammar

        token_bytes = getattr(
            getattr(tokenizer, "model", None), "decode_single_token_bytes", None
        )
        if not isinstance(self.grammar, XGrammarGrammar) or not callable(token_bytes):
            raise ValueError("K3 native boundary requires XGrammar and K3 token bytes")
        self.grammar.set_stop_token_ids_before_decoding(stop_token_ids)
        self._install_kimik3_boundary_metadata(tokenizer)

    def _install_kimik3_boundary_metadata(self, tokenizer):
        """Install tokenizer-local immutable metadata, never matcher progress."""
        from sglang.srt.function_call.kimik3_format import (
            RESPONSE_OPEN,
            THINK_CLOSE,
            TOOLS_OPEN,
        )

        from .xgrammar_backend import XGrammarGrammar

        token_bytes = getattr(
            getattr(tokenizer, "model", None), "decode_single_token_bytes", None
        )
        if not isinstance(self.grammar, XGrammarGrammar) or not callable(token_bytes):
            raise ValueError("K3 native boundary requires XGrammar and K3 token bytes")
        self._kimik3_token_bytes = token_bytes
        markers = (
            (THINK_CLOSE, RESPONSE_OPEN)
            if self._kimik3_json_reasoning is not None
            or self._kimik3_json_grammars is not None
            else (THINK_CLOSE, TOOLS_OPEN, RESPONSE_OPEN)
        )
        self._kimik3_boundary_matchers = tuple(
            TokenSequenceMatcher(marker.encode()) for marker in markers
        )
        self._kimik3_boundary_matches = (0,) * len(self._kimik3_boundary_matchers)
        # Only encode the short ASCII markers' suffixes, never the vocabulary.
        # This also completes an alias-tokenized or partial tools opener when the
        # thinking budget expires, rather than restarting a canonical think end.
        self._kimik3_boundary_next_tokens = tuple(
            tuple(
                tokenizer.encode(marker[offset:], add_special_tokens=False)[0]
                for offset in range(len(marker))
            )
            for marker in markers
        )

    def set_request_think_end_ids(self, think_end_ids: Sequence[int]) -> None:
        """Select one request-specific reasoning terminator before decoding."""
        token_ids = tuple(think_end_ids)
        if not token_ids:
            raise ValueError("request reasoning terminator must not be empty")
        if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise ValueError("request reasoning terminator must contain token IDs")
        if self._decoding_started():
            raise RuntimeError(
                "request reasoning terminator must be selected before decoding"
            )

        self.think_end_ids = token_ids
        self._think_end_matcher = TokenSequenceMatcher(token_ids)
        self._matched_think_end_tokens = 0

    def set_reasoning_in_grammar(self) -> None:
        """Apply the grammar from the first token, as it also describes the
        reasoning."""
        if self._decoding_started():
            raise RuntimeError(
                "the grammar must cover the reasoning before decoding starts"
            )
        self.reasoning_in_grammar = True

    def _decoding_started(self) -> bool:
        return (
            self.current_token is not None
            or bool(self._thinking_match_history)
            or (self.tokens_in_think, self.tokens_after_end) not in ((0, -1), (-1, 0))
        )

    def _grammar_active(self) -> bool:
        return self.grammar is not None and (
            self.reasoning_in_grammar or self._is_generation()
        )

    def _reasoning_opened(self) -> bool:
        """Whether the output so far opens the reasoning it may skip."""
        text = self._opening_text
        return not self.think_starts or any(
            text.startswith(start) or start.startswith(text)
            for start in self.think_starts
        )

    def _opening_undecided(self) -> bool:
        text = self._opening_text
        return any(
            start.startswith(text) and start != text for start in self.think_starts
        )

    def _is_thinking(self):
        return self.tokens_in_think >= 0 and self.tokens_after_end == -1

    def _is_generation(self):
        return self.tokens_after_end >= 0

    @property
    def vocab_mask_is_unconstrained(self):
        # Forward generation cannot re-enter thinking. Speculative DFS may
        # still run for its state bookkeeping, but every descendant mask is
        # also an identity mask. Re-evaluate after any explicit rollback.
        return self.grammar is None and self._is_generation()

    def transfer_state(self, token: int) -> None:
        if self._kimik3_token_bytes is not None:
            if self._is_thinking():
                self._kimik3_thinking_history.append(
                    (
                        self.tokens_in_think,
                        self._matched_think_end_tokens,
                        self._kimik3_boundary_matches,
                    )
                )
                previous = self._matched_think_end_tokens
                matched = self._think_end_matcher.advance(previous, token)
                self.tokens_in_think += previous + 1 - matched
                self._matched_think_end_tokens = matched
                matches = list(self._kimik3_boundary_matches)
                for byte in self._kimik3_token_bytes(token):
                    for index, matcher in enumerate(self._kimik3_boundary_matchers):
                        matches[index] = matcher.advance(matches[index], byte)
                        if matches[index] == len(matcher):
                            self._matched_think_end_tokens = 0
                            self.tokens_after_end = 0
                            self._kimik3_boundary_matches = (0,) * len(matches)
                            return
                self._kimik3_boundary_matches = tuple(matches)
            elif self._is_generation():
                self.tokens_after_end += 1
            return
        if self._is_thinking():
            previous_match = self._matched_think_end_tokens
            self._thinking_match_history.append(previous_match)
            matched = self._think_end_matcher.advance(previous_match, token)
            if matched == len(self._think_end_matcher):
                self._matched_think_end_tokens = 0
                self.tokens_after_end = 0
            else:
                self.tokens_in_think += previous_match + 1 - matched
                self._matched_think_end_tokens = matched
        elif self._is_generation():
            self.tokens_after_end += 1

    def rollback_state(self):
        if self._kimik3_token_bytes is not None:
            if self.tokens_after_end > 0:
                self.tokens_after_end -= 1
            elif self._kimik3_thinking_history:
                (
                    self.tokens_in_think,
                    self._matched_think_end_tokens,
                    self._kimik3_boundary_matches,
                ) = self._kimik3_thinking_history.pop()
                self.tokens_after_end = -1
            return
        if self._is_thinking():
            if self._thinking_match_history:
                previous_match = self._thinking_match_history.pop()
                self.tokens_in_think -= (
                    previous_match + 1 - self._matched_think_end_tokens
                )
                self._matched_think_end_tokens = previous_match
        elif self._is_generation():
            if self.tokens_after_end == 0:
                if self._thinking_match_history:
                    self.tokens_after_end = -1
                    self._matched_think_end_tokens = self._thinking_match_history.pop()
            elif self.tokens_after_end > 0:
                self.tokens_after_end -= 1

    def accept_token(self, token: int):
        if self._kimik3_token_bytes is not None:
            if self.grammar.is_terminated():
                return
            self.grammar.accept_token(token)
            self.current_token = token
            self.transfer_state(token)
            return
        # Track the last accepted token on the wrapper itself (mirroring
        # XGrammarGrammar.accept_token). Disaggregation's process_prebuilt uses
        # `grammar.current_token is None` to detect a retracted request whose
        # token was already accepted and must not be re-accepted. Without this,
        # a ReasonerGrammarObject's current_token stays None forever (the inner
        # grammar's is updated, not the wrapper's), so the guard never fires and
        # the token is accepted twice -> "Tokens not accepted" -> FINISH_ABORT.
        self.current_token = token
        if self._grammar_active():
            self.grammar.accept_token(token)
        if self.reasoning_in_grammar:
            if (
                self.think_starts
                and self._generated == len(self._opening)
                and self._opening_undecided()
            ):
                self._opening.append(token)
                self._opening_text = self.decode(self._opening)
            self._generated += 1
        self.transfer_state(token)

    def is_terminated(self):
        if self._kimik3_token_bytes is not None:
            return self.grammar.is_terminated()
        if self._grammar_active():
            return self.grammar.is_terminated()
        return False

    def rollback(self, k):
        if self._kimik3_token_bytes is not None:
            if type(k) is not int or not 0 <= k <= len(self.grammar.accepted_tokens):
                raise ValueError("K3 native rollback requires an accepted-token count")
            if k:
                self.grammar.rollback(k)
            for _ in range(k):
                self.rollback_state()
            self.current_token = (
                self.grammar.accepted_tokens[-1]
                if self.grammar.accepted_tokens
                else None
            )
            return
        if self.grammar is not None:
            if self.reasoning_in_grammar:
                steps_after = k
            else:
                steps_after = min(k, max(0, self.tokens_after_end))
            if steps_after > 0:
                self.grammar.rollback(steps_after)
        if self.reasoning_in_grammar:
            self._generated -= k
            if len(self._opening) > self._generated:
                del self._opening[self._generated :]
                self._opening_text = self.decode(self._opening)
        for _ in range(k):
            self.rollback_state()

    def _can_think_more(self):
        return (
            self.max_think_tokens < 0
            or self.tokens_in_think + self._matched_think_end_tokens
            < self.max_think_tokens
        )

    def _must_think_more(self):
        return (
            self.min_think_tokens >= 0
            and self.tokens_in_think + self._matched_think_end_tokens
            < self.min_think_tokens
        )

    def _do_token_filter(
        self, vocab_mask, token_ids, idx, is_allowed=True, reset_vocab_mask=True
    ):
        if self.token_filter_fn is not None:
            self.token_filter_fn(
                vocab_mask,
                token_ids,
                idx,
                is_allowed,
                reset_vocab_mask=reset_vocab_mask,
            )

    def _filter_reasoning_in_grammar(self, vocab_mask: torch.Tensor, idx: int):
        # Budget and strict filters can narrow, but never widen, the grammar mask.
        if self._can_think_more():
            if self.think_excluded_token_ids is not None:
                self._do_token_filter(
                    vocab_mask,
                    self.think_excluded_token_ids,
                    idx,
                    is_allowed=False,
                    reset_vocab_mask=False,
                )
            return
        token = self.think_end_ids[self._matched_think_end_tokens]
        if (int(vocab_mask[idx, token // 32]) >> (token % 32)) & 1:
            self._do_token_filter(vocab_mask, [token], idx, is_allowed=True)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        if self._kimik3_token_bytes is not None:
            self.grammar.fill_vocab_mask(vocab_mask, idx)
            if self._is_thinking() and self.enable_token_filter:
                if self._can_think_more():
                    if self.think_excluded_token_ids:
                        self.token_filter_fn(
                            vocab_mask,
                            self.think_excluded_token_ids,
                            idx,
                            is_allowed=False,
                            reset_vocab_mask=False,
                        )
                else:
                    # An allow-list filter ORs bits when reset=False; intersect
                    # a separate allowed-only row with the native mask instead.
                    allowed = torch.zeros_like(vocab_mask[idx : idx + 1])
                    boundary = max(
                        range(len(self._kimik3_boundary_matchers)),
                        key=lambda index: self._kimik3_boundary_matches[index],
                    )
                    next_token = self._kimik3_boundary_next_tokens[boundary][
                        self._kimik3_boundary_matches[boundary]
                    ]
                    self.token_filter_fn(
                        allowed,
                        [next_token],
                        0,
                        is_allowed=True,
                    )
                    vocab_mask[idx].bitwise_and_(allowed[0])
            return
        if self.reasoning_in_grammar and self.grammar is not None:
            self.grammar.fill_vocab_mask(vocab_mask, idx)
            if (
                self._is_thinking()
                and self.enable_token_filter
                and self._reasoning_opened()
            ):
                self._filter_reasoning_in_grammar(vocab_mask, idx)
            return
        if self._is_thinking():
            if not self.enable_token_filter:
                return
            if self._must_think_more():
                blocked_token_ids = [self.think_end_ids[self._matched_think_end_tokens]]
                if self.think_excluded_token_ids is not None:
                    blocked_token_ids.extend(self.think_excluded_token_ids)
                self._do_token_filter(
                    vocab_mask,
                    blocked_token_ids,
                    idx,
                    is_allowed=False,
                )
            elif self._can_think_more():
                if self.think_excluded_token_ids is not None:
                    self._do_token_filter(
                        vocab_mask,
                        self.think_excluded_token_ids,
                        idx,
                        is_allowed=False,
                    )
            else:
                self._do_token_filter(
                    vocab_mask,
                    [self.think_end_ids[self._matched_think_end_tokens]],
                    idx,
                    is_allowed=True,
                )
            return
        if self._is_generation() and self.grammar is not None:
            self.grammar.fill_vocab_mask(vocab_mask, idx)

    def allocate_vocab_mask(self, vocab_size, batch_size, device):
        if self.grammar is not None:
            return self.grammar.allocate_vocab_mask(vocab_size, batch_size, device)
        if self.allocate_vocab_mask_fn is not None:
            return self.allocate_vocab_mask_fn(vocab_size, batch_size, device)
        return None

    def move_vocab_mask(self, vocab_mask, device):
        if self.grammar is not None:
            return self.grammar.move_vocab_mask(vocab_mask, device)
        if self.move_vocab_mask_fn is not None:
            return self.move_vocab_mask_fn(vocab_mask, device)
        return vocab_mask

    @property
    def apply_vocab_mask(self):
        if self.grammar is not None:
            return self.grammar.apply_vocab_mask
        return self.apply_vocab_mask_fn

    def copy(self):
        new_obj = ReasonerGrammarObject(
            grammar=self.grammar.copy() if self.grammar is not None else None,
            think_end_ids=self.think_end_ids,
            think_excluded_token_ids=self.think_excluded_token_ids,
            min_think_tokens=self.min_think_tokens,
            max_think_tokens=self.max_think_tokens,
            enable_token_filter=self.enable_token_filter,
            token_filter_fn=self.token_filter_fn,
            allocate_vocab_mask_fn=self.allocate_vocab_mask_fn,
            move_vocab_mask_fn=self.move_vocab_mask_fn,
            apply_vocab_mask_fn=self.apply_vocab_mask_fn,
            think_starts=self.think_starts,
            decode=self.decode,
        )
        new_obj.tokens_in_think = self.tokens_in_think
        new_obj.tokens_after_end = self.tokens_after_end
        new_obj._matched_think_end_tokens = self._matched_think_end_tokens
        new_obj._thinking_match_history = list(self._thinking_match_history)
        new_obj._finished = self._finished
        new_obj.current_token = self.current_token
        new_obj.reasoning_in_grammar = self.reasoning_in_grammar
        new_obj._opening = list(self._opening)
        new_obj._opening_text = self._opening_text
        new_obj._generated = self._generated
        new_obj._kimik3_json_reasoning = self._kimik3_json_reasoning
        new_obj._kimik3_json_grammars = self._kimik3_json_grammars
        if self._kimik3_token_bytes is not None:
            new_obj._kimik3_token_bytes = self._kimik3_token_bytes
            new_obj._kimik3_boundary_matchers = self._kimik3_boundary_matchers
            new_obj._kimik3_boundary_next_tokens = self._kimik3_boundary_next_tokens
            new_obj._kimik3_boundary_matches = self._kimik3_boundary_matches
            new_obj._kimik3_thinking_history = list(self._kimik3_thinking_history)
            # XGrammar.copy creates a fresh matcher, so restore native progress
            # for a midstream copy as well as preserving wrapper counters.
            for token in self.grammar.accepted_tokens:
                new_obj.grammar.accept_token(token)
            new_obj.grammar.finished = self.grammar.finished
        return new_obj

    def cache_size_bytes(self) -> int:
        return sys.getsizeof(self) + (
            self.grammar.cache_size_bytes() if self.grammar is not None else 0
        )

    @property
    def finished(self):
        if self.grammar is not None:
            return self.grammar.finished
        return self._finished

    @finished.setter
    def finished(self, finished):
        if self.grammar is not None:
            self.grammar.finished = finished
        else:
            self._finished = finished

    def try_jump_forward(self, tokenizer):
        if self._is_generation() and self.grammar is not None:
            return self.grammar.try_jump_forward(tokenizer)
        return None

    def jump_forward_str_state(self, helper):
        if self._is_generation() and self.grammar is not None:
            return self.grammar.jump_forward_str_state(helper)
        return None

    def jump_and_retokenize(self, old_output_ids, new_output_ids, next_state):
        if self._kimik3_token_bytes is not None:
            common = 0
            for old, new in zip(old_output_ids, new_output_ids):
                if old != new:
                    break
                common += 1
            self.rollback(len(old_output_ids) - common)
            for token in new_output_ids[common:]:
                self.accept_token(token)
            return
        if self._is_generation() and self.grammar is not None:
            return self.grammar.jump_and_retokenize(
                old_output_ids, new_output_ids, next_state
            )


class ReasonerGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        grammar_backend: BaseGrammarBackend,
        reasoning_parser: ReasoningParser,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        enable_strict_thinking: bool = False,
    ):
        super().__init__()
        self.grammar_backend = grammar_backend
        self._budget_variant_keys: OrderedDict[Tuple[str, str], None] = OrderedDict()
        from sglang.srt.parser.reasoning_parser import KimiK3Detector

        self._kimik3_json_tokenizer = (
            tokenizer if isinstance(reasoning_parser.detector, KimiK3Detector) else None
        )
        self._kimik3_json_boundary_metadata = None
        think_end_ids = reasoning_parser.detector.get_think_end_token_ids(tokenizer)
        if not think_end_ids:
            raise ValueError(
                f"think_end_token '{reasoning_parser.detector.think_end_token}' "
                f"could not be encoded by the tokenizer."
            )
        self.think_end_ids = think_end_ids
        detector = reasoning_parser.detector
        self.think_starts = ()
        if isinstance(detector, ResponseTemplateReasoningDetector):
            self.think_starts = tuple(detector.message_start_think_openers)
        self._decode = lambda token_ids: tokenizer.decode(
            token_ids, skip_special_tokens=False, spaces_between_special_tokens=False
        )
        self._enable_strict_thinking = enable_strict_thinking
        self._answer_frame = reasoning_parser.detector.get_answer_frame()
        if self._answer_frame is not None:
            self.grammar_backend.reserve_framing_tokens(
                self._answer_frame.framing_token_ids()
            )
        self.think_excluded_token_ids = self._get_think_excluded_token_ids(
            reasoning_parser, tokenizer
        )
        self.max_think_tokens = envs.SGLANG_MAX_THINK_TOKENS.get()
        self.enable_token_filter = self.enable_strict_thinking and (
            self.think_excluded_token_ids is not None or self.max_think_tokens >= 0
        )
        if (
            self.enable_token_filter
            and not self.grammar_backend.is_support_token_filter
        ):
            raise ValueError(
                "Strict reasoning format requested but the grammar backend does not "
                "support token filtering. Use a grammar backend that supports token "
                "filtering (e.g., xgrammar) or disable strict reasoning mode."
            )
        self._token_filter_fn = (
            self.grammar_backend.set_token_filter if self.enable_token_filter else None
        )

    def cache_key(self, key: tuple, require_reasoning: bool) -> tuple:
        if (
            self.use_cache_budget
            and key[0] == "json"
            and self._kimik3_json_tokenizer is not None
        ):
            return (*key[:2], bool(require_reasoning))
        return key[:2]

    def _get_think_excluded_token_ids(
        self,
        reasoning_parser: ReasoningParser,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> Optional[List[int]]:
        excluded_ids = []
        if (not self.enable_strict_thinking) or (
            not reasoning_parser.detector.think_excluded_tokens
        ):
            return None
        for token in reasoning_parser.detector.think_excluded_tokens:
            new_ids = tokenizer.encode(token, add_special_tokens=False)
            if not new_ids:
                raise ValueError(
                    f"think_excluded_token '{token}' could not be encoded by the "
                    f"tokenizer. All excluded tokens must be encodable for strict "
                    f"reasoning mode to function correctly."
                )
            excluded_ids += new_ids
        return excluded_ids

    def _make_grammar_object(
        self, grammar: Optional[BaseGrammarObject], reasoning: bool
    ) -> ReasonerGrammarObject:
        obj = ReasonerGrammarObject(
            grammar=grammar,
            think_end_ids=self.think_end_ids,
            think_excluded_token_ids=self.think_excluded_token_ids,
            max_think_tokens=self.max_think_tokens,
            enable_token_filter=self.enable_token_filter,
            token_filter_fn=self._token_filter_fn,
            allocate_vocab_mask_fn=self.grammar_backend.allocate_vocab_mask,
            move_vocab_mask_fn=self.grammar_backend.move_vocab_mask,
            apply_vocab_mask_fn=self.grammar_backend.apply_vocab_mask,
            think_starts=self.think_starts,
            decode=self._decode,
        )
        obj.maybe_init_reasoning(reasoning)
        return obj

    def init_strict_reasoning_grammar(
        self, reasoning: bool
    ) -> Optional[BaseGrammarObject]:
        """Create a grammar object for strict token filtering only (no inner grammar)."""
        if not self.enable_strict_thinking:
            return None
        return self._make_grammar_object(None, reasoning)

    def get_request_grammar_key(
        self,
        key: Tuple[str, str],
        *,
        input_ids: Sequence[int],
        thinking_budget: Optional[int],
    ) -> Tuple[str, str]:
        key_type, key_string = key
        if not self._is_framed(key_type):
            return key
        if self._answer_frame.ends_in_open_answer_block(input_ids):
            return (f"{key_type}:{_OPEN_ANSWER_VARIANT}", key_string)
        # Mirrors the strict wrapper, which applies a request budget only while filtering.
        if not self.enable_token_filter or thinking_budget is None:
            return key
        budget = max(thinking_budget, -1)
        if budget == max(self.max_think_tokens, -1):
            return key
        return (f"{key_type}:{_THINK_BUDGET_VARIANT}{budget}", key_string)

    def _is_framed(self, key_type: str) -> bool:
        return (
            self._answer_frame is not None
            and key_type in self._answer_frame.framed_key_types
        )

    def get_cached_or_future_value(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> Tuple[BaseGrammarObject | Future[BaseGrammarObject], bool]:
        if key in self._budget_variant_keys:
            self._budget_variant_keys.move_to_end(key)
        return super().get_cached_or_future_value(key, require_reasoning)

    def set_cache(
        self,
        key: Tuple[str, str],
        value: BaseGrammarObject,
        require_reasoning: Optional[bool] = None,
        *,
        future: Optional[Future] = None,
    ):
        super().set_cache(key, value, require_reasoning, future=future)
        if _THINK_BUDGET_VARIANT not in key[0]:
            return
        self._budget_variant_keys[key] = None
        self._budget_variant_keys.move_to_end(key)
        while len(self._budget_variant_keys) > _BUDGET_VARIANT_CACHE_SIZE:
            evicted, _ = self._budget_variant_keys.popitem(last=False)
            if self.use_cache_budget:
                # Budget variants are never K3 JSON keys, so the cache key
                # does not depend on require_reasoning; keep byte accounting.
                with self._cache_lock:
                    self._remove_cache_entry(self.cache_key(evicted, False))
            else:
                self.cache.pop(evicted, None)

    def reset(self):
        super().reset()
        if self.use_cache_budget:
            self.grammar_backend.reset()
        self._budget_variant_keys.clear()

    def _init_value_dispatch(
        self, key: Tuple[str, str], reasoning: bool
    ) -> Optional[BaseGrammarObject]:
        key_type, _, variant = key[0].partition(":")
        key = (key_type, key[1])
        if key[0] == "json" and self._kimik3_json_tokenizer is not None:
            return self._init_kimik3_json(key, reasoning)
        ret = self.grammar_backend._init_value_dispatch(key, reasoning)
        if ret is None or isinstance(ret, InvalidGrammarObject):
            return ret
        if key[0] == "full_assistant_ebnf":
            return ret
        if self._is_framed(key[0]):
            # The frame delimits thinking by message kind, so it owns the whole
            # turn; strict thinking only caps the thinking message.
            framed = self._frame_answer(ret, variant)
            if framed is not None:
                return framed
        if (
            not self.enable_strict_thinking
            and key[0] == "structural_tag"
            and self._answer_frame is not None
            and self._answer_frame.owns_turn(key[1])
        ):
            return ret
        return self._make_grammar_object(ret, reasoning)

    def _frame_answer(
        self, grammar: BaseGrammarObject, variant: str
    ) -> Optional[BaseGrammarObject]:
        in_answer_block = variant == _OPEN_ANSWER_VARIANT
        if variant.startswith(_THINK_BUDGET_VARIANT):
            max_think_tokens = int(variant.removeprefix(_THINK_BUDGET_VARIANT))
        elif self.enable_strict_thinking:
            max_think_tokens = self.max_think_tokens
        else:
            max_think_tokens = -1
        return self.grammar_backend.compile_framed_grammar(
            grammar,
            lambda content: self._answer_frame.structural_tag(
                content,
                in_answer_block=in_answer_block,
                max_think_tokens=max_think_tokens,
            ),
        )

    def _init_kimik3_json(self, key, reasoning):
        from .json_schema_validation import (
            UnsupportedJSONSchemaFeature,
            normalize_json_schema_resources,
            validate_xgrammar_json_schema,
        )
        from .kimik3_json_schema import json_response_format
        from .xgrammar_backend import XGrammarGrammarBackend

        backend = self.grammar_backend
        if not isinstance(backend, XGrammarGrammarBackend):
            return InvalidGrammarObject(
                "Kimi K3 native JSON constraints require XGrammar"
            )
        if not backend.any_whitespace:
            return InvalidGrammarObject(
                "Kimi K3 native JSON requires any_whitespace=True; "
                "fixed whitespace formatting is unsupported"
            )
        try:
            schema = {} if key[1] == "$$ANY$$" else json.loads(key[1])
            schema = normalize_json_schema_resources(schema)
            validate_xgrammar_json_schema(schema)
            templates = []
            modes = (bool(reasoning),) if self.use_cache_budget else (False, True)
            for thinking in modes:
                native_key = (
                    "structural_tag",
                    json.dumps(
                        json_response_format(
                            schema, thinking, backend.max_whitespace_cnt
                        )
                    ),
                )
                grammar = backend._init_value_dispatch(native_key, False)
                if grammar is None or isinstance(grammar, InvalidGrammarObject):
                    return grammar or InvalidGrammarObject(
                        "Kimi K3 JSON compilation failed",
                        status_code=500,
                        error_type="internal_server_error",
                    )
                templates.append(grammar)
        except (ValueError, UnsupportedJSONSchemaFeature) as exc:
            # Do not put schema values or generated content into an error message.
            return InvalidGrammarObject(
                f"Unable to compile Kimi K3 native JSON constraint ({type(exc).__name__})"
            )
        except RuntimeError as exc:
            return InvalidGrammarObject(
                f"Unable to compile Kimi K3 native JSON constraint ({type(exc).__name__})",
                status_code=500,
                error_type="internal_server_error",
            )

        tokenizer = self._kimik3_json_tokenizer
        stops = set(templates[0].ctx.tokenizer_info.stop_token_ids)
        if tokenizer.eos_token_id is not None:
            stops.add(tokenizer.eos_token_id)
        stops.update(getattr(tokenizer, "additional_stop_token_ids", None) or [])
        try:
            for template in templates:
                template.set_stop_token_ids_before_decoding(sorted(stops))
            grammar = (
                templates[0]
                if self.use_cache_budget
                else templates[bool(reasoning)].copy()
            )
            obj = self._make_grammar_object(grammar, reasoning)
            if self.use_cache_budget:
                obj._kimik3_json_reasoning = bool(reasoning)
            else:
                obj._kimik3_json_grammars = tuple(templates)
            metadata = self._kimik3_json_boundary_metadata
            if metadata is None:
                obj._install_kimik3_boundary_metadata(tokenizer)
                # All schemas on this backend use the same tokenizer and JSON
                # channel boundaries. Share only immutable helpers, not states.
                self._kimik3_json_boundary_metadata = (
                    obj._kimik3_token_bytes,
                    obj._kimik3_boundary_matchers,
                    obj._kimik3_boundary_next_tokens,
                )
            else:
                (
                    obj._kimik3_token_bytes,
                    obj._kimik3_boundary_matchers,
                    obj._kimik3_boundary_next_tokens,
                ) = metadata
            obj.maybe_init_reasoning(reasoning)
        except (ValueError, RuntimeError) as exc:
            return InvalidGrammarObject(
                f"Unable to initialize Kimi K3 native JSON constraint ({type(exc).__name__})",
                status_code=500,
                error_type="internal_server_error",
            )
        return obj
