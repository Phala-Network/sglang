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
"""Constrained decoding with xgrammar backend."""

import dataclasses
import json
import logging
from typing import List, Optional, Tuple, Union

import torch
from xgrammar import (
    CompiledGrammar,
    GrammarCompiler,
    GrammarMatcher,
    StructuralTag,
    StructuralTagItem,
    TokenizerInfo,
    bitmask_dtype,
    get_bitmask_shape,
)

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    GrammarStats,
    InvalidGrammarObject,
)
from sglang.srt.constrained.utils import is_legacy_structural_tag
from sglang.srt.constrained.xgrammar_schema import (
    normalize_xgrammar_schema_noops,
    sanitize_xgrammar_structural_format,
    sanitize_xgrammar_structural_tag_structures,
    validate_xgrammar_schema,
    validate_xgrammar_whitespace_limit,
)
from sglang.srt.utils import is_hip
from sglang.srt.utils.common import is_pin_memory_available

_is_hip = is_hip()

if _is_hip:
    from sgl_kernel import apply_token_bitmask_inplace_cuda
else:
    from sglang.kernels.ops.grammar.bitmask_ops import (
        apply_token_bitmask_inplace_triton,
    )

from sglang.kernels.ops.grammar.token_filter_ops import set_token_filter_triton
from sglang.srt.constrained.torch_ops.token_filter_torch_ops import (
    set_token_filter_torch,
)

logger = logging.getLogger(__name__)
MAX_ROLLBACK_TOKENS = 200


def _allocate_token_bitmask(vocab_size: int, batch_size: int) -> torch.Tensor:
    # Pin where pinning exists, so the later H2D can be a genuine non_blocking
    # copy (a pageable source silently downgrades it).  MPS torch has no
    # pin-memory kernel and asserts on pin_memory=True.
    return torch.full(
        get_bitmask_shape(batch_size, vocab_size),
        -1,
        dtype=bitmask_dtype,
        pin_memory=is_pin_memory_available(),
    )


class XGrammarGrammar(BaseGrammarObject):
    def __init__(
        self,
        matcher: GrammarMatcher,
        vocab_size: int,
        ctx: CompiledGrammar,
        override_stop_tokens: Optional[Union[List[int], int]],
        key_string: Optional[str] = None,
        grammar_stats: Optional[GrammarStats] = GrammarStats(),
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.override_stop_tokens = override_stop_tokens
        self.accepted_tokens = []
        self.key_string = key_string
        self.grammar_stats = grammar_stats

    def accept_token(self, token: int):
        if not self.is_terminated():
            self.current_token = token
            accepted = self.matcher.accept_token(token)
            if not accepted:
                # log for debugging
                raise ValueError(
                    f"Tokens not accepted: {token}\n"
                    f"Accepted tokens: {self.accepted_tokens}\n"
                    f"Key string: {self.key_string}"
                )
            else:
                self.accepted_tokens.append(token)

    def rollback(self, k: int):
        self.matcher.rollback(k)
        self.accepted_tokens = self.accepted_tokens[:-k]

    def is_terminated(self):
        return self.matcher.is_terminated()

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return _allocate_token_bitmask(vocab_size, batch_size)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(vocab_mask, idx)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    def apply_vocab_mask(self, logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if logits.device.type in {"cuda", "xpu", "musa"}:
            if _is_hip:
                apply_token_bitmask_inplace_cuda(logits, vocab_mask)
            else:
                apply_token_bitmask_inplace_triton(logits, vocab_mask)
        elif logits.device.type == "npu":
            import sgl_kernel_npu  # noqa: F401

            torch.ops.npu.apply_token_bitmask(logits, vocab_mask)
        elif logits.device.type == "cpu":
            # Used by the MLX backend, which builds its additive mask rows
            # on the CPU before inserting them into the lazy graph.
            from xgrammar import apply_token_bitmask_inplace

            apply_token_bitmask_inplace(logits, vocab_mask, backend="cpu")
        else:
            raise RuntimeError(f"Unsupported device: {logits.device.type}")

    def copy(self):
        matcher = GrammarMatcher(
            self.ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            override_stop_tokens=self.override_stop_tokens,
        )
        if grammar_stats := self.grammar_stats:
            grammar_stats = dataclasses.replace(
                grammar_stats, is_cache_hit=True, tree_traversal_time=[]
            )
        return XGrammarGrammar(
            matcher,
            self.vocab_size,
            self.ctx,
            self.override_stop_tokens,
            self.key_string,
            grammar_stats,
        )

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        s = self.matcher.find_jump_forward_string()
        if s:
            return [], s
        return None

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        _, data = helper
        return data, -1

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        k = 0
        for i, old_id in enumerate(old_output_ids):
            if old_id == new_output_ids[i]:
                k = i + 1
            else:
                break

        # rollback to the last token that is the same
        if k < len(old_output_ids):
            self.matcher.rollback(len(old_output_ids) - k)

        for i in range(k, len(new_output_ids)):
            if not self.matcher.accept_token(new_output_ids[i]):
                raise ValueError(
                    f"Token not accepted during retokenization: {new_output_ids[i]} "
                    f"at position {i}\n"
                    f"Old output IDs: {old_output_ids}\n"
                    f"New output IDs: {new_output_ids}\n"
                    f"Key string: {self.key_string}"
                )

    def __repr__(self):
        return f"XGrammarGrammar({self.key_string=}, {self.accepted_tokens=}, {self.current_token=})"

    def with_response_suffix(self, suffix_ids):
        return XGrammarResponseSuffixGrammar(
            self.matcher,
            self.vocab_size,
            self.ctx,
            self.override_stop_tokens,
            self.key_string,
            self.grammar_stats,
            suffix_ids=suffix_ids,
        )


class XGrammarResponseSuffixGrammar(XGrammarGrammar):
    """Allow an exact native response trailer only after a complete JSON value.

    Its final token is the model EOS. The native matcher must accept that EOS
    before the trailer can begin, and receives it only when the trailer ends.
    """

    def __init__(self, *args, suffix_ids, **kwargs):
        super().__init__(*args, **kwargs)
        self.suffix_ids = tuple(suffix_ids)
        if len(self.suffix_ids) < 2:
            raise ValueError("response suffix must contain a trailer and EOS")
        self.suffix_position = 0

    def accept_token(self, token):
        if self.suffix_position or token == self.suffix_ids[0]:
            if self.suffix_position >= len(self.suffix_ids):
                raise ValueError("response suffix already complete")
            if token != self.suffix_ids[self.suffix_position]:
                raise ValueError("invalid response suffix token")
            if self.suffix_position == 0:
                # Inspect native completion without mutating rollback history.
                mask = _allocate_token_bitmask(self.vocab_size, 1)
                self.matcher.fill_next_token_bitmask(mask, 0)
                eos = self.suffix_ids[-1]
                if not (int(mask[0, eos // 32]) >> (eos % 32)) & 1:
                    # The same token can be literal data inside a JSON string.
                    # Preserve native validation without entering trailer state.
                    super().accept_token(token)
                    return
            if self.suffix_position == len(self.suffix_ids) - 1:
                super().accept_token(token)
            else:
                # accepted_tokens tracks only tokens fed to the native matcher;
                # suffix_position owns the protocol trailer for rollback.
                self.current_token = token
            self.suffix_position += 1
            return
        super().accept_token(token)

    def fill_vocab_mask(self, vocab_mask, idx):
        def allow(token):
            bit = token % 32
            vocab_mask[idx, token // 32] |= -(1 << 31) if bit == 31 else 1 << bit

        if self.suffix_position:
            vocab_mask[idx].zero_()
            if self.suffix_position < len(self.suffix_ids):
                token = self.suffix_ids[self.suffix_position]
                allow(token)
            return
        super().fill_vocab_mask(vocab_mask, idx)
        eos = self.suffix_ids[-1]
        if (int(vocab_mask[idx, eos // 32]) >> (eos % 32)) & 1:
            token = self.suffix_ids[0]
            allow(token)

    def rollback(self, k):
        trailer_steps = min(k, self.suffix_position)
        if trailer_steps and self.suffix_position == len(self.suffix_ids):
            super().rollback(1)  # Only the final EOS entered the native matcher.
        self.suffix_position -= trailer_steps
        if k > trailer_steps:
            super().rollback(k - trailer_steps)

    def copy(self):
        return super().copy().with_response_suffix(self.suffix_ids)

    def try_jump_forward(self, tokenizer):
        # Trailer tokens are owned by this wrapper, not the native matcher.
        return None


class TokenizerNotSupportedError(Exception):
    """Raised when tokenizer is not supported by XGrammar backend."""

    pass


class XGrammarGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
        model_eos_token_ids: Optional[List[int]] = None,
        any_whitespace: bool = True,
        max_whitespace_cnt: Optional[int] = None,
    ):
        super().__init__()
        validate_xgrammar_whitespace_limit(
            max_whitespace_cnt, any_whitespace=any_whitespace
        )

        if hasattr(tokenizer, "init_xgrammar"):
            # For special tokenizer
            tokenizer_info, override_stop_tokens = tokenizer.init_xgrammar()

            if tokenizer_info is None:
                # Not supported tokenizer
                raise TokenizerNotSupportedError(
                    f"Tokenizer type {type(tokenizer).__name__} is not supported by XGrammar"
                )
        else:
            # Create TokenizerInfo with model's EOS tokens as the authoritative stop tokens
            # This ensures consistency between what the model considers EOS and what XGrammar uses
            try:
                tokenizer_info = TokenizerInfo.from_huggingface(
                    tokenizer, vocab_size=vocab_size, stop_token_ids=model_eos_token_ids
                )
                override_stop_tokens = None
            except Exception as e:
                raise TokenizerNotSupportedError(
                    f"Failed to create XGrammar TokenizerInfo from tokenizer: {e}"
                )

        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)
        self.vocab_size = vocab_size
        self.override_stop_tokens = override_stop_tokens
        self.any_whitespace = any_whitespace
        self.max_whitespace_cnt = max_whitespace_cnt

    @property
    def is_support_token_filter(self):
        return True

    @staticmethod
    def allocate_vocab_mask(vocab_size: int, batch_size: int, device) -> torch.Tensor:
        return _allocate_token_bitmask(vocab_size, batch_size)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if logits.device.type in {"cuda", "npu", "xpu", "musa"}:
            if _is_hip:
                apply_token_bitmask_inplace_cuda(logits, vocab_mask)
            else:
                apply_token_bitmask_inplace_triton(logits, vocab_mask)
        else:
            raise RuntimeError(f"Unsupported device: {logits.device.type}")

    @staticmethod
    def set_token_filter(
        vocab_mask: torch.Tensor,
        token_ids: List[int],
        batch_idx: int,
        is_allowed: bool = True,
        reset_vocab_mask: bool = True,
    ):
        if _is_hip or (vocab_mask.device.type != "cuda"):
            set_token_filter_torch(
                vocab_mask,
                token_ids,
                batch_idx,
                is_allowed=is_allowed,
                reset_vocab_mask=reset_vocab_mask,
            )
        else:
            set_token_filter_triton(
                vocab_mask,
                token_ids,
                batch_idx,
                is_allowed=is_allowed,
                reset_vocab_mask=reset_vocab_mask,
            )

    _sanitize_structural_format = staticmethod(sanitize_xgrammar_structural_format)
    _sanitize_structural_tag_structures = staticmethod(
        sanitize_xgrammar_structural_tag_structures
    )

    def _from_context(
        self, ctx: CompiledGrammar, key_string: str, grammar_stats: GrammarStats
    ) -> XGrammarGrammar:
        matcher = GrammarMatcher(
            ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            override_stop_tokens=self.override_stop_tokens,
        )
        return XGrammarGrammar(
            matcher,
            self.vocab_size,
            ctx,
            self.override_stop_tokens,
            key_string,
            grammar_stats,
        )

    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        try:
            if key_string == "$$ANY$$" and self.max_whitespace_cnt is None:
                # Note: This builtin JSON grammar includes *all* valid JSON (including, for example, arrays at the root)
                ctx = self.grammar_compiler.compile_builtin_json_grammar()
            else:
                schema = {} if key_string == "$$ANY$$" else json.loads(key_string)
                validate_xgrammar_schema(schema)
                schema = normalize_xgrammar_schema_noops(schema)
                ctx = self.grammar_compiler.compile_json_schema(
                    schema=schema,
                    any_whitespace=self.any_whitespace,
                    max_whitespace_cnt=self.max_whitespace_cnt,
                )

        except (RuntimeError, json.decoder.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error("Hit invalid json_schema: key_string=<redacted>, e=<redacted>")
            return InvalidGrammarObject(str(e))
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="json"))

    def dispatch_ebnf(self, key_string: str) -> BaseGrammarObject:
        try:
            ctx = self.grammar_compiler.compile_grammar(key_string)
        except RuntimeError as e:
            logger.error("Hit invalid ebnf: key_string=<redacted>, e=<redacted>")
            return InvalidGrammarObject(str(e))
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="ebnf"))

    def dispatch_regex(self, key_string: str) -> BaseGrammarObject:
        try:
            ctx = self.grammar_compiler.compile_regex(key_string)
        except RuntimeError as e:
            logger.error("Hit invalid regex: key_string=<redacted>, e=<redacted>")
            return InvalidGrammarObject(str(e))
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="regex"))

    def dispatch_structural_tag(self, key_string: str) -> BaseGrammarObject:
        try:
            # TODO(dark): it's REALLY stupid to construct object from string and decode it again
            structural_tag = json.loads(key_string)
            if is_legacy_structural_tag(structural_tag):
                self._sanitize_structural_tag_structures(structural_tag)
                tags = [
                    StructuralTagItem(
                        begin=structure["begin"],
                        schema=json.dumps(structure["schema"]),
                        end=structure["end"],
                    )
                    for structure in structural_tag["structures"]
                ]
                new_tag = StructuralTag.from_legacy_structural_tag(
                    tags, structural_tag["triggers"]
                )
                new_tag.format.at_least_one = structural_tag.get("at_least_one", False)
                ctx = self.grammar_compiler.compile_structural_tag(new_tag)
            else:
                format_dict = structural_tag.get("format")
                if isinstance(format_dict, dict):
                    self._sanitize_structural_format(format_dict)
                    structural_tag["format"] = format_dict
                    key_string = json.dumps(structural_tag)
                ctx = self.grammar_compiler.compile_structural_tag(key_string)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(
                "Hit invalid structural_tag: key_string=<redacted>, e=<redacted>"
            )
            return InvalidGrammarObject(str(e))
        return self._from_context(
            ctx, key_string, GrammarStats(dispatch_type="structural_tag")
        )

    def reset(self):
        super().reset()
        self.grammar_compiler.clear_cache()


def demo_test():
    from transformers import AutoConfig, AutoTokenizer

    from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME_FOR_TEST)
    hf_config = AutoConfig.from_pretrained(DEFAULT_MODEL_NAME_FOR_TEST)

    # Should use vocab size from model config
    vocab_size = hf_config.vocab_size
    eos_token_id = tokenizer.eos_token_id

    backend = XGrammarGrammarBackend(
        tokenizer, vocab_size=vocab_size, model_eos_token_ids=[eos_token_id]
    )
    regex = r"hello (world|there)"
    grammar = backend.dispatch_regex(regex)
    tokens = [
        tokenizer.encode(t, add_special_tokens=False)[0] for t in ["hello", " world"]
    ]

    # Test termination
    grammar.accept_token(tokens[0])  # accept "hello"
    grammar.accept_token(tokens[1])  # accept " world"
    grammar.accept_token(eos_token_id)  # accept EOS
    assert grammar.is_terminated()

    # Test rollback the terminated state
    grammar.rollback(1)
    assert not grammar.is_terminated()


if __name__ == "__main__":
    demo_test()
