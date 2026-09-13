"""Request-scoped budgets must activate filtering without global strict mode."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarBackend
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarBackend
from sglang.srt.constrained.torch_ops.token_filter_torch_ops import (
    set_token_filter_torch,
)


def make_wrapper(support=True):
    backend = MagicMock(spec=BaseGrammarBackend)
    backend.is_support_token_filter = support
    backend.set_token_filter = set_token_filter_torch
    backend.allocate_vocab_mask = MagicMock()
    backend.move_vocab_mask = MagicMock()
    backend.apply_vocab_mask = MagicMock()
    parser = SimpleNamespace(
        detector=SimpleNamespace(think_end_token="</think>", think_excluded_tokens=[])
    )
    tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [7, 8])
    reasoner = ReasonerGrammarBackend(
        backend, parser, tokenizer, enable_strict_thinking=False
    )
    return reasoner._make_grammar_object(MagicMock(), True)


def apply_budget(wrapper, params):
    request = SimpleNamespace(
        grammar=wrapper,
        sampling_params=SimpleNamespace(custom_params=params),
        set_finish_with_abort=MagicMock(),
    )
    manager = GrammarManager.__new__(GrammarManager)
    manager._apply_request_reasoning_budget(request)
    return request


def allowed(mask):
    return [i for i in range(32) if int(mask[0, 0]) & (1 << i)]


def test_request_budget_forces_complete_end_sequence_and_returns_to_grammar():
    wrapper = make_wrapper()
    cached = wrapper.copy()
    request = apply_budget(wrapper, {"thinking_budget": 1})
    request.set_finish_with_abort.assert_not_called()
    assert wrapper.enable_token_filter
    assert not cached.enable_token_filter
    wrapper.accept_token(10)
    mask = torch.full((1, 2), -1, dtype=torch.int32)
    wrapper.fill_vocab_mask(mask, 0)
    assert allowed(mask) == [7]
    wrapper.accept_token(7)
    wrapper.fill_vocab_mask(mask, 0)
    assert allowed(mask) == [8]
    wrapper.accept_token(8)
    wrapper.fill_vocab_mask(mask, 0)
    wrapper.grammar.fill_vocab_mask.assert_called_once_with(mask, 0)
    wrapper.rollback(2)
    wrapper.fill_vocab_mask(mask, 0)
    assert allowed(mask) == [7]


@pytest.mark.parametrize("params", [None, {}, {"thinking_budget": -1}])
def test_unbudgeted_non_strict_requests_are_unchanged(params):
    wrapper = make_wrapper()
    apply_budget(wrapper, params)
    assert not wrapper.enable_token_filter
    mask = torch.full((1, 2), -1, dtype=torch.int32)
    wrapper.fill_vocab_mask(mask, 0)
    assert allowed(mask) == list(range(32))


def test_unsupported_filter_is_explicitly_rejected():
    wrapper = make_wrapper(support=False)
    request = apply_budget(wrapper, {"thinking_budget": 8})
    request.set_finish_with_abort.assert_called_once()


def test_zero_budget_still_emits_end_marker_and_does_not_change_context():
    wrapper = make_wrapper()
    apply_budget(wrapper, {"thinking_budget": 0})
    mask = torch.full((1, 2), -1, dtype=torch.int32)
    wrapper.fill_vocab_mask(mask, 0)
    assert allowed(mask) == [7]
