"""Request visibility and Qwen bounds, without starting an inference engine."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager


def request(**kwargs):
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "2+2?"}], **kwargs
    )


def qwen_chat():
    chat = object.__new__(OpenAIServingChat)
    chat.tokenizer_manager = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5"))
    )
    return chat


def test_visibility_is_separate_from_generation_and_nested_policy_wins():
    hidden = request(include_reasoning=False)
    assert hidden.reasoning_exclude is True
    assert hidden.chat_template_kwargs is None
    enabled = request(include_reasoning=True, reasoning={"enabled": False})
    assert enabled.chat_template_kwargs["enable_thinking"] is False
    shown = request(include_reasoning=False, reasoning={"exclude": False})
    assert shown.reasoning_exclude is False
    disabled = request(reasoning_effort="none")
    assert disabled.chat_template_kwargs["enable_thinking"] is False
    assert disabled.reasoning_exclude is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"include_reasoning": "false"},
        {"reasoning": {"exclude": "false"}},
        {"reasoning": {"max_tokens": True}},
        {"reasoning": {"max_tokens": 0}},
        {"reasoning": {"max_tokens": 3.5}},
    ],
)
def test_invalid_reasoning_controls_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        request(**kwargs)


def test_reasoning_budget_is_internal_and_typed():
    req = request(reasoning={"max_tokens": 128, "exclude": True})
    assert req.reasoning_max_tokens == 128
    assert "reasoning_max_tokens" not in req.model_dump()
    assert "reasoning_exclude" not in req.model_dump()


def test_qwen_effort_bounds_keep_tiers_aliases_and_answer_space():
    chat = qwen_chat()
    with patch(
        "sglang.srt.entrypoints.openai.serving_chat.get_serving",
        return_value=SimpleNamespace(enable_strict_thinking=True),
    ):
        low, medium, high = [
            chat._qwen35_reasoning_effort_token_range(effort, 2000)
            for effort in ("low", "medium", "xhigh")
        ]
        assert 0 < low[0] < medium[0] < high[0]
        assert low[1] < medium[1] < high[1] < 2000
        assert chat._qwen35_reasoning_effort_token_range("high", 2000) == high
        assert chat._qwen35_reasoning_effort_token_range("minimal", 2000) == low
        assert chat._qwen35_reasoning_effort_token_range("none", 2000) is None
        assert chat._qwen35_reasoning_effort_token_range(None, None)[1] == 8192
        assert chat._qwen35_reasoning_effort_token_range("xhigh", 2000, 100)[1] == 100
        chat.tokenizer_manager.model_config.hf_config.model_type = "another_model"
        assert chat._qwen35_reasoning_effort_token_range("low", 2000) is None


def test_qwen_system_fold_preserves_tool_history_and_reasoning_input():
    chat = qwen_chat()
    messages = [
        {"role": "system", "content": "first"},
        {"role": "user", "content": "question"},
        {"role": "system", "content": "second"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "earlier",
            "tool_calls": [{}],
        },
        {"role": "tool", "content": "result"},
    ]
    folded = chat._fold_qwen35_system_messages(messages)
    assert folded[0]["content"] == "first\n\nsecond"
    assert messages[0]["content"] == "first"
    chat._expose_qwen35_reasoning_tool_history(folded)
    assert folded[-2]["content"] == "Prior reasoning:\nearlier"
    assert folded[-1] == {"role": "tool", "content": "result"}


def test_minimum_and_maximum_bounds_survive_copy_and_multitoken_end():
    calls = []

    def record(mask, ids, idx, allowed, *, reset_vocab_mask=True):
        calls.append((list(ids), allowed, reset_vocab_mask))

    grammar = ReasonerGrammarObject(
        None,
        [7, 8],
        min_think_tokens=3,
        max_think_tokens=5,
        enable_token_filter=True,
        token_filter_fn=record,
    )
    grammar.maybe_init_reasoning(True)
    grammar.fill_vocab_mask(torch.zeros((1, 1), dtype=torch.int32), 0)
    assert calls[-1] == ([7], False, True)
    for _ in range(5):
        grammar.accept_token(1)
    copied = grammar.copy()
    assert (copied.min_think_tokens, copied.max_think_tokens) == (3, 5)
    copied.fill_vocab_mask(torch.zeros((1, 1), dtype=torch.int32), 0)
    assert calls[-1] == ([7], True, True)
    copied.accept_token(7)
    copied.fill_vocab_mask(torch.zeros((1, 1), dtype=torch.int32), 0)
    assert calls[-1] == ([8], True, True)
    assert grammar._matched_think_end_tokens == 0


def test_minimum_filter_does_not_reset_an_existing_grammar_mask():
    calls = []
    inner = SimpleNamespace(fill_vocab_mask=lambda mask, idx: None)
    grammar = ReasonerGrammarObject(
        inner,
        [7],
        min_think_tokens=3,
        enable_token_filter=True,
        token_filter_fn=lambda *args, **kw: calls.append(kw),
    )
    grammar.maybe_init_reasoning(True)
    grammar.set_reasoning_in_grammar()
    grammar.fill_vocab_mask(torch.zeros((1, 1), dtype=torch.int32), 0)
    assert calls == [{"reset_vocab_mask": False}]


def test_manager_installs_request_bounds_and_batch_slice_keeps_them():
    manager = object.__new__(GrammarManager)
    manager.scheduler = SimpleNamespace(model_config=SimpleNamespace())
    grammar = ReasonerGrammarObject(None, [7])
    req = SimpleNamespace(
        grammar=grammar,
        sampling_params=SimpleNamespace(
            custom_params={"min_thinking_budget": 3, "thinking_budget": 7}
        ),
    )
    manager._apply_request_reasoning_config(req)
    assert (grammar.min_think_tokens, grammar.max_think_tokens) == (3, 7)
    batch = GenerateReqInput(
        input_ids=[[1], [2]], min_thinking_tokens=3, max_thinking_tokens=7
    )
    batch.normalize_batch_and_arguments()
    assert (batch[1].min_thinking_tokens, batch[1].max_thinking_tokens) == (3, 7)


@pytest.mark.parametrize(
    "strict, minimum, maximum, rejected",
    [
        (False, 1, 3, True),
        (True, -1, 3, True),
        (True, True, 3, True),
        (True, 4, 3, True),
        (True, 1, 3, False),
    ],
)
def test_tokenizer_validates_thinking_bounds_before_request_registration(
    strict, minimum, maximum, rejected
):
    class Registered(Exception):
        pass

    manager = object.__new__(TokenizerManager)
    manager.auto_create_handle_loop = lambda: None
    manager._set_default_priority = lambda obj: None

    def registered(*args):
        raise Registered

    manager._init_req_state = registered
    obj = GenerateReqInput(
        input_ids=[1], min_thinking_tokens=minimum, max_thinking_tokens=maximum
    )
    with patch(
        "sglang.srt.managers.tokenizer_manager.get_serving",
        return_value=SimpleNamespace(enable_strict_thinking=strict),
    ):
        with pytest.raises(ValueError if rejected else Registered):
            asyncio.run(manager.generate_request(obj).__anext__())


def test_custom_params_cannot_disable_validated_strict_thinking_bounds():
    manager = object.__new__(GrammarManager)
    manager._enable_strict_thinking = True
    manager.grammar_queue = []
    manager.scheduler = SimpleNamespace(model_config=SimpleNamespace())
    grammar = ReasonerGrammarObject(None, [7])
    manager.grammar_backend = SimpleNamespace(
        init_strict_reasoning_grammar=lambda required: grammar
    )
    req = SimpleNamespace(
        grammar=None,
        require_reasoning=True,
        sampling_params=SimpleNamespace(
            json_schema=None,
            regex=None,
            ebnf=None,
            structural_tag=None,
            custom_params={
                "disable_strict_thinking_grammar": True,
                "min_thinking_budget": 3,
                "thinking_budget": 7,
            },
        ),
    )
    assert manager.process_req_with_grammar(req) is False
    assert req.grammar is grammar
    assert (grammar.min_think_tokens, grammar.max_think_tokens) == (3, 7)


def test_legacy_custom_flag_cannot_bypass_strict_thinking_bounds():
    manager = object.__new__(GrammarManager)
    manager.scheduler = SimpleNamespace(model_config=SimpleNamespace())
    manager._enable_strict_thinking = True
    manager.grammar_queue = []
    grammar = ReasonerGrammarObject(None, [7])
    called = []
    manager.grammar_backend = SimpleNamespace(
        init_strict_reasoning_grammar=lambda reasoning: (
            called.append(reasoning),
            grammar,
        )[1]
    )
    req = SimpleNamespace(
        require_reasoning=True,
        sampling_params=SimpleNamespace(
            json_schema=None,
            regex=None,
            ebnf=None,
            structural_tag=None,
            custom_params={
                "min_thinking_budget": 3,
                "thinking_budget": 7,
                "disable_strict_thinking_grammar": True,
            },
        ),
    )
    assert manager.process_req_with_grammar(req) is False
    assert called == [True]
    assert req.grammar is grammar
    assert (grammar.min_think_tokens, grammar.max_think_tokens) == (3, 7)
