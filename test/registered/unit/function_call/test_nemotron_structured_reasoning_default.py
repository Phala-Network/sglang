"""Nemotron JSON defaults must not consume the final-answer budget implicitly."""

import pytest

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import (
    apply_nemotron_structured_output_reasoning_budget,
)


def request(**overrides):
    body = {
        "model": "nvidia/nemotron-3.5-lightning",
        "messages": [{"role": "user", "content": "Return JSON."}],
        "response_format": {"type": "json_object"},
        "max_tokens": 128,
    }
    body.update(overrides)
    return ChatCompletionRequest.model_validate(body)


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "json_object"},
        {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {"type": "object"},
            },
        },
    ],
)
def test_default_json_reserves_final_budget_without_disabling_thinking(response_format):
    value = request(response_format=response_format)
    apply_nemotron_structured_output_reasoning_budget(value, "nemotron_3")
    assert value.chat_template_kwargs is None
    assert value.custom_params == {"thinking_budget": 0}
    assert value.max_tokens == 128
    assert value.to_sampling_params([], {})["max_new_tokens"] == 128


@pytest.mark.parametrize(
    "controls",
    [
        {"reasoning": {"enabled": False}},
        {"reasoning_effort": "none"},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"chat_template_kwargs": {"thinking": False}},
        {"custom_params": {"thinking_budget": 256}},
        {"input_ids": [1, 2, 3]},
        {"continue_final_message": True},
    ],
)
def test_off_explicit_budget_and_continuation_are_preserved(controls):
    value = request(**controls)
    before = value.model_dump()
    apply_nemotron_structured_output_reasoning_budget(value, "nemotron_3")
    assert value.model_dump() == before


@pytest.mark.parametrize(
    "controls",
    [
        {"reasoning": {"enabled": True}},
        {"reasoning": {"effort": "high"}},
        {"reasoning_effort": "low"},
        {"include_reasoning": True},
        {"chat_template_kwargs": {"enable_thinking": True}},
        {"chat_template_kwargs": {"thinking": True}},
        {"chat_template_kwargs": {"reasoning_effort": "high"}},
    ],
)
def test_enabled_reasoning_keeps_its_controls_and_reserves_final_space(controls):
    value = request(max_tokens=8192, **controls)
    before = value.model_dump()
    apply_nemotron_structured_output_reasoning_budget(value, "nemotron_3")
    assert value.custom_params == {"thinking_budget": 4096}
    after = value.model_dump()
    after["custom_params"] = before["custom_params"]
    assert after == before


@pytest.mark.parametrize(
    "choice", ["auto", "required", {"type": "function", "function": {"name": "record"}}]
)
def test_tools_also_reserve_final_space_without_changing_choice(choice):
    value = request(
        response_format=None,
        max_tokens=4096,
        reasoning={"enabled": True},
        tools=[
            {
                "type": "function",
                "function": {"name": "record", "parameters": {"type": "object"}},
            }
        ],
        tool_choice=choice,
    )
    before = value.model_dump()
    apply_nemotron_structured_output_reasoning_budget(value, "nemotron_3")
    assert value.custom_params == {"thinking_budget": 2048}
    after = value.model_dump()
    after["custom_params"] = before["custom_params"]
    assert after == before


@pytest.mark.parametrize("parser", [None, "muse", "qwen3", "deepseek-r1"])
def test_other_models_are_unchanged(parser):
    value = request()
    before = value.model_dump()
    apply_nemotron_structured_output_reasoning_budget(value, parser)
    assert value.model_dump() == before


@pytest.mark.parametrize("format", [None, {"type": "text"}])
def test_ordinary_chat_is_unchanged(format):
    value = request(response_format=format)
    apply_nemotron_structured_output_reasoning_budget(value, "nemotron_3")
    assert value.chat_template_kwargs is None
    assert value.custom_params is None


@pytest.mark.parametrize(
    "total,expected",
    [(64, 0), (128, 0), (512, 256), (2048, 1024), (8192, 4096), (16384, 12288)],
)
def test_budget_scales_without_changing_total(total, expected):
    value = request(max_tokens=total, custom_params={"other": "retained"})
    apply_nemotron_structured_output_reasoning_budget(value, "nemotron_3")
    assert value.custom_params == {"other": "retained", "thinking_budget": expected}
    assert value.max_tokens == total
    assert value.chat_template_kwargs is None


def test_max_completion_tokens_takes_precedence_and_unspecified_is_unchanged():
    value = request(max_tokens=8192, max_completion_tokens=512)
    apply_nemotron_structured_output_reasoning_budget(value, "nemotron_3")
    assert value.custom_params == {"thinking_budget": 256}
    value = request(max_tokens=None)
    before = value.model_dump()
    apply_nemotron_structured_output_reasoning_budget(value, "nemotron_3")
    assert value.model_dump() == before
