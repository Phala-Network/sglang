"""The Nemotron model must see the same JSON contract as constrained decoding."""

import copy

import pytest

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import (
    nemotron_response_format_template_kwargs,
)


def request(**overrides):
    body = {
        "model": "nvidia/nemotron-3.5-lightning",
        "messages": [
            {"role": "system", "content": "Preserve this instruction."},
            {"role": "user", "content": "Extract the record."},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "record",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"value": {"const": "<think>literal</think>"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        },
    }
    body.update(overrides)
    return ChatCompletionRequest.model_validate(body)


def test_actual_schema_uses_json_alias_and_does_not_mutate_request():
    value = request(
        chat_template_kwargs={
            "response_format": {"type": "text"},
            "enable_thinking": False,
        }
    )
    before = copy.deepcopy(value.model_dump())
    kwargs = dict(value.chat_template_kwargs)
    kwargs.update(nemotron_response_format_template_kwargs(value, "nemotron_3"))
    assert kwargs["response_format"] == value.response_format.model_dump(
        by_alias=True, exclude_unset=True
    )
    assert "schema" in kwargs["response_format"]["json_schema"]
    assert "schema_" not in kwargs["response_format"]["json_schema"]
    assert kwargs["enable_thinking"] is False
    assert value.model_dump() == before


def test_json_object_is_visible_without_inventing_a_schema():
    value = request(response_format={"type": "json_object"})
    assert nemotron_response_format_template_kwargs(value, "nemotron_3") == {
        "response_format": {"type": "json_object"}
    }


@pytest.mark.parametrize("parser", [None, "muse", "qwen3", "deepseek-r1"])
def test_other_models_are_unchanged(parser):
    assert nemotron_response_format_template_kwargs(request(), parser) == {}


@pytest.mark.parametrize(
    "overrides",
    [
        {"response_format": None},
        {"response_format": {"type": "text"}},
        {"input_ids": [1, 2, 3]},
        {"continue_final_message": True},
    ],
)
def test_non_json_and_preencoded_or_continued_requests_are_unchanged(overrides):
    assert (
        nemotron_response_format_template_kwargs(request(**overrides), "nemotron_3")
        == {}
    )
