"""Synthetic Chat HTTP entrypoint: a JSON constraint must own its output.

Prompt rendering and engine generation are isolated CPU boundary doubles. The
request's constraint selection, reasoning/tool parsers and JSON/SSE response
builders are real. No model weights, sockets or native compilation are used.
"""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from sglang.srt.entrypoints.http_server import app
from sglang.srt.entrypoints.openai import serving_chat as module
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)


@pytest.fixture(autouse=True, params=[0, 2])
def strict_level(request, monkeypatch):
    monkeypatch.setenv("SGLANG_TOOL_STRICT_LEVEL", str(request.param))


def _request(
    native, *, stream, thinking, response_format=None, choice="auto",
    reasoning_parser="kimi_k3", encoding_spec="kimi_k3",
):
    state = {"generated": 0, "closed": False}
    serving = object.__new__(module.OpenAIServingChat)
    # Attributes __init__ sets on the fork's main line.
    serving._encoding_adapter = None
    serving.is_minimax_m3 = False
    serving.tool_call_parser = "kimi_k3"
    serving.reasoning_parser = reasoning_parser
    serving.chat_encoding_spec = encoding_spec
    # A template default must not override the request's thinking=False.
    serving.template_manager = SimpleNamespace(force_reasoning=True, reasoning_config=None)
    serving._get_reasoning_from_request = lambda request: thinking
    serving._validate_request = lambda request: None

    def convert(request, raw_request):
        parser = FunctionCallParser(
            request.tools, "kimi_k3", tokenizer=None, tool_choice=request.tool_choice
        )
        constraint = parser.get_structure_constraint(
            request.tool_choice, thinking_mode=False
        )
        state["sampling_params"] = request.to_sampling_params(
            stop=[],
            model_generation_config={},
            tool_call_constraint=constraint,
            renderer_handles_response_format=encoding_spec == "kimi_k3",
        )
        return SimpleNamespace(), request

    serving._convert_to_internal_request = convert
    generators = []

    async def generate(*args):
        try:
            # Cumulative engine text split inside markers and JSON escapes.
            ends = range(1, len(native) + 1) if stream else (len(native),)
            for end in ends:
                state["generated"] += 1
                yield {
                    "text": native[:end],
                    "index": 0,
                    "meta_info": {
                        "id": "chatcmpl-synthetic-k3-json-tools",
                        "prompt_tokens": 13,
                        "completion_tokens": end,
                        "cached_tokens": 0,
                        "finish_reason": {"type": "stop", "matched": None}
                        if end == len(native)
                        else None,
                        "weight_version": "synthetic",
                    },
                }
        finally:
            state["closed"] = True

    def generate_request(*args):
        generator = generate(*args)
        generators.append(generator)
        return generator

    serving.tokenizer_manager = SimpleNamespace(
        tokenizer=None,
        request_logger=SimpleNamespace(log_requests=False),
        generate_request=generate_request,
        create_abort_task=lambda request: None,
    )
    body = {
        "model": "synthetic",
        "messages": [{"role": "user", "content": "synthetic"}],
        "tools": [{"type": "function", "function": {
            "name": "fixture",
            "strict": True,
            "parameters": {
                "type": "object", "properties": {"x": {"const": 7}},
                "required": ["x"], "additionalProperties": False,
            },
        }}],
        "tool_choice": choice,
        "stream": stream,
    }
    if response_format is not None:
        body["response_format"] = copy.deepcopy(response_format)
    config = SimpleNamespace(
        incremental_streaming_output=False,
        stream_response_default_include_usage=False,
        enable_cache_report=False,
        return_input_ids=False,
        return_output_ids=False,
    )

    async def round_trip():
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://synthetic"
            ) as client:
                return await client.post("/v1/chat/completions", json=body)
        finally:
            for generator in generators:
                await generator.aclose()

    with (
        patch.object(app.state, "openai_serving_chat", serving, create=True),
        patch.object(module, "get_serving", return_value=config),
    ):
        return asyncio.run(round_trip()), state


def _events(response):
    assert response.status_code == 200
    data = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert data.count("[DONE]") == 1 and data[-1] == "[DONE]"
    events = [json.loads(item) for item in data[:-1]]
    assert not any("error" in event for event in events)
    return events


def _format(kind, strict=None):
    if kind == "json_object":
        return {"type": kind}
    schema = {"name": "fixture", "schema": {"type": "object"}}
    if strict is not None:
        schema["strict"] = strict
    return {"type": "json_schema", "json_schema": schema}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("kind,strict", [
    ("json_object", None), ("json_schema", True), ("json_schema", None),
])
def test_auto_tools_do_not_reparse_constrained_json(stream, thinking, kind, strict):
    value = {"text": " ".join((
        RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE, TOOLS_OPEN, TOOLS_CLOSE,
        THINK_CLOSE, 'escaped " and slash \\ and Unicode é',
    ))}
    native = (
        ("reasoning" + THINK_CLOSE if thinking else "") + RESPONSE_OPEN
        + json.dumps(value, ensure_ascii=False) + RESPONSE_CLOSE + MESSAGE_CLOSE
    )
    response, state = _request(
        native, stream=stream, thinking=thinking, response_format=_format(kind, strict)
    )
    assert response.status_code == 200
    assert state["sampling_params"].get("json_schema") is not None
    assert state["sampling_params"].get("structural_tag") is None
    if stream:
        choices = [choice for event in _events(response) for choice in event.get("choices", [])]
        text = "".join(choice.get("delta", {}).get("content") or "" for choice in choices)
        reasoning = "".join(choice.get("delta", {}).get("reasoning_content") or "" for choice in choices)
        assert not any(choice.get("delta", {}).get("tool_calls") for choice in choices)
        assert [choice["finish_reason"] for choice in choices if choice.get("finish_reason")] == ["stop"]
    else:
        choice = response.json()["choices"][0]
        text = choice["message"]["content"]
        reasoning = choice["message"].get("reasoning_content") or ""
        assert not choice["message"].get("tool_calls")
        assert choice["finish_reason"] == "stop"
    assert json.loads(text) == value
    assert reasoning == ("reasoning" if thinking else "")
    assert state["closed"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("response_format", [None, _format("json_schema", False)])
def test_plain_and_strict_false_keep_tool_parser(stream, thinking, response_format):
    native = (
        ("reasoning" + THINK_CLOSE if thinking else "") + TOOLS_OPEN
        + '<|open|>call tool="fixture" index="1"<|sep|>'
        + '<|open|>argument key="x" type="number"<|sep|>7' + ARGUMENT_CLOSE
        + CALL_CLOSE + TOOLS_CLOSE + MESSAGE_CLOSE
    )
    response, state = _request(
        native, stream=stream, thinking=thinking, response_format=response_format
    )
    assert response.status_code == 200
    assert state["sampling_params"].get("json_schema") is None
    assert state["sampling_params"].get("structural_tag") is not None
    if stream:
        choices = [choice for event in _events(response) for choice in event.get("choices", [])]
        calls = [call for choice in choices for call in choice.get("delta", {}).get("tool_calls") or []]
        arguments = "".join(call["function"].get("arguments") or "" for call in calls)
        assert [call["function"].get("name") for call in calls if call["function"].get("name")] == ["fixture"]
        assert [choice["finish_reason"] for choice in choices if choice.get("finish_reason")] == ["tool_calls"]
    else:
        choice = response.json()["choices"][0]
        calls = choice["message"]["tool_calls"]
        assert len(calls) == 1 and calls[0]["function"]["name"] == "fixture"
        arguments = calls[0]["function"]["arguments"]
        assert choice["finish_reason"] == "tool_calls"
    assert json.loads(arguments) == {"x": 7}
    assert state["closed"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "fixture"}}])
def test_required_json_conflict_is_rejected_before_generation(stream, choice):
    response, state = _request(
        "unused", stream=stream, thinking=False,
        response_format=_format("json_object"), choice=choice,
    )
    assert response.status_code == 400
    assert state["generated"] == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("kind,strict", [
    ("json_object", None), ("json_schema", True), ("json_schema", None),
])
def test_constrained_json_without_reasoning_parser(stream, kind, strict):
    value = {"text": TOOLS_OPEN + RESPONSE_CLOSE}
    response, state = _request(
        json.dumps(value), stream=stream, thinking=False, reasoning_parser=None,
        response_format=_format(kind, strict),
    )
    assert response.status_code == 200
    assert state["sampling_params"].get("json_schema") is not None
    if stream:
        choices = [choice for event in _events(response) for choice in event.get("choices", [])]
        text = "".join(choice.get("delta", {}).get("content") or "" for choice in choices)
        assert not any(choice.get("delta", {}).get("tool_calls") for choice in choices)
        assert [choice["finish_reason"] for choice in choices if choice.get("finish_reason")] == ["stop"]
    else:
        choice = response.json()["choices"][0]
        text = choice["message"]["content"]
        assert not choice["message"].get("tool_calls")
        assert choice["finish_reason"] == "stop"
    assert json.loads(text) == value
    assert state["closed"]


@pytest.mark.parametrize("stream", [False, True])
def test_strict_false_remains_constrained_without_k3_renderer(stream):
    value = {"text": TOOLS_OPEN + RESPONSE_CLOSE}
    response, state = _request(
        json.dumps(value), stream=stream, thinking=False, reasoning_parser=None,
        encoding_spec=None, response_format=_format("json_schema", False),
    )
    assert response.status_code == 200
    assert state["sampling_params"].get("json_schema") is not None
    assert state["sampling_params"].get("structural_tag") is None
    if stream:
        choices = [choice for event in _events(response) for choice in event.get("choices", [])]
        text = "".join(choice.get("delta", {}).get("content") or "" for choice in choices)
        assert not any(choice.get("delta", {}).get("tool_calls") for choice in choices)
    else:
        choice = response.json()["choices"][0]
        text = choice["message"]["content"]
        assert not choice["message"].get("tool_calls")
    assert json.loads(text) == value
    assert state["closed"]
