"""Never publish an unfinished Nemotron native invocation as a tool call."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, Tool
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat


TOOLS = [Tool.model_validate({"type": "function", "function": {
    "name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": False}}})]
CALL = "<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>"
TAILS = ["<tool_call><function=get_weather>",
         "<tool_call><function=get_weather><parameter=city>Tokyo",
         "<tool_call><function=get_weather><parameter=city>Tokyo</parameter>"]
CHOICES = ["auto", "required", {"type": "function", "function": {"name": "get_weather"}}]


def service(reasoning_parser="nemotron_3"):
    instance = object.__new__(OpenAIServingChat)
    instance.reasoning_parser = reasoning_parser
    instance.tool_call_parser = "qwen3_coder"
    instance.tokenizer_manager = SimpleNamespace(tokenizer=None)
    return instance


def request(choice):
    return ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "test"}],
                                 tools=TOOLS, tool_choice=choice, stream=True)


async def stream(instance, text, choice, width):
    req = request(choice)
    state, has_calls, events = {}, {}, []
    for offset in range(0, len(text), width):
        async for event in instance._process_tool_call_stream(
                0, text[offset:offset + width], state, {"meta_info": {"id": "test"}},
                req, has_calls, flush=offset + width >= len(text)):
            events.append((offset + width, json.loads(event.removeprefix("data: ").strip())))
    calls = {}
    for offset, event in events:
        for choice in event["choices"]:
            for delta in choice["delta"].get("tool_calls") or []:
                entry = calls.setdefault(delta["index"], {"name": "", "arguments": "", "ids": []})
                entry["name"] += delta["function"].get("name") or ""
                entry["arguments"] += delta["function"].get("arguments") or ""
                if delta.get("id"):
                    entry["ids"].append(delta["id"])
    return calls, events


@pytest.mark.parametrize("choice", CHOICES)
@pytest.mark.parametrize("tail", TAILS)
@pytest.mark.parametrize("finish", ["length", "abort"])
def test_nonstream_preserves_complete_call_not_unfinished_tail(choice, tail, finish):
    req = request(choice)
    result = service()._process_tool_calls(CALL + tail, TOOLS, {"type": finish}, req.tool_choice)
    assert result.finish_reason["type"] == finish
    assert len(result.tool_calls) == 1
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Paris"}


@pytest.mark.parametrize("choice", CHOICES)
@pytest.mark.parametrize("tail", TAILS)
@pytest.mark.parametrize("width", [1, 7, 10000])
def test_stream_preserves_complete_call_not_unfinished_tail(choice, tail, width):
    calls, events = asyncio.run(stream(service(), CALL + tail, choice, width))
    assert list(calls) == [0]
    assert calls[0]["name"] == "get_weather"
    assert json.loads(calls[0]["arguments"]) == {"city": "Paris"}
    assert len(calls[0]["ids"]) == 1
    assert min(offset for offset, _ in events) >= CALL.index("</function>") + len("</function>")


@pytest.mark.parametrize("choice", CHOICES)
@pytest.mark.parametrize("width", [1, 7, 10000])
def test_valid_repeated_calls_are_not_deduplicated(choice, width):
    calls, _ = asyncio.run(stream(service(), CALL + CALL, choice, width))
    assert list(calls) == [0, 1]
    assert all(json.loads(c["arguments"]) == {"city": "Paris"} for c in calls.values())
    assert calls[0]["ids"] != calls[1]["ids"]


@pytest.mark.parametrize("tail", TAILS)
def test_incomplete_only_has_no_executable_call(tail):
    result = service()._process_tool_calls(tail, TOOLS, {"type": "length"}, "auto")
    assert not result.tool_calls
    assert result.finish_reason["type"] == "length"
    calls, _ = asyncio.run(stream(service(), tail, "auto", 1))
    assert calls == {}


def test_other_qwen_parser_users_keep_existing_incremental_behavior():
    calls, _ = asyncio.run(stream(service("qwen3"), TAILS[0], "auto", 1))
    assert calls[0]["name"] == "get_weather"


def test_bare_complete_call_is_retained():
    bare = CALL.removeprefix("<tool_call>").removesuffix("</tool_call>")
    calls, _ = asyncio.run(stream(service(), bare, "auto", 1))
    assert json.loads(calls[0]["arguments"]) == {"city": "Paris"}
