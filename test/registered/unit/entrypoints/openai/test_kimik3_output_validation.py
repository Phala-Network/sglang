"""K3 output guard through the real HTTP route and JSON/SSE response builders.

Generation is synthetic; request validation/tokenization are boundary doubles.
These tests make no model calls and do not claim grammar or GPU acceptance.
"""

import asyncio
import copy
import json
import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from sglang.srt.entrypoints.http_server import app
from sglang.srt.entrypoints.openai import serving_chat as module
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.environ import ToolStrictLevel
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ERROR_MESSAGE = "Kimi K3 generated tool output is invalid."
INTEGER_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "integer", "const": 7}},
    "required": ["x"],
    "additionalProperties": False,
}


def _tool(schema=INTEGER_SCHEMA, *, strict=True, name="fixture"):
    return {
        "type": "function",
        "function": {"name": name, "strict": strict, "parameters": schema},
    }


def _argument(key, kind, value):
    raw = value if kind == "string" else json.dumps(value, ensure_ascii=False)
    return f'<|open|>argument key="{key}" type="{kind}"<|sep|>{raw}{ARGUMENT_CLOSE}'


def _call(*arguments, name="fixture", index=1):
    return (
        f'<|open|>call tool="{name}" index="{index}"<|sep|>'
        + "".join(arguments)
        + CALL_CLOSE
    )


def _section(*calls):
    return TOOLS_OPEN + "".join(calls) + TOOLS_CLOSE + MESSAGE_CLOSE


def _item(text, *, index=0, finish="stop"):
    return {
        "text": text,
        "index": index,
        "meta_info": {
            "id": "chatcmpl-synthetic-k3-output",
            "prompt_tokens": 13,
            "completion_tokens": 11,
            "cached_tokens": 0,
            "finish_reason": {"type": finish, "matched": None} if finish else None,
            "weight_version": "synthetic",
        },
    }


def _request(
    items,
    *,
    stream=False,
    tools=None,
    level=0,
    choice="required",
    n=1,
    messages=None,
    parser="kimi_k3",
    reasoning="kimi_k3",
    parser_failure=False,
):
    """Exercise FastAPI -> handle_request -> real parser/builders, no sockets."""
    state = {"generated": 0, "closed": False, "published_ids": []}
    serving = object.__new__(OpenAIServingChat)
    # Attributes __init__ sets on the fork's main line.
    serving._encoding_adapter = None
    serving.is_minimax_m3 = False
    serving.tool_call_parser = parser
    serving.reasoning_parser = reasoning
    serving.chat_encoding_spec = "kimi_k3" if parser == "kimi_k3" else None
    serving.template_manager = SimpleNamespace(
        force_reasoning=False, reasoning_config=None
    )
    serving._get_reasoning_from_request = lambda request: False
    serving._validate_request = lambda request: None
    serving._convert_to_internal_request = lambda request, raw: (
        SimpleNamespace(),
        request,
    )
    generators = []

    async def generate(*args):
        try:
            for item in items:
                state["generated"] += 1
                yield copy.deepcopy(item)
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
    process_id = serving._process_tool_call_id

    def record_id(call, count):
        state["published_ids"].append(call.name)
        return process_id(call, count)

    serving._process_tool_call_id = record_id
    parser_factory = module.FunctionCallParser

    def faulting_parser(*args, **kwargs):
        parser_instance = parser_factory(*args, **kwargs)

        def fail_parse(*args, **kwargs):
            state["parser_fault_injected"] = True
            raise RuntimeError("SECRET_PARSER_VALUE")

        parser_instance.parse_non_stream = fail_parse
        return parser_instance

    parser_patch = (
        patch.object(module, "FunctionCallParser", side_effect=faulting_parser)
        if parser_failure
        else nullcontext()
    )
    config = SimpleNamespace(
        incremental_streaming_output=False,
        stream_response_default_include_usage=False,
        enable_cache_report=False,
        return_input_ids=False,
        return_output_ids=False,
    )
    body = {
        "model": "synthetic",
        "messages": messages or [{"role": "user", "content": "synthetic"}],
        "tools": tools if tools is not None else [_tool()],
        "tool_choice": choice,
        "stream": stream,
        "n": n,
    }

    async def round_trip():
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://synthetic"
            ) as client:
                response = await client.post("/v1/chat/completions", json=body)
                # Record closure before fixture cleanup can close the generator.
                state["response_closed_generation"] = state["closed"]
                return response
        finally:
            for generator in generators:
                await generator.aclose()

    with (
        patch.object(app.state, "openai_serving_chat", serving, create=True),
        patch.object(module, "get_serving", return_value=config),
        patch.dict(os.environ, {"SGLANG_TOOL_STRICT_LEVEL": str(level)}),
        parser_patch,
    ):
        return asyncio.run(round_trip()), state


def _events(response):
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    data = [
        line[6:] for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert data.count("[DONE]") == 1
    assert data[-1] == "[DONE]"
    return [json.loads(item) for item in data[:-1]]


def _calls(events):
    return [
        (choice["index"], call)
        for event in events
        for choice in event.get("choices", [])
        for call in choice.get("delta", {}).get("tool_calls") or []
    ]


def _assert_error(response, state, stream, *, earlier_names=()):
    if stream:
        events = _events(response)
        errors = [event["error"] for event in events if "error" in event]
        assert len(errors) == 1
        error = errors[0]
        assert events[-1] == {"error": error}
        assert all(
            choice.get("finish_reason") is None
            for event in events
            for choice in event.get("choices", [])
        )
        assert [call["function"]["name"] for _, call in _calls(events)] == list(
            earlier_names
        )
        assert state["response_closed_generation"]
    else:
        assert response.status_code == 500
        assert "application/json" in response.headers["content-type"]
        error = response.json()
        assert "choices" not in error
        assert "data:" not in response.text
    assert error == {
        "object": "error",
        "message": ERROR_MESSAGE,
        "type": "InternalServerError",
        "param": None,
        "code": 500,
    }
    assert state["published_ids"] == list(earlier_names)
    assert "<|" not in response.text
    return error


INVALID_CASES = [
    pytest.param(INTEGER_SCHEMA, [_argument("x", "number", 8)], id="const"),
    pytest.param(
        {**INTEGER_SCHEMA, "properties": {"x": {"enum": [7, 9]}}},
        [_argument("x", "number", 8)],
        id="enum",
    ),
    pytest.param(
        INTEGER_SCHEMA,
        [_argument("x", "number", 7), _argument("private", "string", "SECRET_VALUE")],
        id="extra",
    ),
    pytest.param(INTEGER_SCHEMA, [], id="missing-required"),
    pytest.param(INTEGER_SCHEMA, [_argument("x", "string", "SECRET_VALUE")], id="type"),
    pytest.param(
        {**INTEGER_SCHEMA, "properties": {"x": {"type": "number"}}},
        [_argument("x", "number", float("nan"))],
        id="nan",
    ),
]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("schema,arguments", INVALID_CASES)
def test_invalid_output_is_redacted_500(schema, arguments, stream):
    response, state = _request(
        [_item(_section(_call(*arguments)))],
        stream=stream,
        tools=[_tool(schema)],
    )
    _assert_error(response, state, stream)
    assert "SECRET_VALUE" not in response.text
    assert "additionalProperties" not in response.text
    assert state["generated"] == 1


@pytest.mark.parametrize("stream", [False, True])
def test_raw_argument_delimiter_cannot_publish_schema_violation(stream):
    value = (
        "a" + ARGUMENT_CLOSE + '<|open|>argument key="injected" type="string"<|sep|>b'
    )
    schema = {
        "type": "object",
        "properties": {"payload": {"const": value}},
        "required": ["payload"],
        "additionalProperties": False,
    }
    response, state = _request(
        [_item(_section(_call(_argument("payload", "string", value))))],
        stream=stream,
        tools=[_tool(schema)],
    )
    _assert_error(response, state, stream)
    assert "injected" not in response.text


@pytest.mark.parametrize("stream", [False, True])
def test_strict_auto_output_is_redacted_500(stream):
    response, state = _request(
        [_item(_section(_call(_argument("x", "number", 8))))],
        stream=stream,
        choice="auto",
    )
    _assert_error(response, state, stream)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "level,strict,reject",
    [(0, False, False), (1, False, False), (2, False, True), (0, True, True)],
)
def test_strict_policy_gates_parameters(level, strict, reject, stream):
    response, state = _request(
        [_item(_section(_call(_argument("x", "number", 8))))],
        stream=stream,
        tools=[_tool(strict=strict)],
        level=level,
    )
    if reject:
        _assert_error(response, state, stream)
    elif stream:
        assert json.loads(_calls(_events(response))[0][1]["function"]["arguments"]) == {
            "x": 8
        }
    else:
        assert response.status_code == 200
        assert json.loads(
            response.json()["choices"][0]["message"]["tool_calls"][0]["function"][
                "arguments"
            ]
        ) == {"x": 8}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "name,level,reject", [("strict", 0, True), ("loose", 0, False), ("loose", 2, True)]
)
def test_mixed_tools_keep_selected_tool_strictness(name, level, reject, stream):
    response, state = _request(
        [_item(_section(_call(_argument("x", "number", 8), name=name)))],
        stream=stream,
        tools=[_tool(name="strict"), _tool(name="loose", strict=False)],
        level=level,
    )
    if reject:
        _assert_error(response, state, stream)
    elif stream:
        assert _calls(_events(response))[0][1]["function"]["name"] == "loose"
    else:
        assert response.status_code == 200
        assert (
            response.json()["choices"][0]["message"]["tool_calls"][0]["function"][
                "name"
            ]
            == "loose"
        )


LOCAL_REF_SCHEMA = {
    "$defs": {
        "Number": {"type": "integer", "minimum": 1},
        "Box": {
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/Number"}},
            "required": ["x"],
            "additionalProperties": False,
        },
    },
    "type": "object",
    "properties": {"payload": {"$ref": "#/$defs/Box"}},
    "required": ["payload"],
    "additionalProperties": False,
}


def test_input_validation_grammar_and_renderer_preserve_guard_schema():
    request = ChatCompletionRequest(
        model="synthetic",
        messages=[{"role": "user", "content": "synthetic"}],
        tools=[_tool(copy.deepcopy(LOCAL_REF_SCHEMA))],
        tool_choice="required",
    )
    schema = request.tools[0].function.parameters
    before = copy.deepcopy(schema)
    seen = {}

    def render(messages, **kwargs):
        serialized = kwargs["tools"][0]["function"]["parameters"]
        seen["rendered_schema"] = copy.deepcopy(serialized)
        # A renderer receives detached serialization, even if it edits nested
        # definitions in place. This would accept x=0 if it reached the guard.
        serialized["$defs"]["Number"]["minimum"] = -100
        return [1]

    serving = object.__new__(OpenAIServingChat)
    # Attributes __init__ sets on the fork's main line.
    serving._encoding_adapter = None
    serving.is_minimax_m3 = False
    serving.chat_encoding_spec = serving.tool_call_parser = "kimi_k3"
    serving.tokenizer_manager = SimpleNamespace(
        model_config=SimpleNamespace(is_multimodal=False),
        tokenizer=SimpleNamespace(apply_chat_template=render),
    )
    with patch.object(
        module, "get_model", return_value=SimpleNamespace(context_length=1048576)
    ):
        assert serving._validate_request(request) is None
    assert request.tools[0].function.parameters is schema
    assert schema == before

    parser = FunctionCallParser(request.tools, "kimi_k3", tool_choice="required")
    constraint = parser.get_structure_constraint("required")
    assert constraint is not None and constraint[0] == "structural_tag"
    assert schema == before

    messages = [message.model_dump(exclude_none=True) for message in request.messages]
    assert serving._encode_messages(
        messages, request, thinking_mode=False, image_data=[]
    ) == [1]
    assert seen["rendered_schema"] == before
    assert request.tools[0].function.parameters is schema
    assert schema == before

    valid = ToolCallItem(tool_index=0, name="fixture", parameters='{"payload":{"x":1}}')
    invalid = valid.model_copy(update={"parameters": '{"payload":{"x":0}}'})
    serving._validate_kimi_k3_tool_call(valid, request.tools, ToolStrictLevel.OFF)
    with pytest.raises(
        module._KimiK3ToolCallValidationError,
        match="^" + ERROR_MESSAGE.replace(".", r"\.") + "$",
    ):
        serving._validate_kimi_k3_tool_call(invalid, request.tools, ToolStrictLevel.OFF)
    assert schema == before


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_full_local_ref_scope(stream, valid):
    value = {"x": 1 if valid else "SECRET_VALUE"}
    response, state = _request(
        [_item(_section(_call(_argument("payload", "object", value))))],
        stream=stream,
        tools=[_tool(LOCAL_REF_SCHEMA)],
    )
    if not valid:
        _assert_error(response, state, stream)
    elif stream:
        assert json.loads(_calls(_events(response))[0][1]["function"]["arguments"]) == {
            "payload": value
        }
    else:
        assert response.status_code == 200
        assert json.loads(
            response.json()["choices"][0]["message"]["tool_calls"][0]["function"][
                "arguments"
            ]
        ) == {"payload": value}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "ref", ["https://invalid.example/SECRET_SCHEMA", "#/$defs/Missing"]
)
def test_unresolved_refs_fail_without_retrieval(stream, ref):
    schema = {"type": "object", "properties": {"x": {"$ref": ref}}}
    response, state = _request(
        [_item(_section(_call(_argument("x", "number", 7))))],
        stream=stream,
        tools=[_tool(schema)],
    )
    _assert_error(response, state, stream)
    assert "SECRET_SCHEMA" not in response.text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("extra", [False, True])
def test_none_parameters_preserves_no_arguments_contract(stream, extra):
    arguments = [_argument("x", "number", 7)] if extra else []
    response, state = _request(
        [_item(_section(_call(*arguments)))], stream=stream, tools=[_tool(None)]
    )
    if extra:
        _assert_error(response, state, stream)
    elif stream:
        assert (
            json.loads(_calls(_events(response))[0][1]["function"]["arguments"]) == {}
        )
    else:
        assert response.status_code == 200
        assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("same_delta", [False, True])
def test_invalid_call_does_not_publish_after_valid_call(same_delta):
    good = _call(_argument("x", "number", 7), name="good")
    bad = _call(_argument("x", "string", "SECRET_VALUE"), name="bad", index=2)
    prefix = TOOLS_OPEN + good
    full = prefix + bad + TOOLS_CLOSE + MESSAGE_CLOSE
    items = [_item(full)] if same_delta else [_item(prefix, finish=None), _item(full)]
    response, state = _request(
        items, stream=True, tools=[_tool(name="good"), _tool(name="bad")]
    )
    _assert_error(response, state, True, earlier_names=["good"])
    assert "SECRET_VALUE" not in response.text
    assert '"name":"bad"' not in response.text


def test_nonstream_invalid_second_call_returns_error_without_choices():
    text = _section(
        _call(_argument("x", "number", 7), name="good"),
        _call(_argument("x", "string", "SECRET_VALUE"), name="bad", index=2),
    )
    response, state = _request(
        [_item(text)], tools=[_tool(name="good"), _tool(name="bad")]
    )
    _assert_error(response, state, False, earlier_names=["good"])
    assert "SECRET_VALUE" not in response.text
    assert '"name":"good"' not in response.text


@pytest.mark.parametrize("stream", [False, True])
def test_unknown_name_is_guarded_before_id(stream):
    response, state = _request(
        [_item(_section(_call(_argument("x", "number", 7), name="SECRET_NAME")))],
        stream=stream,
    )
    _assert_error(response, state, stream)
    assert "SECRET_NAME" not in response.text


@pytest.mark.parametrize("stream", [False, True])
def test_effective_developer_tool_is_validated(stream):
    messages = [
        {"role": "developer", "content": "synthetic", "tools": [_tool()]},
        {"role": "user", "content": "synthetic"},
    ]
    response, state = _request(
        [_item(_section(_call(_argument("x", "number", 8))))],
        stream=stream,
        tools=[],
        messages=messages,
    )
    _assert_error(response, state, stream)


@pytest.mark.parametrize("stream", [False, True])
def test_n_choices_are_isolated(stream):
    good = _item(_section(_call(_argument("x", "number", 7))), index=0)
    bad = _item(_section(_call(_argument("x", "number", 8))), index=1)
    response, state = _request(
        [good, bad] if stream else [[good, bad]], stream=stream, n=2
    )
    _assert_error(response, state, stream, earlier_names=["fixture"])
    if stream:
        assert [index for index, _ in _calls(_events(response))] == [0]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "choice",
    ["auto", "required", {"type": "function", "function": {"name": "fixture"}}],
)
@pytest.mark.parametrize("kind", ["string", "object", "json-string"])
def test_legal_marker_values_survive_guard(stream, choice, kind):
    value = '雪 " \\ ' + CALL_CLOSE + TOOLS_CLOSE + MESSAGE_CLOSE
    if kind == "json-string":
        value += (
            ARGUMENT_CLOSE + '<|open|>argument key="injected" type="string"<|sep|>b'
        )
    payload = {"inner": [value, True]} if kind == "object" else value
    schema = {
        "type": "object",
        "properties": {"payload": {"const": payload}},
        "required": ["payload"],
        "additionalProperties": False,
    }
    text = (
        RESPONSE_OPEN
        + "Checking."
        + RESPONSE_CLOSE
        + _section(_call(_argument("payload", kind, payload)))
    )
    response, state = _request(
        [_item(text)], stream=stream, tools=[_tool(schema)], choice=choice
    )
    assert response.status_code == 200
    if stream:
        events = _events(response)
        assert not any("error" in event for event in events)
        calls = _calls(events)
        assert len(calls) == 1
        function = calls[0][1]["function"]
        assert any(
            choice["finish_reason"] == "tool_calls"
            for event in events
            for choice in event.get("choices", [])
        )
    else:
        result = response.json()["choices"][0]
        assert result["finish_reason"] == "tool_calls"
        assert result["message"]["content"] == "Checking."
        function = result["message"]["tool_calls"][0]["function"]
    assert json.loads(function["arguments"]) == {"payload": payload}
    assert state["published_ids"] == ["fixture"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", ["auto", "none"])
def test_auto_and_none_prose_remain_normal_success(stream, choice):
    text = RESPONSE_OPEN + "A plain reply." + RESPONSE_CLOSE + MESSAGE_CLOSE
    response, state = _request([_item(text)], stream=stream, choice=choice)
    assert response.status_code == 200
    if stream:
        events = _events(response)
        assert not _calls(events)
        assert not any("error" in event for event in events)
        assert "A plain reply." in response.text
    else:
        choice = response.json()["choices"][0]
        assert choice["message"]["content"] == "A plain reply."
        assert not choice["message"].get("tool_calls")
        assert choice["finish_reason"] == "stop"
    assert not state["published_ids"]


@pytest.mark.parametrize("stream", [False, True])
def test_guard_checks_generated_call_with_history(stream):
    messages = [
        {"role": "user", "content": "earlier"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "old",
                    "type": "function",
                    "function": {"name": "fixture", "arguments": '{"x":8}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "old", "content": "synthetic history"},
        {"role": "user", "content": "next synthetic turn"},
    ]
    response, state = _request(
        [_item(_section(_call(_argument("x", "number", 7))))],
        stream=stream,
        messages=messages,
    )
    assert response.status_code == 200
    if stream:
        events = _events(response)
        assert not any("error" in event for event in events)
        assert len(_calls(events)) == 1
    else:
        assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert state["published_ids"] == ["fixture"]


def _malformed_output(kind):
    if kind == "missing":
        return RESPONSE_OPEN + "Synthetic reply." + RESPONSE_CLOSE + MESSAGE_CLOSE
    wire_type, value = ("object", "{broken}") if kind == "json" else ("unknown", "{}")
    return _section(
        _call(
            f'<|open|>argument key="x" type="{wire_type}"<|sep|>'
            + value
            + ARGUMENT_CLOSE
        )
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "choice", ["required", {"type": "function", "function": {"name": "fixture"}}]
)
@pytest.mark.parametrize("kind", ["json", "unknown", "missing"])
def test_required_stopped_without_complete_call_is_generation_error(
    stream, choice, kind
):
    response, state = _request(
        [_item(_malformed_output(kind))], stream=stream, choice=choice
    )
    _assert_error(response, state, stream)


@pytest.mark.parametrize("stream", [False, True])
def test_auto_malformed_output_keeps_existing_completion_behavior(stream):
    response, state = _request(
        [_item(_malformed_output("json"))], stream=stream, choice="auto"
    )
    assert response.status_code == 200
    if stream:
        events = _events(response)
        assert not _calls(events)
        assert not any("error" in event for event in events)
        assert any(
            choice["finish_reason"] == "stop"
            for event in events
            for choice in event.get("choices", [])
        )
    else:
        result = response.json()["choices"][0]
        assert not result["message"].get("tool_calls")
        assert result["finish_reason"] == "stop"
    assert not state["published_ids"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "choice", ["required", {"type": "function", "function": {"name": "fixture"}}]
)
def test_existing_complete_call_satisfies_required_even_with_later_bad_frame(
    stream, choice
):
    first = TOOLS_OPEN + _call(_argument("x", "number", 7))
    full = (
        first
        + _call(
            '<|open|>argument key="x" type="number"<|sep|>{broken}' + ARGUMENT_CLOSE,
            index=2,
        )
        + TOOLS_CLOSE
        + MESSAGE_CLOSE
    )
    items = [_item(first, finish=None), _item(full)] if stream else [_item(full)]
    response, state = _request(items, stream=stream, choice=choice)
    assert response.status_code == 200
    if stream:
        events = _events(response)
        assert not any("error" in event for event in events)
        assert len(_calls(events)) == 1
        assert any(
            choice["finish_reason"] == "tool_calls"
            for event in events
            for choice in event.get("choices", [])
        )
    else:
        result = response.json()["choices"][0]
        assert len(result["message"]["tool_calls"]) == 1
        assert result["finish_reason"] == "tool_calls"
    assert state["published_ids"] == ["fixture"]


@pytest.mark.parametrize("stream", [False, True])
def test_required_completion_checks_each_choice(stream):
    good = _item(_section(_call(_argument("x", "number", 7))), index=0)
    missing = _item(_malformed_output("missing"), index=1)
    response, state = _request(
        [good, missing] if stream else [[good, missing]], stream=stream, n=2
    )
    _assert_error(response, state, stream, earlier_names=["fixture"])
    if stream:
        assert [index for index, _ in _calls(_events(response))] == [0]


@pytest.mark.parametrize("parser", ["kimi_k3", "hermes"])
def test_parser_exception_cannot_bypass_k3_required_completion(parser):
    text = (
        _section(_call(_argument("x", "number", 7)))
        if parser == "kimi_k3"
        else '<tool_call>{"name":"fixture","arguments":{"x":7}}</tool_call>'
    )
    response, state = _request(
        [_item(text)],
        parser=parser,
        reasoning="kimi_k3" if parser == "kimi_k3" else None,
        choice="required" if parser == "kimi_k3" else "auto",
        parser_failure=True,
    )
    assert state.get("parser_fault_injected")
    if parser == "kimi_k3":
        _assert_error(response, state, False)
    else:
        assert response.status_code == 200
        assert response.json()["choices"][0]["finish_reason"] == "stop"
    assert "SECRET_PARSER_VALUE" not in response.text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "choice", ["required", {"type": "function", "function": {"name": "fixture"}}]
)
@pytest.mark.parametrize("finish", ["length", "abort"])
def test_truncated_call_never_publishes_partial_tool(stream, choice, finish):
    text = TOOLS_OPEN + _call(_argument("x", "number", 7))[: -len(CALL_CLOSE)]
    response, state = _request(
        [_item(text, finish=finish)], stream=stream, choice=choice
    )
    assert response.status_code == 200
    if stream:
        events = _events(response)
        assert not _calls(events)
        assert any(
            choice["finish_reason"] == finish
            for event in events
            for choice in event.get("choices", [])
        )
    else:
        choice = response.json()["choices"][0]
        assert not choice["message"].get("tool_calls")
        assert choice["finish_reason"] == finish
    assert not state["published_ids"]
    assert "<|" not in response.text


@pytest.mark.parametrize("stream", [False, True])
def test_other_parser_retains_existing_parameter_behavior(stream):
    text = '<tool_call>{"name":"fixture","arguments":{"x":8}}</tool_call>'
    response, state = _request(
        [_item(text)], stream=stream, parser="hermes", reasoning=None, choice="auto"
    )
    assert response.status_code == 200
    if stream:
        events = _events(response)
        assert not any("error" in event for event in events)
        assert _calls(events)
    else:
        assert response.json()["choices"][0]["message"]["tool_calls"]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
