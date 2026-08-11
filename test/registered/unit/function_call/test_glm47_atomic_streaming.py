"""Focused regressions for atomic GLM-4.7/GLM-5 streaming tool calls."""

import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector


def _tool(name, properties=None):
    properties = properties or {}
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"Run {name}",
            parameters={"type": "object", "properties": properties},
        ),
    )


@pytest.fixture
def tools():
    return [
        _tool(
            "get_weather",
            {
                "city": {"type": "string"},
                "payload": {"type": "object"},
                "items": {"type": "array"},
                "note": {"type": "string"},
            },
        ),
        _tool("known"),
        _tool("noop"),
    ]


def _call(name, *pairs):
    arguments = "".join(
        f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
        for key, value in pairs
    )
    return f"<tool_call>{name}{arguments}</tool_call>"


def _group(calls):
    grouped = {}
    for item in calls:
        current = grouped.setdefault(item.tool_index, {"name": None, "parameters": ""})
        if item.name is not None:
            current["name"] = item.name
        if item.parameters:
            current["parameters"] += item.parameters
    return [grouped[index] for index in sorted(grouped)]


def _stream(detector, text, tools, chunk_size):
    calls = []
    normal_text = ""
    for offset in range(0, len(text), chunk_size):
        result = detector.parse_streaming_increment(
            text[offset : offset + chunk_size], tools
        )
        calls.extend(result.calls)
        normal_text += result.normal_text
    return calls, normal_text


def test_truncated_call_emits_no_tool_delta(tools):
    detector = Glm47MoeDetector()
    partial = (
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>San Franc"
    )

    result = detector.parse_streaming_increment(partial, tools)

    assert result.calls == []
    assert detector.prev_tool_call_arr == []
    assert detector.streamed_args_for_tool == []


@pytest.mark.parametrize("chunk_size", [1, 7])
def test_call_is_atomic_until_close_for_small_chunks(tools, chunk_size):
    detector = Glm47MoeDetector()
    text = _call("get_weather", ("city", "San Francisco"))
    calls = []
    seen = ""

    for offset in range(0, len(text), chunk_size):
        chunk = text[offset : offset + chunk_size]
        seen += chunk
        result = detector.parse_streaming_increment(chunk, tools)
        if "</tool_call>" not in seen:
            assert result.calls == []
        calls.extend(result.calls)

    assert _group(calls) == [
        {"name": "get_weather", "parameters": '{"city": "San Francisco"}'}
    ]
    assert detector.current_tool_id == 1
    assert detector.prev_tool_call_arr == []
    assert detector.streamed_args_for_tool == []


def test_complete_call_then_truncated_tail_keeps_only_complete_call(tools):
    detector = Glm47MoeDetector()
    text = _call("get_weather", ("city", "Tokyo")) + (
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par"
    )

    result = detector.parse_streaming_increment(text, tools)

    assert _group(result.calls) == [
        {"name": "get_weather", "parameters": '{"city": "Tokyo"}'}
    ]
    assert detector.current_tool_id == 1


def test_two_complete_calls_in_one_increment_are_drained(tools):
    detector = Glm47MoeDetector()
    text = _call("get_weather", ("city", "Tokyo")) + _call("noop")

    result = detector.parse_streaming_increment(text, tools)

    assert _group(result.calls) == [
        {"name": "get_weather", "parameters": '{"city": "Tokyo"}'},
        {"name": "noop", "parameters": "{}"},
    ]
    assert detector.current_tool_id == 2


def test_unknown_then_known_drops_unknown_without_consuming_index(tools):
    detector = Glm47MoeDetector()
    text = _call("unknown", ("value", "discard")) + _call("known")

    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        result = detector.parse_streaming_increment(text, tools)

    assert _group(result.calls) == [{"name": "known", "parameters": "{}"}]
    assert result.calls[0].tool_index == 0
    assert detector.current_tool_id == 1


def test_unknown_tool_is_forwarded_atomically_when_enabled(tools):
    detector = Glm47MoeDetector()
    text = _call("unknown", ("value", "forward"))

    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        result = detector.parse_streaming_increment(text, tools)

    assert _group(result.calls) == [
        {"name": "unknown", "parameters": '{"value": "forward"}'}
    ]


@pytest.mark.parametrize("chunk_size", [1, 7])
def test_complex_arguments_remain_valid_json(tools, chunk_size):
    long_note = "x" * 4096 + r" \d+ C:\work\file.txt"
    text = _call(
        "get_weather",
        ("city", "Paris"),
        ("payload", '{"nested":{"ok":true}}'),
        ("items", '[1,{"two":2}]'),
        ("note", long_note),
    )

    calls, _ = _stream(Glm47MoeDetector(), text, tools, chunk_size)
    [call] = _group(calls)
    parameters = json.loads(call["parameters"])

    assert call["name"] == "get_weather"
    assert parameters == {
        "city": "Paris",
        "payload": {"nested": {"ok": True}},
        "items": [1, {"two": 2}],
        "note": long_note,
    }


def _schema_incident_tool():
    return _tool(
        "record_union_values",
        {
            "retry": {"type": ["integer", "null"], "const": None},
            "payload": {
                "oneOf": [{"type": "string"}, {"type": "integer"}],
                "const": 7,
            },
            "state": {"const": "ready"},
            "selector": {"enum": ["alpha", 2, False], "const": False},
            "mode": {"enum": ["safe", "fast"], "const": "safe"},
        },
    )


def _schema_incident_call():
    return _call(
        "record_union_values",
        ("retry", "null"),
        ("payload", "7"),
        ("state", "ready"),
        ("selector", "False"),
        ("mode", "safe"),
    )


def test_non_streaming_schema_coercion_matches_incident_contract():
    result = Glm47MoeDetector().detect_and_parse(
        _schema_incident_call(), [_schema_incident_tool()]
    )

    assert len(result.calls) == 1
    assert json.loads(result.calls[0].parameters) == {
        "retry": None,
        "payload": 7,
        "state": "ready",
        "selector": False,
        "mode": "safe",
    }


@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
def test_atomic_streaming_schema_coercion_matches_incident_contract(chunk_size):
    calls, normal_text = _stream(
        Glm47MoeDetector(),
        _schema_incident_call(),
        [_schema_incident_tool()],
        chunk_size,
    )

    [call] = _group(calls)
    assert normal_text == ""
    assert call["name"] == "record_union_values"
    assert json.loads(call["parameters"]) == {
        "retry": None,
        "payload": 7,
        "state": "ready",
        "selector": False,
        "mode": "safe",
    }


def test_mixed_enum_without_const_uses_only_schema_valid_candidate():
    tool = _tool(
        "record_mixed_enum",
        {"selector": {"enum": ["alpha", 2, False]}},
    )
    result = Glm47MoeDetector().detect_and_parse(
        _call("record_mixed_enum", ("selector", "False")), [tool]
    )

    assert json.loads(result.calls[0].parameters) == {"selector": False}


def test_ambiguous_union_preserves_genuine_numeric_string():
    tool = _tool(
        "lookup",
        {
            "union_id": {
                "oneOf": [{"type": "string"}, {"type": "integer"}]
            },
            "enum_id": {"enum": ["7", 7]},
            "quoted_id": {
                "oneOf": [{"type": "string"}, {"type": "integer"}]
            },
            "quoted_const": {
                "oneOf": [{"type": "string"}, {"type": "integer"}],
                "const": 7,
            },
            "integer_id": {"type": "integer"},
        },
    )
    result = Glm47MoeDetector().detect_and_parse(
        _call(
            "lookup",
            ("union_id", "7"),
            ("enum_id", "7"),
            ("quoted_id", '"7"'),
            ("quoted_const", '"7"'),
            ("integer_id", "7"),
        ),
        [tool],
    )

    assert json.loads(result.calls[0].parameters) == {
        "union_id": "7",
        "enum_id": "7",
        "quoted_id": "7",
        "quoted_const": 7,
        "integer_id": 7,
    }


def test_invalid_value_is_not_replaced_with_schema_const():
    tool = _tool("record_const", {"payload": {"type": "integer", "const": 7}})
    result = Glm47MoeDetector().detect_and_parse(
        _call("record_const", ("payload", "8")), [tool]
    )

    assert json.loads(result.calls[0].parameters) == {"payload": 8}


def test_local_defs_and_top_level_oneof_parameter_lookup():
    tool = Tool(
        type="function",
        function=Function(
            name="resolve_branch",
            description="Resolve a top-level schema branch",
            parameters={
                "$defs": {"seven": {"type": "integer", "const": 7}},
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"payload": {"$ref": "#/$defs/seven"}},
                        "required": ["payload"],
                    },
                    {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                ],
            },
        ),
    )
    result = Glm47MoeDetector().detect_and_parse(
        _call("resolve_branch", ("payload", "7")), [tool]
    )

    assert json.loads(result.calls[0].parameters) == {"payload": 7}


@pytest.mark.parametrize("chunk_size", [1, 7])
def test_normal_text_is_preserved_around_atomic_call(tools, chunk_size):
    text = "before <not-a-tool> " + _call("known") + " after"

    calls, normal_text = _stream(Glm47MoeDetector(), text, tools, chunk_size)

    assert _group(calls) == [{"name": "known", "parameters": "{}"}]
    assert normal_text == "before <not-a-tool>  after"
