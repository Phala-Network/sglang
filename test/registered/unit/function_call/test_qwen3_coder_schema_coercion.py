import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector


def _tool(name, properties, **parameter_overrides):
    parameters = {"type": "object", "properties": properties}
    parameters.update(parameter_overrides)
    return Tool(
        type="function",
        function=Function(name=name, description="test", parameters=parameters),
    )


def _call(name, *parameters):
    body = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in parameters
    )
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _stream(detector, text, tools, chunk_size):
    calls = []
    for start in range(0, len(text), chunk_size):
        calls.extend(
            detector.parse_streaming_increment(
                text[start : start + chunk_size], tools
            ).calls
        )
    grouped = {}
    for call in calls:
        item = grouped.setdefault(call.tool_index, {"name": "", "parameters": ""})
        if call.name:
            item["name"] = call.name
        if call.parameters:
            item["parameters"] += call.parameters
    return [grouped[index] for index in sorted(grouped)]


def test_named_defs_ref_nested_object_non_streaming():
    tool = _tool(
        "get_weather_by_coordinate",
        {"location": {"$ref": "#/$defs/Coordinate"}},
        **{
            "$defs": {
                "Coordinate": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                    },
                    "required": ["lat", "lon"],
                }
            }
        },
    )
    result = Qwen3CoderDetector().detect_and_parse(
        _call(
            "get_weather_by_coordinate",
            ("location", '{"lat": 48.8566, "lon": 2.3522}'),
        ),
        [tool],
    )

    assert json.loads(result.calls[0].parameters) == {
        "location": {"lat": 48.8566, "lon": 2.3522}
    }


@pytest.mark.parametrize("chunk_size", [1, 7, 64])
def test_named_defs_ref_nested_object_streaming(chunk_size):
    tool = _tool(
        "get_weather_by_coordinate",
        {"location": {"$ref": "#/$defs/Coordinate"}},
        **{
            "$defs": {
                "Coordinate": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                    },
                }
            }
        },
    )
    calls = _stream(
        Qwen3CoderDetector(),
        _call(
            "get_weather_by_coordinate",
            ("location", '{"lat": 48.8566, "lon": 2.3522}'),
        ),
        [tool],
        chunk_size,
    )

    assert len(calls) == 1
    assert json.loads(calls[0]["parameters"]) == {
        "location": {"lat": 48.8566, "lon": 2.3522}
    }


def test_mixed_union_enum_const_uses_schema_valid_scalar_types():
    tool = _tool(
        "record_union_values",
        {
            "retry": {"type": ["integer", "null"]},
            "payload": {"oneOf": [{"type": "string"}, {"type": "integer"}], "const": 7},
            "state": {"enum": ["ready", "queued"]},
            "selector": {"enum": ["alpha", 2, False], "const": False},
            "mode": {"type": "string", "const": "safe"},
        },
    )
    result = Qwen3CoderDetector().detect_and_parse(
        _call(
            "record_union_values",
            ("retry", "null"),
            ("payload", "7"),
            ("state", "ready"),
            ("selector", "false"),
            ("mode", "safe"),
        ),
        [tool],
    )

    assert json.loads(result.calls[0].parameters) == {
        "retry": None,
        "payload": 7,
        "state": "ready",
        "selector": False,
        "mode": "safe",
    }


def test_ambiguous_union_preserves_genuine_numeric_string():
    tool = _tool(
        "lookup",
        {
            "union_id": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            "enum_id": {"enum": ["7", 7]},
            "quoted_id": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            "quoted_const": {
                "oneOf": [{"type": "string"}, {"type": "integer"}],
                "const": 7,
            },
            "integer_id": {"type": "integer"},
        },
    )
    result = Qwen3CoderDetector().detect_and_parse(
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
    result = Qwen3CoderDetector().detect_and_parse(
        _call("record_const", ("payload", "8")), [tool]
    )

    assert json.loads(result.calls[0].parameters) == {"payload": 8}


def test_top_level_oneof_parameter_lookup_preserves_local_defs():
    tool = Tool(
        type="function",
        function=Function(
            name="resolve_branch",
            description="test",
            parameters={
                "$defs": {"seven": {"type": "integer", "const": 7}},
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"payload": {"$ref": "#/$defs/seven"}},
                    },
                    {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                    },
                ],
            },
        ),
    )
    result = Qwen3CoderDetector().detect_and_parse(
        _call("resolve_branch", ("payload", "7")), [tool]
    )

    assert json.loads(result.calls[0].parameters) == {"payload": 7}
