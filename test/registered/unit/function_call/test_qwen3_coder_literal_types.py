"""Native XML conversion must not change schema-declared literal types."""

import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "schema,raw,expected",
    [
        ({"type": "string"}, "null", "null"),
        ({"type": "string"}, "NULL", "NULL"),
        ({"type": "string"}, "None", "None"),
        ({"enum": ["null", "other"]}, "null", "null"),
        ({"const": "null"}, "null", "null"),
        ({"const": 7}, "7", 7),
        ({"const": True}, "true", True),
        ({"const": False}, "false", False),
        ({"const": {"ok": True}}, '{"ok":true}', {"ok": True}),
        ({"const": [2, 4]}, "[2,4]", [2, 4]),
        ({"const": None}, "null", None),
        ({"type": "null"}, "null", None),
        ({"type": ["string", "null"]}, "null", None),
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "null", None),
        ({"type": ["object", "null"]}, "null", None),
        ({"type": "object"}, '{"nested":null}', {"nested": None}),
        ({"type": "string"}, "10220_3939392", "10220_3939392"),
        ({"anyOf": [{"const": 7}, {"type": "null"}]}, "7", 7),
        ({"anyOf": [{"const": 7}, {"type": "null"}]}, "null", None),
        ({"allOf": [{"type": "string"}, {"const": "null"}]}, "null", "null"),
        ({"anyOf": [{"const": "null"}, {"const": "other"}]}, "null", "null"),
    ],
)
def test_literal_parameter_type(stream, schema, raw, expected):
    tools = [
        Tool.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "record",
                    "parameters": {"type": "object", "properties": {"value": schema}},
                },
            }
        )
    ]
    detector = Qwen3CoderDetector(require_complete_calls=True)
    text = f"<tool_call><function=record><parameter=value>{raw}</parameter></function></tool_call>"
    if stream:
        parts = []
        for char in text:
            parts.extend(detector.parse_streaming_increment(char, tools).calls)
        arguments = "".join(part.parameters or "" for part in parts)
    else:
        result = detector.detect_and_parse(text, tools)
        assert len(result.calls) == 1
        arguments = result.calls[0].parameters
    value = json.loads(arguments)["value"]
    assert value == expected
    assert type(value) is type(expected)
