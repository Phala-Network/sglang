"""JSON Schema references must retain parameter types in native XML tools."""

import copy
import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "definition,value",
    [
        ({"type": "object", "properties": {"lat": {"type": "number"}}}, {"lat": 48.85}),
        ({"type": "array", "items": {"type": "integer"}}, [2, 4, 6]),
        ({"type": "integer"}, 7),
        ({"type": "boolean"}, True),
        ({"anyOf": [{"type": "null"}, {"type": "object"}]}, {"ok": True}),
        ({"type": "string"}, "10220_3939392"),
    ],
)
def test_local_reference_parameter_conversion(stream, definition, value):
    parameters = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/Alias"}},
        "$defs": {"Alias": {"$ref": "#/$defs/A~1B~0C"}, "A/B~C": definition},
    }
    tools = [
        Tool.model_validate(
            {
                "type": "function",
                "function": {"name": "record", "parameters": parameters},
            }
        )
    ]
    before = copy.deepcopy(tools[0].function.parameters)
    detector = Qwen3CoderDetector(require_complete_calls=True)
    encoded = value if isinstance(value, str) else json.dumps(value)
    text = f"<tool_call><function=record><parameter=value>{encoded}</parameter></function></tool_call>"
    if stream:
        items = []
        for char in text:
            items.extend(detector.parse_streaming_increment(char, tools).calls)
        arguments = "".join(item.parameters or "" for item in items)
    else:
        result = detector.detect_and_parse(text, tools)
        assert len(result.calls) == 1
        arguments = result.calls[0].parameters
    assert json.loads(arguments) == {"value": value}
    assert tools[0].function.parameters == before


def test_cyclic_and_remote_references_do_not_recurse_or_fetch():
    for reference in ["#/$defs/Loop", "https://example.invalid/schema.json"]:
        parameters = {
            "type": "object",
            "properties": {"value": {"$ref": reference}},
            "$defs": {"Loop": {"$ref": "#/$defs/Loop"}},
        }
        tools = [
            Tool.model_validate(
                {
                    "type": "function",
                    "function": {"name": "record", "parameters": parameters},
                }
            )
        ]
        config = Qwen3CoderDetector()._get_arguments_config("record", tools)
        assert "value" in config
