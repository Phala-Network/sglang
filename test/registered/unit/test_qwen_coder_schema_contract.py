"""Regressions from the real Qwen3.8 tool failures on the test CVM."""
import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.schema_argument_coercion import coerce_argument_to_schema


def tools_for(properties, definitions=None):
    schema = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    if definitions:
        schema["$defs"] = definitions
    return [Tool.model_validate({"type": "function", "function": {"name": "record", "parameters": schema, "strict": True}})]


class QwenCoderSchemaContract(unittest.TestCase):
    def check_roundtrip(self, properties, raw_values, expected, definitions=None):
        tools = tools_for(properties, definitions)
        text = "<tool_call>\n<function=record>\n" + "".join(
            f"<parameter={key}>{value}</parameter>\n" for key, value in raw_values.items()
        ) + "</function>\n</tool_call>"
        output = Qwen3CoderDetector().detect_and_parse(text, tools)
        self.assertEqual(len(output.calls), 1)
        self.assertEqual(json.loads(output.calls[0].parameters), expected)
        for size in (1, 3, 11, 64):
            detector = Qwen3CoderDetector()
            parameters = ""
            names = []
            for start in range(0, len(text), size):
                parsed = detector.parse_streaming_increment(text[start:start + size], tools)
                for call in parsed.calls:
                    if call.name:
                        names.append(call.name)
                    parameters += call.parameters or ""
            self.assertEqual(names, ["record"])
            self.assertEqual(json.loads(parameters), expected)

    def test_ref_object_in_nonstream_and_stream(self):
        self.check_roundtrip(
            {"location": {"$ref": "#/$defs/Location"}},
            {"location": '{"lat":48.8566,"lon":2.3522}'},
            {"location": {"lat": 48.8566, "lon": 2.3522}},
            {"Location": {"type": "object", "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}}, "required": ["lat", "lon"], "additionalProperties": False}},
        )

    def test_heterogeneous_union_enum_and_const(self):
        self.check_roundtrip(
            {"payload": {"oneOf": [{"type": "string"}, {"type": "integer"}], "const": 7}, "selector": {"enum": ["alpha", 2, False], "const": False}, "retry": {"type": ["integer", "null"], "const": None}},
            {"payload": "7", "selector": "false", "retry": "null"},
            {"payload": 7, "selector": False, "retry": None},
        )

    def test_string_literals_and_underscore_ids_are_preserved(self):
        self.check_roundtrip(
            {"id": {"type": "string"}, "literal": {"type": "string"}},
            {"id": "10220_3939392", "literal": "null"},
            {"id": "10220_3939392", "literal": "null"},
        )

    def test_const_never_supplies_an_unemitted_value(self):
        actual, valid = coerce_argument_to_schema("999", {"type": "integer", "const": 7})
        self.assertFalse(valid)
        self.assertEqual(actual, "999")

    def test_external_refs_never_fetch(self):
        actual, valid = coerce_argument_to_schema("{}", {"$ref": "https://invalid.example/schema"})
        self.assertFalse(valid)
        self.assertEqual(actual, "{}")

    def test_two_complete_calls_keep_distinct_indexes(self):
        tools = tools_for({"value": {"type": "integer"}})
        def block(value):
            return f"<tool_call>\n<function=record>\n<parameter=value>{value}</parameter>\n</function>\n</tool_call>"
        parsed = Qwen3CoderDetector().detect_and_parse(block(1) + "\n" + block(2), tools)
        self.assertEqual([c.tool_index for c in parsed.calls], [0, 1])
        self.assertEqual([json.loads(c.parameters) for c in parsed.calls], [{"value": 1}, {"value": 2}])

    def test_truncation_never_publishes_a_partial_call(self):
        tools = tools_for({"value": {"type": "integer"}})
        text = "<tool_call>\n<function=record>\n<parameter=value>1</parameter>\n</function>\n</tool_call>"
        for cut in range(text.index("<function="), len(text)):
            detector = Qwen3CoderDetector()
            self.assertEqual(detector.parse_streaming_increment(text[:cut], tools).calls, [])
            final = detector.finish(tools)
            self.assertEqual(final.calls, [])
            self.assertEqual(final.normal_text, "")
            self.assertEqual(Qwen3CoderDetector().detect_and_parse(text[:cut], tools).calls, [])

    def test_unknown_call_does_not_consume_a_known_call_index(self):
        tools = tools_for({"value": {"type": "integer"}})
        unknown = "<tool_call><function=missing><parameter=value>1</parameter></function></tool_call>"
        known = "<tool_call><function=record><parameter=value>2</parameter></function></tool_call>"
        detector = Qwen3CoderDetector()
        parsed = detector.parse_streaming_increment(unknown + "\n" + known, tools)
        self.assertEqual([(c.tool_index, c.name) for c in parsed.calls], [(0, "record")])


if __name__ == "__main__":
    unittest.main()
