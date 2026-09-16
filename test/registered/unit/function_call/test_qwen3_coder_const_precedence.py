"""Constrained literal types survive XML tool parsing and stream boundaries."""
import copy
import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector


class TestConstPrecedence(unittest.TestCase):
    def parse(self, schema, raw, streaming):
        parameters = {"type": "object", "properties": {"value": schema}}
        before = copy.deepcopy(parameters)
        tools = [Tool.model_validate({"type": "function", "function": {
            "name": "record", "parameters": parameters}})]
        detector = Qwen3CoderDetector(require_complete_calls=True)
        text = ("<tool_call><function=record><parameter=value>" + raw
                + "</parameter></function></tool_call>")
        if streaming:
            calls = []
            for char in text:
                calls.extend(detector.parse_streaming_increment(char, tools).calls)
            encoded = "".join(c.parameters or "" for c in calls)
        else:
            calls = detector.detect_and_parse(text, tools).calls
            self.assertEqual(len(calls), 1)
            encoded = calls[0].parameters
        self.assertEqual(tools[0].function.parameters, before)
        return json.loads(encoded)["value"]

    def test_const_narrows_union_and_enum(self):
        cases = [
            ({"oneOf": [{"type": "string"}, {"type": "integer"}], "const": 7}, "7", 7),
            ({"anyOf": [{"type": "string"}, {"type": "number"}], "const": 1.25}, "1.25", 1.25),
            ({"type": ["string", "integer"], "const": 8}, "8", 8),
            ({"enum": ["alpha", 2, False], "const": False}, "false", False),
            ({"enum": ["true", True, 1], "const": True}, "true", True),
            ({"oneOf": [{"type": "string"}, {"type": "object"}], "const": {"x": 1}}, '{"x":1}', {"x": 1}),
            ({"anyOf": [{"type": "string"}, {"type": "array"}], "const": [2]}, "[2]", [2]),
            ({"oneOf": [{"type": "string"}, {"type": "integer"}], "const": "007"}, "007", "007"),
            ({"enum": [False, "false"], "const": "false"}, "false", "false"),
            ({"type": ["integer", "null"], "const": None}, "null", None),
        ]
        for streaming in (False, True):
            for schema, raw, expected in cases:
                with self.subTest(streaming=streaming, schema=schema):
                    actual = self.parse(schema, raw, streaming)
                    self.assertEqual(actual, expected)
                    self.assertIs(type(actual), type(expected))

    def test_conversion_never_substitutes_the_constant(self):
        schema = {"oneOf": [{"type": "string"}, {"type": "integer"}], "const": 7}
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.assertEqual(self.parse(schema, "8", streaming), 8)
                self.assertEqual(self.parse(schema, "not-a-number", streaming), "not-a-number")

    def test_unconstrained_union_and_string_ids_are_unchanged(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
                self.assertEqual(self.parse(schema, "007", streaming), "007")
                self.assertEqual(self.parse({"type": "string"}, "10220_3939392", streaming), "10220_3939392")


if __name__ == "__main__":
    unittest.main()
