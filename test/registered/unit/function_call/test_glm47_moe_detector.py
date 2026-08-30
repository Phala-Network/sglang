"""Focused regressions for GLM-4.7/GLM-5 tool-call argument parsing."""

import json
import unittest
from typing import Any

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1.0, suite="base-a-test-cpu")


def _make_tool(name: str, properties: dict[str, Any]) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            parameters={"type": "object", "properties": properties},
        ),
    )


def _make_tool_call(name: str, pairs: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
        for key, value in pairs
    )
    return f"<tool_call>{name}{body}</tool_call>"


def _stream_calls(
    text: str, tools: list[Tool], chunk_size: int
) -> list[tuple[int, str, str]]:
    detector = Glm47MoeDetector()
    accumulated: dict[int, list[str]] = {}
    order: list[int] = []
    chunks = [
        text[offset : offset + chunk_size] for offset in range(0, len(text), chunk_size)
    ]
    for chunk in chunks + ["", ""]:
        for call in detector.parse_streaming_increment(chunk, tools).calls:
            if call.tool_index not in accumulated:
                accumulated[call.tool_index] = ["", ""]
                order.append(call.tool_index)
            if call.name:
                accumulated[call.tool_index][0] = call.name
            if call.parameters:
                accumulated[call.tool_index][1] += call.parameters
    return [
        (tool_index, accumulated[tool_index][0], accumulated[tool_index][1])
        for tool_index in order
    ]


class TestGlm47NullableArguments(CustomTestCase):
    CASES = (
        (
            "nullable-null",
            {"enum": ["value", None], "type": ["string", "null"]},
            "null",
            None,
        ),
        ("string-literal-null", {"type": "string"}, "null", "null"),
        (
            "nullable-string",
            {"enum": ["value", None], "type": ["string", "null"]},
            "value",
            "value",
        ),
        (
            "anyof-nullable-null",
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "null",
            None,
        ),
        (
            "anyof-nullable-string",
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "value",
            "value",
        ),
        (
            "oneof-nullable-null",
            {"oneOf": [{"type": "string"}, {"type": "null"}]},
            "null",
            None,
        ),
    )

    def test_non_streaming_preserves_nullable_semantics(self) -> None:
        for case_name, value_schema, wire_value, expected_value in self.CASES:
            with self.subTest(case=case_name):
                tools = [_make_tool("record_value", {"value": value_schema})]
                text = _make_tool_call("record_value", [("value", wire_value)])
                result = Glm47MoeDetector().detect_and_parse(text, tools)
                self.assertEqual(len(result.calls), 1)
                self.assertEqual(
                    json.loads(result.calls[0].parameters),
                    {"value": expected_value},
                )

    def test_fragmented_streaming_preserves_nullable_semantics(self) -> None:
        for case_name, value_schema, wire_value, expected_value in self.CASES:
            tools = [_make_tool("record_value", {"value": value_schema})]
            text = _make_tool_call("record_value", [("value", wire_value)])
            for chunk_size in (1, 3, 11):
                with self.subTest(case=case_name, chunk_size=chunk_size):
                    calls = _stream_calls(text, tools, chunk_size)
                    self.assertEqual(
                        [(index, name) for index, name, _ in calls],
                        [(0, "record_value")],
                    )
                    self.assertEqual(
                        json.loads(calls[0][2]),
                        {"value": expected_value},
                    )


class TestGlm47StreamingParity(CustomTestCase):
    TOOLS = [
        _make_tool(
            "typed",
            {
                "s": {"type": "string"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "o": {"type": "object"},
                "a": {"type": "array"},
            },
        ),
        _make_tool("untyped", {}),
    ]

    CASES = (
        ("typed-string", "typed", [("s", "hello")]),
        ("typed-number", "typed", [("n", "42")]),
        ("typed-negative-number", "typed", [("n", "-7")]),
        ("typed-float", "typed", [("n", "3.14")]),
        ("typed-boolean", "typed", [("b", "true")]),
        ("typed-object", "typed", [("o", '{"k": 1}')]),
        ("typed-array", "typed", [("a", "[1, 2]")]),
        ("typed-empty-string", "typed", [("s", "")]),
        ("typed-multiple", "typed", [("s", "abc"), ("n", "5")]),
        ("untyped-int", "untyped", [("x", "123")]),
        ("untyped-negative-int", "untyped", [("x", "-123")]),
        ("untyped-float", "untyped", [("x", "1.5")]),
        ("untyped-bool", "untyped", [("x", "true")]),
        ("untyped-null", "untyped", [("x", "null")]),
        ("untyped-text", "untyped", [("x", "hello world")]),
        ("untyped-alphanumeric", "untyped", [("x", "123abc")]),
        ("untyped-object", "untyped", [("x", '{"k": 1}')]),
        ("untyped-array", "untyped", [("x", "[1, 2]")]),
        ("untyped-empty", "untyped", [("x", "")]),
        ("untyped-unicode", "untyped", [("x", "北京")]),
        ("untyped-mixed", "untyped", [("x", "7"), ("y", "text")]),
    )

    def test_fragmented_stream_matches_non_streaming(self) -> None:
        for case_name, func_name, pairs in self.CASES:
            text = _make_tool_call(func_name, pairs)
            final = Glm47MoeDetector().detect_and_parse(text, self.TOOLS)
            self.assertEqual(len(final.calls), 1, case_name)
            expected = json.loads(final.calls[0].parameters)
            for chunk_size in (1, 2, 7, 19):
                with self.subTest(case=case_name, chunk_size=chunk_size):
                    calls = _stream_calls(text, self.TOOLS, chunk_size)
                    self.assertEqual(
                        [(index, name) for index, name, _ in calls], [(0, func_name)]
                    )
                    self.assertEqual(json.loads(calls[0][2]), expected)

    def test_regression_types_are_exact(self) -> None:
        expected_values = {
            "number": 123,
            "object": {"k": 1},
            "alphanumeric": "123abc",
            "null": None,
        }
        cases = {
            "number": "123",
            "object": '{"k": 1}',
            "alphanumeric": "123abc",
            "null": "null",
        }
        for case_name, wire_value in cases.items():
            text = _make_tool_call("untyped", [("x", wire_value)])
            streamed = _stream_calls(text, self.TOOLS, 3)
            final = Glm47MoeDetector().detect_and_parse(text, self.TOOLS)
            with self.subTest(case=case_name):
                self.assertEqual(
                    json.loads(final.calls[0].parameters)["x"],
                    expected_values[case_name],
                )
                self.assertEqual(
                    json.loads(streamed[0][2])["x"], expected_values[case_name]
                )

    def test_repeated_same_function_keeps_sequential_indexes(self) -> None:
        text = _make_tool_call("untyped", [("x", "123")]) + _make_tool_call(
            "untyped", [("x", '{"k": 1}')]
        )
        final = Glm47MoeDetector().detect_and_parse(text, self.TOOLS)
        streamed = _stream_calls(text, self.TOOLS, 3)
        self.assertEqual([call.tool_index for call in final.calls], [0, 1])
        self.assertEqual([index for index, _, _ in streamed], [0, 1])
        self.assertEqual(
            [json.loads(call.parameters) for call in final.calls],
            [{"x": 123}, {"x": {"k": 1}}],
        )
        self.assertEqual(
            [json.loads(arguments) for _, _, arguments in streamed],
            [{"x": 123}, {"x": {"k": 1}}],
        )


if __name__ == "__main__":
    unittest.main()
