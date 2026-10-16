"""CPU-only regressions for complete Xing4 calls, not malformed-output recovery.

Exercise real detectors and the public parser wrapper, without a server or GPU.
Streaming indices are response ordinals; nonstream indices remain definition IDs.
"""

import json
import random
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.xing4_0_detector import Xing4_0Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1, "base-a-test-cpu")


def tool(name, properties):
    return Tool(
        type="function",
        function=Function(
            name=name,
            parameters={
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
        ),
    )


def tag_arguments(name, arguments):
    return (
        f"<tool_call>{name}"
        + "".join(
            f"<param_key>{key}</param_key><param_value>{value}</param_value>"
            for key, value in arguments.items()
        )
        + "</tool_call>"
    )


def tag_call(name, value):
    return tag_arguments(name, {"value": value})


def json_call(name, value, *, argument_key="arguments"):
    return (
        "<tool_call>"
        + json.dumps(
            {"name": name, argument_key: {"value": value}}, ensure_ascii=False
        )
        + "</tool_call>"
    )


def name_json_call(name, value):
    return (
        f"<tool_call>{name}"
        + json.dumps({"value": value}, ensure_ascii=False)
        + "</tool_call>"
    )


def chunkings(text, exhaustive=False):
    """Fixed, reproducibly irregular, and optionally every two-chunk split.

    Splits are character offsets, matching the parser API's decoded str input.
    Empty chunks at the ends also exercise no-op increments.
    """
    for size in (1, 2, 3, 7, 31, len(text)):
        yield f"size={size}", [
            text[start : start + size] for start in range(0, len(text), size)
        ]
    for seed in range(4):
        rng = random.Random(seed)
        chunks, start = [""], 0
        while start < len(text):
            end = start + rng.randint(1, 97)
            chunks.extend((text[start:end], ""))
            start = end
        yield f"irregular={seed}", chunks
    if exhaustive:
        for split in range(len(text) + 1):
            yield f"split={split}", [text[:split], text[split:]]


class TestXing4ToolStreaming(unittest.TestCase):
    def setUp(self):
        self.tools = [
            tool(name, {"value": {"type": "string"}}) for name in ("read", "bash")
        ]

    def stream(self, chunks, tools=None, wrapper=False):
        definitions = self.tools if tools is None else tools
        parser = (
            FunctionCallParser(definitions, "xing4_0")
            if wrapper
            else Xing4_0Detector()
        )
        calls, content = [], []
        for chunk in chunks:
            if wrapper:
                normal, emitted = parser.parse_stream_chunk(chunk)
            else:
                result = parser.parse_streaming_increment(chunk, definitions)
                normal, emitted = result.normal_text, result.calls
            calls.extend(emitted)
            content.append(normal or "")
        if wrapper:
            normal, emitted = parser.parse_stream_end()
        else:
            result = parser.finish(definitions)
            normal, emitted = result.normal_text, result.calls
        calls.extend(emitted)
        content.append(normal or "")
        return "".join(content), calls

    def assert_call_items(self, calls, expected, indices=None):
        if indices is None:
            indices = list(range(len(expected)))
        self.assertEqual([call.tool_index for call in calls], indices)
        self.assertEqual(
            [(call.name, json.loads(call.parameters)) for call in calls], expected
        )

    def assert_streams(
        self, text, expected, *, tools=None, content="", exhaustive=False, wrapper=False
    ):
        for label, chunks in chunkings(text, exhaustive):
            with self.subTest(chunks=label, wrapper=wrapper):
                normal, calls = self.stream(chunks, tools, wrapper)
                self.assertEqual(normal, content)
                self.assert_call_items(calls, expected)
                # Emulate clients concatenating argument deltas by SSE index.
                # A repeated definition index would concatenate distinct JSON
                # objects and fail instead of yielding separate argument dicts.
                assembled = {}
                for call in calls:
                    assembled.setdefault(call.tool_index, "")
                    assembled[call.tool_index] += call.parameters
                self.assertEqual(
                    [json.loads(assembled[i]) for i in range(len(expected))],
                    [args for _, args in expected],
                )

    def assert_nonstream(self, text, expected, tools=None, wrapper=False):
        definitions = self.tools if tools is None else tools
        if wrapper:
            _, calls = FunctionCallParser(definitions, "xing4_0").parse_non_stream(text)
        else:
            calls = Xing4_0Detector().detect_and_parse(text, definitions).calls
        definition_indices = {
            definition.function.name: index
            for index, definition in enumerate(definitions)
        }
        self.assert_call_items(
            calls, expected, [definition_indices[name] for name, _ in expected]
        )

    def test_second_defined_tool_starts_at_zero(self):
        self.assert_streams(
            tag_call("bash", "one"), [("bash", {"value": "one"})], exhaustive=True
        )

    def test_many_repeated_calls_are_distinct(self):
        for encode in (tag_call, json_call):
            with self.subTest(format=encode.__name__):
                values = [f"  call {i}: 雪 🦑\n" for i in range(40)]
                text = "".join(encode("bash", value) for value in values)
                expected = [("bash", {"value": value}) for value in values]
                self.assert_streams(text, expected)
                self.assert_nonstream(text, expected)

    def mixed_response(self, count):
        formats = (tag_call, json_call, name_json_call)
        expected, blocks = [], []
        for i in range(count):
            name = ("bash", "bash", "read", "bash", "read")[i % 5]
            value = f"  {i}: café 雪 🦑\n\t"
            blocks.append(formats[i % len(formats)](name, value))
            expected.append((name, {"value": value}))
        # Only the streaming path promises to preserve inter-call/trailing text.
        return "before\n" + "\n".join(blocks) + "\nafter", expected

    def test_mixed_formats_at_every_two_chunk_boundary(self):
        text, expected = self.mixed_response(4)
        self.assert_streams(
            text, expected, content="before\n" + "\n" * 4 + "after", exhaustive=True
        )
        self.assert_nonstream(text, expected)

    def test_many_mixed_calls_across_irregular_boundaries(self):
        text, expected = self.mixed_response(45)
        self.assert_streams(text, expected, content="before\n" + "\n" * 45 + "after")
        self.assert_nonstream(text, expected)

    def test_json_parameters_alias_preserves_arguments(self):
        value = " \t雪\n "
        text = json_call("bash", value, argument_key="parameters")
        expected = [("bash", {"value": value})]
        self.assert_streams(text, expected, exhaustive=True)
        self.assert_nonstream(text, expected)

    def test_unknown_names_do_not_consume_indices_or_get_remapped(self):
        unknown = [
            tag_call("Bash", "case mismatch"),
            json_call("missing", "not defined"),
            tag_call("bas", "prefix"),
            json_call("bash_extra", "suffix"),
        ]
        # Unknowns before, between, and after valid calls must not create gaps.
        text = (
            unknown[0]
            + tag_call("bash", "first")
            + unknown[1]
            + unknown[2]
            + json_call("read", "second")
            + unknown[3]
            + name_json_call("bash", "third")
            + unknown[0]
        )
        expected = [
            ("bash", {"value": "first"}),
            ("read", {"value": "second"}),
            ("bash", {"value": "third"}),
        ]
        with self.assertLogs(
            "sglang.srt.function_call.xing4_0_detector", level="WARNING"
        ):
            self.assert_streams(text, expected, content="".join(unknown + unknown[:1]))
            self.assert_nonstream(text, expected)
            self.assert_streams(
                unknown[0] + unknown[1], [], content=unknown[0] + unknown[1]
            )

    def test_independent_responses_do_not_share_buffers_or_ordinals(self):
        # Interleave real wrappers: response B must not finish A's partial call
        # or inherit A's count, even with the same tool definitions.
        a = FunctionCallParser(self.tools, "xing4_0")
        b = FunctionCallParser(self.tools, "xing4_0")
        self.assertIsNot(a.detector, b.detector)
        first = tag_call("bash", "response A, first")
        next_call = json_call("read", "response A, second")
        normal, calls = a.parse_stream_chunk(first + next_call[:17])
        self.assertEqual(normal, "")
        self.assert_call_items(calls, [("bash", {"value": "response A, first"})])
        normal, calls = b.parse_stream_chunk(tag_call("bash", "response B"))
        self.assertEqual(normal, "")
        self.assert_call_items(calls, [("bash", {"value": "response B"})])
        self.assertEqual(b.parse_stream_end(), ("", []))
        normal, calls = a.parse_stream_chunk(next_call[17:])
        self.assertEqual(normal, "")
        self.assert_call_items(calls, [("read", {"value": "response A, second"})], [1])
        self.assertEqual(a.parse_stream_end(), ("", []))
        c = FunctionCallParser(self.tools, "xing4_0")
        _, calls = c.parse_stream_chunk(tag_call("bash", "response C"))
        self.assert_call_items(calls, [("bash", {"value": "response C"})])

    def test_nonstream_definition_indices_do_not_advance_stream_state(self):
        detector = Xing4_0Detector()
        text = (
            tag_call("bash", "one")
            + json_call("bash", "two")
            + tag_call("read", "three")
        )
        expected = [
            ("bash", {"value": "one"}),
            ("bash", {"value": "two"}),
            ("read", {"value": "three"}),
        ]
        for ordinal in range(3):
            self.assert_call_items(
                detector.detect_and_parse(text, self.tools).calls, expected, [1, 1, 0]
            )
            self.assert_call_items(
                detector.parse_streaming_increment(
                    tag_call("bash", "stream"), self.tools
                ).calls,
                [("bash", {"value": "stream"})],
                [ordinal],
            )

    def test_string_values_remain_literal_including_whitespace_and_unicode(self):
        values = (
            "",
            " ",
            "\t\n  \r\n",
            "  first line\n\tsecond line  \n",
            "\u2003雪 café e\u0301 🦑\u00a0",
            " 42 ",
            "true",
            "null",
            '[1, "雪"]',
            '{"nested": true}',
            '  "quoted"  ',
            r"C:\tmp\file\n",
            "'python literal'",
        )
        for value in values:
            for encode in (tag_call, json_call, name_json_call):
                with self.subTest(value=repr(value), format=encode.__name__):
                    text = encode("bash", value)
                    expected = [("bash", {"value": value})]
                    self.assert_streams(text, expected)
                    self.assert_nonstream(text, expected)

    def test_string_type_unions_preserve_raw_tag_values(self):
        for union in (["string", "null"], ["null", "string"], ["integer", "string"]):
            tools = [tool("union", {"value": {"type": union}})]
            for value in ("", " \n\t", " 42 ", "null", '  "雪 🦑"\n'):
                with self.subTest(type=union, value=repr(value)):
                    text = tag_call("union", value)
                    expected = [("union", {"value": value})]
                    self.assert_streams(text, expected, tools=tools)
                    self.assert_nonstream(text, expected, tools=tools)

    def test_non_string_values_still_deserialize(self):
        properties = {
            "integer": {"type": "integer"},
            "number": {"type": "number"},
            "enabled": {"type": "boolean"},
            "disabled": {"type": "boolean"},
            "items": {"type": "array"},
            "object": {"type": "object"},
            "nullable": {"type": ["integer", "null"]},
            "literal": {"type": "string"},
        }
        tools = [tool("typed", properties)]
        raw = {
            "integer": " 42 \n",
            "number": " \t-1.25e2\n",
            "enabled": " true ",
            "disabled": " false\n",
            "items": ' [1, true, null, {"message": " 雪 "}] \n',
            "object": ' {"nested": [false, 2.5], "empty": ""} ',
            "nullable": " null ",
            "literal": " 42 \n",
        }
        arguments = {
            "integer": 42,
            "number": -125.0,
            "enabled": True,
            "disabled": False,
            "items": [1, True, None, {"message": " 雪 "}],
            "object": {"nested": [False, 2.5], "empty": ""},
            "nullable": None,
            "literal": " 42 \n",
        }
        expected = [("typed", arguments)]
        for text in (
            tag_arguments("typed", raw),
            "<tool_call>"
            + json.dumps({"name": "typed", "arguments": arguments})
            + "</tool_call>",
        ):
            with self.subTest(format=text[:30]):
                self.assert_streams(text, expected, tools=tools)
                self.assert_nonstream(text, expected, tools=tools)
        # Check actual Python types too: bool compares equal to integer 0/1.
        calls = (
            Xing4_0Detector()
            .detect_and_parse(tag_arguments("typed", raw), tools)
            .calls
        )
        parsed = json.loads(calls[0].parameters)
        for key, expected_type in (
            ("integer", int),
            ("number", float),
            ("enabled", bool),
            ("disabled", bool),
            ("items", list),
            ("object", dict),
            ("literal", str),
        ):
            self.assertIs(type(parsed[key]), expected_type)

    def test_existing_python_literal_deserialization_is_preserved(self):
        tools = [tool("typed", {"value": {"type": "object"}})]
        text = tag_call("typed", "  {'ready': True, 'items': [False, None, ' 雪 ']} \n")
        expected = [
            ("typed", {"value": {"ready": True, "items": [False, None, " 雪 "]}})
        ]
        self.assert_streams(text, expected, tools=tools)
        self.assert_nonstream(text, expected, tools=tools)

    def test_function_call_parser_wrapper_streaming_and_nonstream(self):
        text, expected = self.mixed_response(6)
        self.assert_streams(
            text,
            expected,
            content="before\n" + "\n" * 6 + "after",
            exhaustive=True,
            wrapper=True,
        )
        self.assert_nonstream(text, expected, wrapper=True)
        unknown = json_call("Bash", "not bash")
        with self.assertLogs(
            "sglang.srt.function_call.xing4_0_detector", level="WARNING"
        ):
            self.assert_streams(
                unknown + tag_call("bash", " \n雪\t "),
                [("bash", {"value": " \n雪\t "})],
                content=unknown,
                wrapper=True,
            )


if __name__ == "__main__":
    unittest.main()
