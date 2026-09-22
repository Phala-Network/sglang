"""Execute real Qwen methods and schema helpers without the GPU serving imports.

Only protocol data envelopes and the environment getter are substituted. Schema
validation uses the real jsonschema library, not a fake validator. These tests
do not qualify native grammar compilation, SSE framing, IDs or finish_reason.
"""

import ast
import copy
import json
import logging
import math
import re
import threading
import unittest
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from jsonschema import Draft202012Validator
from referencing import Registry
from referencing.exceptions import NoSuchResource

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/function_call"


@dataclass
class ToolCallItem:
    tool_index: int
    parameters: str
    name: str | None = None


@dataclass
class StreamingParseResult:
    normal_text: str = ""
    calls: list = field(default_factory=list)


def load_detector(forward_unknown=False, source_override=None):
    namespace = dict(
        ast=ast,
        json=json,
        logging=logging,
        math=math,
        re=re,
        threading=threading,
        warnings=warnings,
        Draft202012Validator=Draft202012Validator,
        Registry=Registry,
        NoSuchResource=NoSuchResource,
        ToolCallItem=ToolCallItem,
        StreamingParseResult=StreamingParseResult,
        logger=logging.getLogger(__name__),
        envs=SimpleNamespace(
            SGLANG_FORWARD_UNKNOWN_TOOLS=SimpleNamespace(get=lambda: forward_unknown)
        ),
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    tree = ast.parse((SOURCE / "utils.py").read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign))
    ]
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(SOURCE / "utils.py"),
            "exec",
        ),
        namespace,
    )
    tree = ast.parse((SOURCE / "base_format_detector.py").read_text(encoding="utf-8"))
    base = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseFormatDetector"
    )
    base.bases = []
    base.body = [
        node
        for node in base.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    source = source_override or (SOURCE / "qwen3_coder_detector.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    detector = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3CoderDetector"
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, base, detector], type_ignores=[])
            ),
            str(SOURCE / "qwen3_coder_detector.py"),
            "exec",
        ),
        namespace,
    )
    return namespace["Qwen3CoderDetector"]


def tool(name="probe", parameters=None):
    return SimpleNamespace(
        type="function",
        function=SimpleNamespace(name=name, parameters=parameters or {}),
    )


def call(name="probe", body=""):
    return f"<tool_call><function={name}>{body}</function></tool_call>"


def parameter(name, value):
    return f"<parameter={name}>\n{value}\n</parameter>"


def signature(result):
    return [
        (item.tool_index, item.name, json.loads(item.parameters))
        for item in result.calls
    ]


class QwenCompleteCallTests(unittest.TestCase):
    def setUp(self):
        self.subject = load_detector()
        self.tools = [tool()]

    def stream(self, text, size, tools=None):
        detector = self.subject()
        normal, calls = [], []
        for start in range(0, len(text), size):
            result = detector.parse_streaming_increment(
                text[start : start + size], tools or self.tools
            )
            normal.append(result.normal_text)
            calls.extend(result.calls)
        tail = detector.finish(tools or self.tools)
        normal.append(tail.normal_text)
        calls.extend(tail.calls)
        self.assertEqual(detector.finish(self.tools).calls, [])
        self.assertEqual(detector.finish(self.tools).normal_text, "")
        return StreamingParseResult("".join(normal), calls)

    def test_every_truncation_before_outer_close_emits_no_call(self):
        text = call(body=parameter("message", 'hello "world"'))
        for end in range(len("<tool_call>"), len(text)):
            with self.subTest(end=end):
                partial = text[:end]
                self.assertEqual(
                    self.subject().detect_and_parse(partial, self.tools).calls, []
                )
                parsed = self.stream(partial, 1)
                self.assertEqual(parsed.calls, [])
                self.assertEqual(parsed.normal_text, "")

    def test_const_intersects_union_in_shared_complete_call_executor(self):
        cases = [
            ({"oneOf": [{"type": "string"}, {"type": "integer"}], "const": 7}, "7", 7),
            ({"type": ["string", "integer"], "const": 8}, "8", 8),
            ({"enum": ["alpha", 2, False], "const": False}, "false", False),
            ({"enum": ["true", True, 1], "const": True}, "true", True),
            ({"type": ["string", "object"], "const": {"x": 1}}, '{"x":1}', {"x": 1}),
            ({"type": ["string", "array"], "const": [2]}, "[2]", [2]),
            ({"type": ["string", "integer"], "const": "007"}, "007", "007"),
            ({"type": ["integer", "null"], "const": None}, "null", None),
        ]
        for schema, raw, expected in cases:
            parameters = {"type": "object", "properties": {"value": schema}}
            before = copy.deepcopy(parameters)
            tools = [tool(parameters=parameters)]
            text = call(body=parameter("value", raw))
            for result in (
                self.subject().detect_and_parse(text, tools),
                self.stream(text, 1, tools),
            ):
                value = signature(result)[0][2]["value"]
                self.assertEqual(value, expected)
                self.assertIs(type(value), type(expected))
            self.assertEqual(parameters, before)
        # Conversion never replaces the generated value with the constant.
        tools = [tool(parameters={"properties": {"value": {"const": 7}}})]
        text = call(body=parameter("value", "8"))
        self.assertEqual(
            signature(self.subject().detect_and_parse(text, tools))[0][2], {"value": 8}
        )
        self.assertEqual(signature(self.stream(text, 1, tools))[0][2], {"value": 8})

    def test_every_two_chunk_split_is_atomic_and_complete(self):
        text = call()
        for split in range(1, len(text)):
            with self.subTest(split=split):
                detector = self.subject()
                self.assertEqual(
                    detector.parse_streaming_increment(text[:split], self.tools).calls,
                    [],
                )
                result = detector.parse_streaming_increment(text[split:], self.tools)
                self.assertEqual(signature(result), [(0, "probe", {})])
                self.assertEqual(detector.finish(self.tools).calls, [])

    def test_unknown_rejected_in_both_modes_without_consuming_index(self):
        text = call("missing") + call()
        self.assertEqual(
            signature(self.subject().detect_and_parse(text, self.tools)),
            [(0, "probe", {})],
        )
        for size in (1, 7, 64):
            self.assertEqual(signature(self.stream(text, size)), [(0, "probe", {})])

    def test_explicit_unknown_forwarding_compatibility(self):
        detector = load_detector(forward_unknown=True)
        self.assertEqual(
            signature(detector().detect_and_parse(call("missing"), self.tools)),
            [(0, "missing", {})],
        )
        self.assertEqual(
            signature(
                detector().parse_streaming_increment(call("missing"), self.tools)
            ),
            [(0, "missing", {})],
        )

    def test_empty_or_prefix_names_never_match(self):
        for name in ("", "pro", "probe ", "probe_extra"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.subject().detect_and_parse(call(name), self.tools).calls, []
                )
                self.assertEqual(self.stream(call(name), 1).calls, [])

    def test_missing_function_end_in_closed_block_is_not_a_call(self):
        text = "<tool_call><function=probe><parameter=x>1</parameter></tool_call>"
        self.assertEqual(self.subject().detect_and_parse(text, self.tools).calls, [])
        self.assertEqual(self.stream(text, 1).calls, [])

    def test_closed_first_function_does_not_validate_incomplete_second(self):
        text = "<tool_call><function=probe></function><function=probe></tool_call>"
        self.assertEqual(
            signature(self.subject().detect_and_parse(text, self.tools)),
            [(0, "probe", {})],
        )
        self.assertEqual(signature(self.stream(text, 1)), [(0, "probe", {})])

    def test_parallel_and_repeated_calls_have_contiguous_indexes(self):
        tools = [tool(), tool("second")]
        text = call() + "\n" + call("second") + "\n" + call()
        expected = [(0, "probe", {}), (1, "second", {}), (2, "probe", {})]
        self.assertEqual(
            signature(self.subject().detect_and_parse(text, tools)), expected
        )
        for size in (1, 2, 7, 64, len(text)):
            self.assertEqual(signature(self.stream(text, size, tools)), expected)

    def test_two_functions_in_one_block_are_both_drained(self):
        text = "<tool_call><function=probe></function><function=probe></function></tool_call>"
        expected = [(0, "probe", {}), (1, "probe", {})]
        self.assertEqual(
            signature(self.subject().detect_and_parse(text, self.tools)), expected
        )
        self.assertEqual(signature(self.stream(text, len(text))), expected)

    def test_orphan_parameters_never_emit_argument_only_delta(self):
        for text in (
            "<tool_call><parameter=x>1</parameter></function></tool_call>",
            "<parameter=x>1</parameter></function>",
        ):
            self.assertEqual(
                self.subject().detect_and_parse(text, self.tools).calls, []
            )
            self.assertEqual(self.stream(text, 1).calls, [])

    def test_complete_call_survives_truncated_followup(self):
        text = call() + "\n<tool_call><function=probe>"
        self.assertEqual(
            signature(self.subject().detect_and_parse(text, self.tools)),
            [(0, "probe", {})],
        )
        self.assertEqual(signature(self.stream(text, 1)), [(0, "probe", {})])

    def test_text_and_partial_marker_flush_parity(self):
        for text in ("plain final answer", "a < b <tool", "before" + call() + "after"):
            with self.subTest(text=text):
                expected = self.subject().detect_and_parse(text, self.tools)
                for size in (1, 7, 64):
                    parsed = self.stream(text, size)
                    self.assertEqual(parsed.normal_text, expected.normal_text)
                    self.assertEqual(signature(parsed), signature(expected))

    def test_refs_unions_enum_const_nested_and_large_string(self):
        schema = {
            "$defs": {
                "record": {"type": "object", "properties": {"n": {"type": "integer"}}}
            },
            "type": "object",
            "properties": {
                "payload": {"$ref": "#/$defs/record"},
                "nullable": {"type": ["integer", "null"]},
                "enum": {"enum": ["ready", 2]},
                "const": {"const": 7},
                "message": {"type": "string"},
                "path": {"type": "string"},
                "regex": {"type": "string"},
                "array": {"type": "array"},
            },
        }
        message = 'Unicode 北京 "quoted" \\\\ ' * 256
        values = {
            "payload": '{"n": 7}',
            "nullable": "null",
            "enum": "2",
            "const": "7",
            "message": message,
            "path": r"C:\data\file.txt",
            "regex": r"^\d+_[a-z]+$",
            "array": '[true, null, {"a": 1}]',
        }
        text = call(
            body="".join(parameter(key, value) for key, value in values.items())
        )
        tools = [tool(parameters=schema)]
        expected = [
            (
                0,
                "probe",
                {
                    "payload": {"n": 7},
                    "nullable": None,
                    "enum": 2,
                    "const": 7,
                    "message": message,
                    "path": r"C:\data\file.txt",
                    "regex": r"^\d+_[a-z]+$",
                    "array": [True, None, {"a": 1}],
                },
            )
        ]
        self.assertEqual(
            signature(self.subject().detect_and_parse(text, tools)), expected
        )
        for size in (1, 7, 64, len(text)):
            self.assertEqual(signature(self.stream(text, size, tools)), expected)

    def test_empty_arguments_and_lenient_parameter_end_remain_supported(self):
        for body, expected in (
            ("", {}),
            ("<parameter=x>value", {"x": "value"}),
            ("<parameter=x>1<parameter=y>2", {"x": "1", "y": "2"}),
        ):
            with self.subTest(body=body):
                text = call(body=body)
                self.assertEqual(
                    signature(self.subject().detect_and_parse(text, self.tools)),
                    [(0, "probe", expected)],
                )
                self.assertEqual(
                    signature(self.stream(text, 1)), [(0, "probe", expected)]
                )

    def test_no_tools_does_not_authorize_unknown_name(self):
        self.assertEqual(self.subject().detect_and_parse(call(), []).calls, [])
        self.assertEqual(self.subject().parse_streaming_increment(call(), []).calls, [])

    def test_nonstream_reuse_does_not_carry_stream_indexes(self):
        detector = self.subject()
        detector.parse_streaming_increment(call(), self.tools)
        self.assertEqual(
            signature(detector.detect_and_parse(call(), self.tools)), [(0, "probe", {})]
        )


if __name__ == "__main__":
    unittest.main()
