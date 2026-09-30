"""Kimi schema prechecks against unchanged candidate methods or installed modules.

Default mode loads the real guard module and complete source methods with CPU
fixtures. PHALA_KIMI_INSTALLED=1 imports the real serving/protocol modules and
requires native XGrammar; missing dependencies fail rather than skip. No model
or GPU generation is performed. This is not end-to-end HTTP acceptance.
"""

import ast
import copy
import hashlib
import importlib.util
import json
import os
import signal
import unittest
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from jsonschema import Draft202012Validator, SchemaError

INSTALLED = os.environ.get("PHALA_KIMI_INSTALLED") == "1"


class ToolStrictLevel(IntEnum):
    OFF = 0
    FUNCTION = 1
    PARAMETER = 2


class ToolChoice(NS):
    pass


class AllowedToolsChoice(NS):
    pass


class ChatCompletionMessageGenericParam(NS):
    pass


if INSTALLED:
    signal.alarm(180)
    import sglang.srt.constrained.xgrammar_schema as guard
    import sglang.srt.entrypoints.openai.serving_chat as serving
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    from sglang.srt.environ import ToolStrictLevel
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    Serving = serving.OpenAIServingChat
    SRT = Path(serving.__file__).resolve().parents[2]
else:
    ROOT = Path(__file__).resolve().parents[3]
    SRT = ROOT / "python/sglang/srt"
    spec = importlib.util.spec_from_file_location(
        "candidate_guard", SRT / "constrained/xgrammar_schema.py"
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    namespace = dict(
        SchemaError=SchemaError,
        Draft202012Validator=Draft202012Validator,
        ToolStrictLevel=ToolStrictLevel,
        ToolChoice=ToolChoice,
        AllowedToolsChoice=AllowedToolsChoice,
        ChatCompletionMessageGenericParam=ChatCompletionMessageGenericParam,
        envs=NS(SGLANG_TOOL_STRICT_LEVEL=NS(get=lambda: 0)),
        get_model=lambda: NS(context_length=4096),
        get_serving=lambda: NS(allow_auto_truncate=False),
        has_xgrammar_unsupported_json_features=guard.has_xgrammar_unsupported_json_features,
    )
    utils = ast.parse((SRT / "function_call/utils.py").read_text(encoding="utf-8"))
    normalize_names = {
        "normalize_json_schema_types",
        "_normalize_single_type",
        "_normalize_type_list",
        "_matches_type_prefix",
    }
    constants = {
        "_STANDARD_JSON_SCHEMA_TYPES",
        "_JSON_SCHEMA_TYPE_ALIASES",
        "_PREFIX_BOUNDARY_CHARS",
        "_PREFIX_RULES",
    }
    nodes = [
        n
        for n in utils.body
        if getattr(n, "name", None) in normalize_names
        or (
            isinstance(n, ast.Assign)
            and any(isinstance(x, ast.Name) and x.id in constants for x in n.targets)
        )
        or (
            isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id in constants
        )
    ]
    nodes.insert(
        0,
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            "utils.py",
            "exec",
        ),
        namespace,
    )
    tree = ast.parse(
        (SRT / "entrypoints/openai/serving_chat.py").read_text(encoding="utf-8")
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "OpenAIServingChat"
    )
    names = {
        "_validate_request",
        "_all_tools",
        "_allowed_tool_names",
        "_effective_tools",
        "_effective_tool_choice",
    }
    cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
    cls.bases = []
    assert {n.name for n in cls.body} == names
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(
        compile(
            ast.fix_missing_locations(module), "candidate_serving_methods.py", "exec"
        ),
        namespace,
    )
    Serving = namespace["OpenAIServingChat"]
    serving = NS(**namespace)


def tool(name="loose", strict=False, unique=True):
    return {
        "type": "function",
        "function": {
            "name": name,
            "strict": strict,
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "uniqueItems": unique,
                    }
                },
            },
        },
    }


def request(tools, choice="auto", stream=False, response_format=None):
    body = dict(
        model="fixture",
        messages=[dict(role="user", content="test")],
        tools=copy.deepcopy(tools),
        tool_choice=choice,
        stream=stream,
    )
    if response_format:
        body["response_format"] = response_format
    if INSTALLED:
        return ChatCompletionRequest.model_validate(body)
    result = NS(
        **body,
        return_sampling_mask=False,
        return_meta_info=False,
        reasoning_effort=None,
        max_completion_tokens=None,
        max_tokens=None,
    )
    result.tools = [NS(function=NS(**t["function"])) for t in body["tools"]]
    result.messages = [NS(**m) for m in body["messages"]]
    if isinstance(choice, dict):
        result.tool_choice = ToolChoice(function=NS(**choice["function"]))
    result.response_format = (
        NS(
            type="json_schema",
            json_schema=NS(schema_=response_format["json_schema"]["schema"]),
        )
        if response_format
        else None
    )
    return result


class SchemaPrecheckTests(unittest.TestCase):
    def setUp(self):
        self.server = Serving.__new__(Serving)
        self.server.chat_encoding_spec = None
        self.server.tool_call_parser = "kimi_k3"
        self.server._grammar_backend = "xgrammar"
        self.server.template_manager = NS(reasoning_config=None)
        self.server._apply_dsv41_reasoning_off = lambda request: None
        self.server._validate_media_content = lambda request: None

    def check(self, req, level=0):
        with patch.object(
            serving.envs.SGLANG_TOOL_STRICT_LEVEL,
            "get",
            return_value=ToolStrictLevel(level),
        ):
            if INSTALLED:
                with patch.object(
                    serving, "get_model", return_value=NS(context_length=4096)
                ):
                    return self.server._validate_request(req)
            return self.server._validate_request(req)

    def test_choice_strict_level_stream_matrix(self):
        choices = [
            "none",
            "auto",
            "required",
            {"type": "function", "function": {"name": "loose"}},
        ]
        for choice in choices:
            for strict in (False, True):
                for level in (0, 1, 2):
                    for stream in (False, True):
                        with self.subTest(
                            choice=choice, strict=strict, level=level, stream=stream
                        ):
                            error = self.check(
                                request([tool(strict=strict)], choice, stream), level
                            )
                            self.assertEqual(bool(error), strict or level >= 2)
                            if error:
                                self.assertIn("unsupported by xgrammar", error)

    def test_mixed_strict_and_named_unselected_validation(self):
        for choice in (
            "none",
            "auto",
            "required",
            {"type": "function", "function": {"name": "loose"}},
        ):
            self.assertTrue(
                self.check(request([tool(), tool("strict", True)], choice)).startswith(
                    "Tool 1 "
                )
            )
        self.assertIsNone(self.check(request([tool(), tool("other")], "required")))

    def test_schema_validity_and_other_parser_preserved(self):
        bad = tool()
        bad["function"]["parameters"]["properties"]["ids"]["uniqueItems"] = 0
        self.assertIn("invalid 'parameters' schema", self.check(request([bad])))
        self.server.tool_call_parser = "qwen3_coder"
        self.assertIn("unsupported by xgrammar", self.check(request([tool()])))

    def test_response_format_and_exact_false(self):
        for value in (False, True):
            schema = {
                "type": "array",
                "items": {"type": "integer"},
                "uniqueItems": value,
            }
            fmt = {
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": schema},
            }
            self.assertEqual(
                bool(self.check(request([tool()], response_format=fmt))), value
            )
        for value in (True, 0, None, "false"):
            self.assertTrue(
                guard.has_xgrammar_unsupported_json_features({"uniqueItems": value})
            )
        self.assertFalse(
            guard.has_xgrammar_unsupported_json_features({"uniqueItems": False})
        )
        self.assertFalse(
            guard.has_xgrammar_unsupported_json_features(
                {"properties": {"uniqueItems": {"type": "boolean"}}}
            )
        )
        for keyword in (
            "contains",
            "dependentSchemas",
            "maxContains",
            "minContains",
            "multipleOf",
            "patternProperties",
            "propertyNames",
        ):
            self.assertTrue(
                guard.has_xgrammar_unsupported_json_features(
                    {"properties": {"x": {keyword: {}}}}
                )
            )
        self.assertTrue(
            guard.has_xgrammar_unsupported_json_features(
                {"type": "string", "pattern": "^a+$", "maxLength": 2}
            )
        )


class InstalledNativeTests(unittest.TestCase):
    @unittest.skipUnless(INSTALLED, "requires explicit installed-native mode")
    def test_native_enforcement_boundary(self):
        import xgrammar as x

        info = x.TokenizerInfo(
            [bytes([i]) for i in range(256)] + [b"<eos>"], stop_token_ids=[256]
        )
        compiler = x.GrammarCompiler(info, max_threads=1)
        cases = [
            (
                {"type": "array", "items": {"type": "integer"}, "uniqueItems": False},
                "[1,1]",
                True,
                True,
            ),
            (
                {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
                "[1,1]",
                True,
                False,
            ),
            (
                {"type": "string", "pattern": "^a+$", "maxLength": 2},
                '"aaa"',
                True,
                False,
            ),
            ({"type": "integer", "multipleOf": 3}, "6", True, True),
            ({"type": "integer", "multipleOf": 3}, "7", False, False),
            ({"type": "number", "multipleOf": 3}, "7", True, False),
        ]
        for schema, text, accepts, valid in cases:
            with self.subTest(schema=schema, text=text):
                matcher = x.GrammarMatcher(compiler.compile_json_schema(schema))
                self.assertEqual(
                    matcher.accept_string(text) and matcher.accept_token(256), accepts
                )
                self.assertEqual(
                    Draft202012Validator(schema).is_valid(json.loads(text)), valid
                )

    @unittest.skipUnless(INSTALLED, "requires explicit installed-native mode")
    def test_actual_parser_retains_structural_tags(self):
        for level in (0, 1, 2):
            for strict in (False, True):
                for choice in (
                    "auto",
                    "required",
                    {"type": "function", "function": {"name": "loose"}},
                ):
                    req = request([tool(strict=strict, unique=False)], choice)
                    with patch.object(
                        serving.envs.SGLANG_TOOL_STRICT_LEVEL,
                        "get",
                        return_value=ToolStrictLevel(level),
                    ):
                        parser = FunctionCallParser(req.tools, "kimi_k3")
                        constraint = parser.get_structure_constraint(req.tool_choice)
                    self.assertIsNotNone(constraint)
                    self.assertEqual(constraint[0], "structural_tag")
                    encoded = constraint[1].model_dump_json()
                    self.assertEqual("uniqueItems" in encoded, strict or level >= 2)
        req = request([tool()], "none")
        parser = FunctionCallParser(req.tools, "kimi_k3")
        self.assertIsNone(parser.get_structure_constraint(req.tool_choice))


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "mode": "installed-modules-native"
                if INSTALLED
                else "source-method-cpu-fixtures",
                "source_sha256": {
                    str(p.relative_to(SRT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [
                        SRT / "entrypoints/openai/serving_chat.py",
                        SRT / "constrained/xgrammar_schema.py",
                    ]
                },
            }
        )
    )
    unittest.main()
