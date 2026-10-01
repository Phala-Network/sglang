"""Selected-grammar regressions; PHALA_KIMI_INSTALLED=1 uses real protocol/parsers.

Default runs exact source methods with CPU fixtures, no serving/CUDA stand-ins
claimed as installed acceptance. Native mode must import actual installed modules.
"""

import ast
import copy
import inspect
import json
import logging
import unittest
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import test_kimi_schema_precheck_cpu as pre
from jsonschema import Draft202012Validator

BAD = {"type": "array", "items": {"type": "integer"}, "uniqueItems": True}
PATTERN = {"type": "string", "pattern": "^a+$", "maxLength": 2}

if not pre.INSTALLED:

    class CPURequest(NS):
        _DEFAULT_SAMPLING_PARAMS = dict.fromkeys(
            ["temperature", "top_p", "top_k", "min_p", "repetition_penalty"], 1
        )

        def __getattr__(self, name):
            return None

        def effective_tool_choice(self):
            return self.tool_choice

    tree = ast.parse(
        (pre.SRT / "entrypoints/openai/protocol.py").read_text(encoding="utf-8")
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ChatCompletionRequest"
    )
    methods = [
        n
        for n in cls.body
        if getattr(n, "name", None)
        in {"uses_json_schema_constraint", "to_sampling_params"}
    ]
    scope = {
        "ToolChoice": pre.ToolChoice,
        "convert_json_schema_to_str": json.dumps,
        "logger": logging.getLogger(__name__),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    exec(
        compile(
            ast.fix_missing_locations(module), "actual_sampling_methods.py", "exec"
        ),
        scope,
    )
    for method in methods:
        setattr(CPURequest, method.name, scope[method.name])


def request(schema=None, strict=None, *, tools=None, choice="auto", input_ids=None):
    body = dict(
        model="fixture",
        messages=[dict(role="user", content="test")],
        tools=tools or [],
        tool_choice=choice,
    )
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": copy.deepcopy(schema)},
        }
        if strict != "omitted":
            body["response_format"]["json_schema"]["strict"] = strict
    if input_ids is not None:
        body["input_ids"] = input_ids
    if pre.INSTALLED:
        return pre.ChatCompletionRequest.model_validate(body)
    fmt = body.pop("response_format", None)
    return CPURequest(
        **body,
        response_format=NS(
            type="json_schema",
            json_schema=NS(
                schema_=schema, strict=None if strict == "omitted" else strict
            ),
        )
        if fmt
        else None,
    )


def sampling(req, constraint=None, renderer=False):
    return req.to_sampling_params(
        stop=[],
        model_generation_config={},
        tool_call_constraint=constraint,
        renderer_handles_response_format=renderer,
    )


def check(params):
    return pre.guard.validate_xgrammar_sampling_constraints(params)


class SelectedGrammarTests(unittest.TestCase):
    def test_response_strict_renderer_matrix(self):
        for schema in (BAD, PATTERN):
            for strict in ("omitted", None, False, True):
                for renderer in (False, True):
                    for input_ids in (None, [1, 2]):
                        with self.subTest(
                            strict=strict, renderer=renderer, input_ids=input_ids
                        ):
                            req = request(schema, strict, input_ids=input_ids)
                            params = sampling(req, renderer=renderer)
                            constrained = (
                                strict is not False
                                or not renderer
                                or input_ids is not None
                            )
                            self.assertEqual("json_schema" in params, constrained)
                            if constrained:
                                with self.assertRaisesRegex(
                                    ValueError, "unsupported by xgrammar"
                                ):
                                    check(params)
                            else:
                                check(params)

    def test_actual_tool_constraint_selected_after_precedence(self):
        for name, value in (
            ("json_schema", BAD),
            (
                "structural_tag",
                NS(
                    model_dump=lambda **kwargs: {
                        "structures": [{"schema": BAD}],
                        "triggers": ["call"],
                    }
                ),
            ),
        ):
            constraint = (name, value)
            with self.assertRaises(ValueError):
                check(sampling(request(), constraint))
            # An auto tool grammar is not selected when response_format owns output.
            check(sampling(request({"type": "string"}, True), constraint))
        check(sampling(request(), ("full_assistant_ebnf", 'root ::= "OK"')))
        check(sampling(request(), None))

    def test_structural_format_schema_positions_only(self):
        for fmt in ("json_schema", "qwen_xml_parameter"):
            selected = {
                "structural_tag": json.dumps(
                    {
                        "format": {
                            "type": "tag",
                            "content": {"type": fmt, "json_schema": BAD},
                        }
                    }
                )
            }
            with self.assertRaises(ValueError):
                check(selected)
        check(
            {
                "structural_tag": json.dumps(
                    {
                        "format": {
                            "type": "json_schema",
                            "json_schema": {"const": {"uniqueItems": True}},
                        }
                    }
                )
            }
        )

    def test_new_error_is_private_valueerror_before_engine(self):
        secret = {
            "type": "array",
            "uniqueItems": True,
            "description": "private-value-should-not-leak",
        }
        with self.assertRaises(ValueError) as raised:
            check({"json_schema": json.dumps(secret)})
        self.assertNotIn("private-value", str(raised.exception))
        source = (pre.SRT / "entrypoints/openai/serving_chat.py").read_text(
            encoding="utf-8"
        )
        conversion = source[
            source.index("    def _convert_to_internal_request(") : source.index(
                "    def _process_messages("
            )
        ]
        self.assertLess(
            conversion.index("request.to_sampling_params("),
            conversion.index("validate_xgrammar_sampling_constraints(sampling_params)"),
        )
        self.assertLess(
            conversion.index("validate_xgrammar_sampling_constraints(sampling_params)"),
            conversion.index("adapted_request ="),
        )

    def test_kimi_template_request_schema_precedence(self):
        if pre.INSTALLED:
            req = request(BAD, False)
            req.chat_template_kwargs = {
                "response_format": {"type": "text"},
                "response_schema": {"type": "string"},
            }
            original = copy.deepcopy(req.chat_template_kwargs)
            tokenizer = NS(apply_chat_template=Mock(return_value=[1]))
            server = pre.Serving.__new__(pre.Serving)
            server.chat_encoding_spec = "kimi_k3"
            server.tokenizer_manager = NS(tokenizer=tokenizer)
            messages = [{"role": "user", "content": "test"}]
            self.assertEqual(
                server._encode_messages(messages, req, pre.serving.ThinkingMode.CHAT),
                [1],
            )
            kwargs = tokenizer.apply_chat_template.call_args.kwargs
            self.assertEqual(
                kwargs["response_format"],
                req.response_format.model_dump(exclude_unset=True, by_alias=True),
            )
            self.assertEqual(kwargs["response_schema"], BAD)
            self.assertEqual(req.chat_template_kwargs, original)
            # The native schema override is local to the Kimi branch.
            server.chat_encoding_spec = "dsv4"
            tokenizer.apply_chat_template.reset_mock()
            self.assertIsNone(
                server._encode_messages(messages, req, pre.serving.ThinkingMode.CHAT)
            )
            tokenizer.apply_chat_template.assert_not_called()
            self.assertEqual(req.chat_template_kwargs, original)
            return
        source = ast.parse(
            (pre.SRT / "entrypoints/openai/serving_chat.py").read_text(encoding="utf-8")
        )
        # Execute the real kwargs assignment block, not a copied policy.
        block = next(
            n
            for n in ast.walk(source)
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "request.response_format is not None"
            and any(isinstance(x, ast.Assign) for x in n.body)
        )
        fmt = {
            "type": "json_schema",
            "json_schema": {"name": "test", "strict": False, "schema": BAD},
        }
        req = NS(
            response_format=NS(
                type="json_schema",
                json_schema=NS(schema_=BAD),
                model_dump=lambda **kwargs: copy.deepcopy(fmt),
            )
        )
        kwargs = {
            "response_format": {"type": "text"},
            "response_schema": {"type": "string"},
        }
        ns = {"request": req, "template_kwargs": kwargs}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])),
                "actual_template_block.py",
                "exec",
            ),
            ns,
        )
        self.assertEqual(kwargs["response_format"], fmt)
        self.assertEqual(kwargs["response_schema"], BAD)


class ErrorEnvelopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_handle_request_preserves_prevalidation_400(self):
        tree = ast.parse(
            (pre.SRT / "entrypoints/openai/serving_base.py").read_text(encoding="utf-8")
        )
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "OpenAIServingBase"
        )
        methods = [
            n
            for n in cls.body
            if getattr(n, "name", None) in {"handle_request", "create_error_response"}
        ]

        class ErrorPayload(NS):
            def model_dump(self):
                return vars(self)

        scope = dict(
            monotonic_time=lambda: 1,
            ErrorResponse=ErrorPayload,
            ORJSONResponse=lambda **kwargs: NS(**kwargs),
            HTTPException=type("HTTPException", (Exception,), {}),
            UnsupportedXGrammarSchema=pre.guard.UnsupportedXGrammarSchema,
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *methods,
            ],
            type_ignores=[],
        )
        exec(
            compile(
                ast.fix_missing_locations(module), "actual_error_mapping.py", "exec"
            ),
            scope,
        )

        async def convert(*args):
            check({"json_schema": json.dumps(BAD)})

        server = NS(
            _validate_request=lambda req: None,
            tokenizer_manager=NS(request_logger=NS(log_requests=False)),
            _convert_to_internal_request_async=convert,
        )
        if pre.INSTALLED:
            base = pre.Serving.__mro__[1]
            handle = base.handle_request
            server.create_error_response = MethodType(
                base.create_error_response, server
            )
        else:
            handle = scope["handle_request"]
            server.create_error_response = MethodType(
                scope["create_error_response"], server
            )
        for stream in (False, True):
            response = await handle(server, NS(stream=stream), None)
            content = json.loads(response.body) if pre.INSTALLED else response.content
            self.assertEqual(response.status_code, 400)
            self.assertEqual(content["type"], "BadRequestError")
            self.assertEqual(content["object"], "error")
            self.assertEqual(content["code"], 400)


class GlmSelectionTests(unittest.TestCase):
    def test_parameter_level_does_not_select_shallow_ebnf(self):
        if pre.INSTALLED:
            for level in (0, 1, 2):
                for strict in (False, True):
                    req = request(tools=[pre.tool(strict=strict)])
                    with patch.object(
                        pre.serving.envs.SGLANG_TOOL_STRICT_LEVEL,
                        "get",
                        return_value=pre.ToolStrictLevel(level),
                    ):
                        parser = pre.FunctionCallParser(req.tools, "glm47")
                    self.assertEqual(
                        parser.detector.use_full_assistant_constraint,
                        not strict and level < 2,
                    )
            return
        tree = ast.parse(
            (pre.SRT / "function_call/function_call_parser.py").read_text(
                encoding="utf-8"
            )
        )
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "FunctionCallParser"
        )
        init = next(n for n in cls.body if getattr(n, "name", None) == "__init__")

        class Detector:
            pass

        scope = dict(
            inspect=inspect,
            Glm47MoeDetector=Detector,
            ToolStrictLevel=pre.ToolStrictLevel,
        )
        for level in (0, 1, 2):
            for strict in (False, True):
                scope["envs"] = NS(
                    SGLANG_TOOL_STRICT_LEVEL=NS(get=lambda: pre.ToolStrictLevel(level))
                )
                module = ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        init,
                    ],
                    type_ignores=[],
                )
                exec(
                    compile(
                        ast.fix_missing_locations(module),
                        "actual_glm_parser_init.py",
                        "exec",
                    ),
                    scope,
                )
                parser = NS(ToolCallParserEnum={"glm47": Detector})
                scope["__init__"](parser, [NS(function=NS(strict=strict))], "glm47")
                self.assertEqual(
                    parser.detector.use_full_assistant_constraint,
                    not strict and level < 2,
                )


class SchemaCapabilityTests(unittest.TestCase):
    def test_noops_and_supported_integer(self):
        schemas = [
            {"type": "string", "multipleOf": 2},
            {"type": "string", "uniqueItems": True},
            {"type": "integer", "multipleOf": 1, "minimum": 0},
            {"type": "integer", "multipleOf": 2},
            {"type": "integer", "multipleOf": 3},
            {"type": "integer", "multipleOf": 1024},
            {"type": "object", "dependentSchemas": {}},
            {"type": "object", "patternProperties": {}},
            {"type": "object", "propertyNames": {}},
            {"type": "array", "contains": {}, "minContains": 0},
            {"type": "array", "minContains": 4},
            {"type": "string", "pattern": "^a+$", "minLength": 0},
            {"type": "string", "$defs": {"unused": BAD}},
            {"type": "string", "properties": {"inapplicable": BAD}},
        ]
        for schema in schemas:
            with self.subTest(schema=schema):
                Draft202012Validator.check_schema(schema)
                check({"json_schema": json.dumps(schema)})

    def test_active_gaps_remain_rejected(self):
        schemas = [
            BAD,
            PATTERN,
            {"allOf": [{"type": "integer", "multipleOf": 2}, {"minimum": 0}]},
            {
                "$defs": {"n": {"type": "integer", "multipleOf": 2}},
                "$ref": "#/$defs/n",
                "minimum": 0,
            },
            {"anyOf": [{"type": "integer", "multipleOf": 2}, {"type": "null"}]},
            {"type": "number", "multipleOf": 2},
            {"type": "integer", "multipleOf": 1025},
            {"type": "integer", "multipleOf": 1.5},
            {"type": "integer", "multipleOf": 2, "minimum": 0},
            {"type": "array", "contains": {}},
            {"type": "array", "contains": {}, "minContains": 0, "maxContains": 2},
            {"$defs": {"used": BAD}, "$ref": "#/$defs/used"},
            {
                "definitions": {"used": PATTERN},
                "properties": {"x": {"$ref": "#/definitions/used"}},
            },
            {"type": ["string", "array"], "uniqueItems": True},
            {"type": "object", "patternProperties": {"^x": {"type": "integer"}}},
        ]
        for schema in schemas:
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                check({"json_schema": json.dumps(schema)})

    def test_noop_normalization_preserves_request_and_data(self):
        original = {
            "type": "object",
            "propertyNames": True,
            "properties": {
                "array": {"type": "array", "minContains": 1, "maxContains": 2}
            },
            "const": {"propertyNames": True, "minContains": 3},
        }
        before = copy.deepcopy(original)
        normalized = pre.guard.normalize_xgrammar_schema_noops(original)
        self.assertEqual(original, before)
        self.assertNotIn("propertyNames", normalized)
        self.assertEqual(normalized["properties"]["array"], {"type": "array"})
        self.assertEqual(normalized["const"], original["const"])
        active = {"type": "array", "contains": {}, "minContains": 1, "maxContains": 2}
        self.assertEqual(pre.guard.normalize_xgrammar_schema_noops(active), active)

    def test_recursive_local_refs_and_data_keywords(self):
        recursive = {
            "type": "object",
            "properties": {"next": {"anyOf": [{"type": "null"}, {"$ref": "#"}]}},
        }
        check({"json_schema": json.dumps(recursive)})
        recursive["properties"]["bad"] = BAD
        with self.assertRaises(ValueError):
            check({"json_schema": json.dumps(recursive)})
        check(
            {
                "json_schema": json.dumps(
                    {"enum": [{"uniqueItems": True, "contains": 3}]}
                )
            }
        )


class InstalledParserTests(unittest.TestCase):
    @unittest.skipUnless(pre.INSTALLED, "requires actual installed protocol/serving")
    def test_malformed_response_and_strict_wire_types(self):
        from pydantic import ValidationError

        server = pre.Serving.__new__(pre.Serving)
        server.chat_encoding_spec = "kimi_k3"
        server.tool_call_parser = "kimi_k3"
        server._grammar_backend = "xgrammar"
        server.template_manager = NS(reasoning_config=None)
        server._apply_dsv41_reasoning_off = lambda request: None
        server._validate_media_content = lambda request: None
        for strict in ("omitted", None, False, True):
            req = request({"type": "string", "maxLength": -1}, strict)
            with patch.object(
                pre.serving, "get_model", return_value=NS(context_length=4096)
            ):
                self.assertIn(
                    "Invalid response_format JSON schema", server._validate_request(req)
                )
        for strict in ("false", "true", 0, 1):
            with self.assertRaises(ValidationError):
                request({"type": "string"}, strict)

    @unittest.skipUnless(pre.INSTALLED, "requires actual installed parsers")
    def test_real_auto_tool_is_superseded_by_response_format(self):
        req = request({"type": "string"}, True, tools=[pre.tool(strict=True)])
        parser = pre.FunctionCallParser(req.tools, "kimi_k3")
        constraint = parser.get_structure_constraint(req.tool_choice)
        self.assertIsNotNone(constraint)
        params = sampling(req, constraint, renderer=True)
        self.assertNotIn("structural_tag", params)
        self.assertEqual(json.loads(params["json_schema"]), {"type": "string"})
        check(params)
        req.tool_choice = "required"
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            sampling(
                req, parser.get_structure_constraint(req.tool_choice), renderer=True
            )

    @unittest.skipUnless(pre.INSTALLED, "requires actual native backend")
    def test_real_backend_noop_normalization_and_strict_protection(self):
        import xgrammar as xgr

        from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
        from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend

        info = xgr.TokenizerInfo(
            [bytes([i]) for i in range(256)] + [b"<eos>"], stop_token_ids=[256]
        )
        backend = XGrammarGrammarBackend.__new__(XGrammarGrammarBackend)
        backend.grammar_compiler = xgr.GrammarCompiler(info, max_threads=1)
        backend.override_stop_tokens = [256]
        backend.vocab_size = 257
        backend.any_whitespace = True
        backend.max_whitespace_cnt = None
        for schema in (
            BAD,
            PATTERN,
            {"allOf": [{"type": "integer", "multipleOf": 2}, {"minimum": 0}]},
            {
                "$defs": {"n": {"type": "integer", "multipleOf": 2}},
                "$ref": "#/$defs/n",
                "minimum": 0,
            },
        ):
            self.assertIsInstance(
                backend.dispatch_json(json.dumps(schema)), InvalidGrammarObject
            )
        fixtures = [
            ({"type": "integer", "multipleOf": 2}, "2", "3"),
            (
                {"type": "object", "propertyNames": True, "additionalProperties": True},
                '{"x":1}',
                "[]",
            ),
            (
                {"type": "array", "items": {}, "minContains": 1, "maxContains": 1},
                "[1,2]",
                "{}",
            ),
        ]
        for schema, valid, invalid in fixtures:
            before = copy.deepcopy(schema)
            for text, expected in ((valid, True), (invalid, False)):
                grammar = backend.dispatch_json(json.dumps(schema))
                self.assertNotIsInstance(grammar, InvalidGrammarObject)
                self.assertEqual(
                    grammar.matcher.accept_string(text)
                    and grammar.matcher.accept_token(256),
                    expected,
                )
            self.assertEqual(schema, before)
        for legacy in (False, True):

            def tag(schema):
                if legacy:
                    return {
                        "structures": [
                            {"begin": "<call>", "schema": schema, "end": "</call>"}
                        ],
                        "triggers": ["<call>"],
                    }
                return {
                    "type": "structural_tag",
                    "format": {
                        "type": "tag",
                        "begin": "<call>",
                        "content": {"type": "json_schema", "json_schema": schema},
                        "end": "</call>",
                    },
                }

            self.assertIsInstance(
                backend.dispatch_structural_tag(json.dumps(tag(BAD))),
                InvalidGrammarObject,
            )
            schema = {
                "type": "object",
                "propertyNames": True,
                "additionalProperties": True,
            }
            grammar = backend.dispatch_structural_tag(json.dumps(tag(schema)))
            self.assertNotIsInstance(grammar, InvalidGrammarObject)
            self.assertTrue(
                grammar.matcher.accept_string('<call>{"x":1}</call>')
                and grammar.matcher.accept_token(256)
            )

    @unittest.skipUnless(pre.INSTALLED, "requires actual installed native tags")
    def test_kimi_named_excludes_unselected_strict_schema(self):
        loose = pre.tool(name="selected", unique=False)
        strict = pre.tool(name="unselected", strict=True)
        req = request(
            tools=[loose, strict],
            choice={"type": "function", "function": {"name": "selected"}},
        )
        with patch.object(
            pre.serving.envs.SGLANG_TOOL_STRICT_LEVEL,
            "get",
            return_value=pre.ToolStrictLevel(0),
        ):
            parser = pre.FunctionCallParser(req.tools, "kimi_k3")
            constraint = parser.get_structure_constraint(req.tool_choice)
        self.assertIsNotNone(constraint)
        check(sampling(req, constraint))

    @unittest.skipUnless(
        pre.INSTALLED, "requires actual installed protocol/parsers/native tags"
    )
    def test_cross_model_selected_tool_schemas(self):
        for parser_name in ("kimi_k3", "glm47", "qwen3_coder", "llama3"):
            for level in (0, 1, 2):
                for strict in (False, True):
                    for choice in (
                        "none",
                        "auto",
                        "required",
                        {"type": "function", "function": {"name": "loose"}},
                    ):
                        for schema in (BAD, PATTERN):
                            with self.subTest(
                                parser=parser_name,
                                level=level,
                                strict=strict,
                                choice=choice,
                                schema=schema,
                            ):
                                t = pre.tool(strict=strict)
                                t["function"]["parameters"] = {
                                    "type": "object",
                                    "properties": {"value": schema},
                                    "required": ["value"],
                                }
                                req = request(tools=[t], choice=choice)
                                with patch.object(
                                    pre.serving.envs.SGLANG_TOOL_STRICT_LEVEL,
                                    "get",
                                    return_value=pre.ToolStrictLevel(level),
                                ):
                                    parser = pre.FunctionCallParser(
                                        req.tools, parser_name
                                    )
                                    constraint = parser.get_structure_constraint(
                                        req.tool_choice
                                    )
                                params = sampling(req, constraint)
                                if choice == "none":
                                    check(params)
                                elif strict or level >= 2:
                                    # Required/named fallback or strict native tags must
                                    # carry active parameters to the shared validator.
                                    self.assertIsNotNone(constraint)
                                    with self.assertRaises(ValueError):
                                        check(params)
                                elif parser_name in ("kimi_k3", "glm47"):
                                    check(params)
                                # Other non-strict parsers may intentionally enforce
                                # fallback JSON parameters: validate actual selection.


if __name__ == "__main__":
    unittest.main()
