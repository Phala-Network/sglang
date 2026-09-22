"""Execute actual adapter methods; optional installed-native compiler/matcher.

PHALA_XGRAMMAR_NATIVE=1 requires the installed union candidate and never skips
its tests. No GPU runtime imports, model downloads or kernel stand-ins are used
for native grammar compilation, token masks or rollback.
"""

import ast
import importlib.util
import json
import logging
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"
NATIVE = os.environ.get("PHALA_XGRAMMAR_NATIVE") == "1"
spec = importlib.util.spec_from_file_location(
    "adapter_schema", SRT / "constrained/xgrammar_schema.py"
)
schema_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schema_module)


class InvalidGrammar:
    def __init__(self, message):
        self.message = message


def source_nodes(path, names, namespace, *, remove_imports=False):
    tree = ast.parse((SRT / path).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    if remove_imports:

        class RemoveImports(ast.NodeTransformer):
            def visit_Import(self, node):
                return None

            def visit_ImportFrom(self, node):
                return None

        tree = RemoveImports().visit(tree)
    tree.body.insert(
        0,
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
    )
    exec(compile(ast.fix_missing_locations(tree), str(SRT / path), "exec"), namespace)


def adapter_namespace(compiler_factory=None):
    namespace = {
        "BaseGrammarBackend": object,
        "json": json,
        "logger": logging.getLogger(__name__),
        "GrammarStats": lambda **kwargs: SimpleNamespace(**kwargs),
        "InvalidGrammarObject": InvalidGrammar,
        "GrammarCompiler": compiler_factory or Mock(),
        "validate_xgrammar_schema": schema_module.validate_xgrammar_schema,
        "validate_xgrammar_whitespace_limit": schema_module.validate_xgrammar_whitespace_limit,
    }
    source_nodes("constrained/utils.py", {"is_legacy_structural_tag"}, namespace)
    source_nodes(
        "constrained/xgrammar_backend.py",
        {"TokenizerNotSupportedError", "XGrammarGrammarBackend"},
        namespace,
    )
    namespace["XGrammarGrammarBackend"]._from_context = lambda self, ctx, key, stats: (
        ctx
    )
    return namespace


class SchemaTests(unittest.TestCase):
    def test_pattern_and_length_rejected_only_in_schema_positions(self):
        bad = {"type": "string", "pattern": "^a+$", "minLength": 3}
        for schema in (
            bad,
            {"items": bad},
            {"anyOf": [bad]},
            {"$defs": {"x": bad}},
            {"properties": {"pattern": bad}},
            {"additionalProperties": bad},
        ):
            with self.subTest(schema=schema):
                self.assertTrue(
                    schema_module.has_xgrammar_unsupported_json_features(schema)
                )
        for key in ("const", "enum", "examples", "default"):
            data = [bad] if key in {"enum", "examples"} else bad
            self.assertFalse(
                schema_module.has_xgrammar_unsupported_json_features({key: data})
            )
        self.assertFalse(
            schema_module.has_xgrammar_unsupported_json_features(
                {
                    "properties": {
                        "pattern": {"type": "string"},
                        "minLength": {"type": "integer"},
                    }
                }
            )
        )

    def test_limit_validation(self):
        for limit in (0, -1, True, 1.5, "2", 2**31):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                schema_module.validate_xgrammar_whitespace_limit(limit)
        for backend in ("none", "llguidance", "outlines", "custom"):
            with self.assertRaises(ValueError):
                schema_module.validate_xgrammar_whitespace_limit(2, backend=backend)
            schema_module.validate_xgrammar_whitespace_limit(None, backend=backend)
        with self.assertRaises(ValueError):
            schema_module.validate_xgrammar_whitespace_limit(2, any_whitespace=False)
        schema_module.validate_xgrammar_whitespace_limit(1)

    def test_nested_structural_schema_validation(self):
        target = adapter_namespace()["XGrammarGrammarBackend"]
        bad = {"type": "array", "uniqueItems": True}
        wrappers = [
            lambda x: {"type": "tag", "content": x},
            lambda x: {"type": "sequence", "elements": [x]},
            lambda x: {"type": "or", "elements": [x]},
            lambda x: {
                "type": "triggered_tags",
                "tags": [{"type": "tag", "content": x}],
            },
            lambda x: {
                "type": "tags_with_separator",
                "tags": [{"type": "tag", "content": x}],
            },
            lambda x: {"type": "dispatch", "rules": [["<call>", x]]},
            lambda x: {"type": "token_dispatch", "rules": [[12, x]]},
        ] + [
            (lambda x, name=name: {"type": name, "content": x})
            for name in ("optional", "plus", "star", "repeat")
        ]
        for fmt_type in ("json_schema", "qwen_xml_parameter"):
            for wrap in wrappers:
                with self.assertRaises(RuntimeError):
                    target._sanitize_structural_format(
                        wrap({"type": fmt_type, "json_schema": bad})
                    )
        with self.assertRaises(RuntimeError):
            target._sanitize_structural_tag_structures(
                {"structures": [{"schema": bad}]}
            )
        literal = {
            "type": "json_schema",
            "json_schema": {"const": {"uniqueItems": True}},
        }
        target._sanitize_structural_format(literal)
        self.assertEqual(literal["json_schema"], {"const": {"uniqueItems": True}})

    def test_factory_config_and_unsupported_tokenizer_fail_closed(self):
        namespace = adapter_namespace()
        cfg = SimpleNamespace(
            constrained_json_max_whitespace_cnt=2,
            constrained_json_disable_any_whitespace=False,
            enable_strict_thinking=False,
            reasoning_parser=None,
        )
        execution = SimpleNamespace(kernel=SimpleNamespace(grammar_backend="xgrammar"))
        namespace.update(
            get_exec=lambda: execution,
            get_serving=lambda: cfg,
            GRAMMAR_BACKEND_REGISTRY={},
            get_context=lambda: SimpleNamespace(override=Mock()),
        )
        source_nodes(
            "constrained/base_grammar_backend.py",
            {"create_grammar_backend"},
            namespace,
            remove_imports=True,
        )
        factory = namespace["create_grammar_backend"]
        backend = factory(
            None, SimpleNamespace(init_xgrammar=lambda: ("info", None)), 257
        )
        self.assertEqual(backend.max_whitespace_cnt, 2)
        with self.assertRaises(ValueError):
            factory(None, SimpleNamespace(init_xgrammar=lambda: (None, None)), 257)
        for name in ("none", "llguidance", "outlines"):
            execution.kernel.grammar_backend = name
            with self.assertRaises(ValueError):
                factory(None, None, 257)
        execution.kernel.grammar_backend = "xgrammar"
        cfg.constrained_json_max_whitespace_cnt = None
        self.assertIsNone(
            factory(None, SimpleNamespace(init_xgrammar=lambda: (None, None)), 257)
        )

    def test_resolution_and_cli_field_wiring(self):
        namespace = {
            "resolving_view": lambda args: args,
            "declare_resolution": Mock(),
            "validate_xgrammar_whitespace_limit": schema_module.validate_xgrammar_whitespace_limit,
        }
        source_nodes(
            "arg_groups/serving_hook.py",
            {"handle_grammar_backend"},
            namespace,
            remove_imports=True,
        )
        cfg = SimpleNamespace(
            grammar_backend=None,
            constrained_json_max_whitespace_cnt=2,
            constrained_json_disable_any_whitespace=False,
        )
        namespace["handle_grammar_backend"](cfg)
        namespace["declare_resolution"].assert_called_once()
        cfg.grammar_backend = "outlines"
        with self.assertRaises(ValueError):
            namespace["handle_grammar_backend"](cfg)
        fields = ast.parse((SRT / "arg_groups/fields/serving.py").read_text())
        cls = next(x for x in fields.body if isinstance(x, ast.ClassDef))
        field = next(
            x
            for x in cls.body
            if isinstance(x, ast.AnnAssign)
            and x.target.id == "constrained_json_max_whitespace_cnt"
        )
        self.assertIsNone(ast.literal_eval(field.value))
        self.assertIn("Optional[int]", ast.unparse(field.annotation))
        self.assertIn(
            '"constrained_json_max_whitespace_cnt"',
            (SRT / "arg_groups/field_order.py").read_text(),
        )


@unittest.skipUnless(
    NATIVE, "set PHALA_XGRAMMAR_NATIVE=1 for installed candidate tests"
)
class InstalledNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.metadata

        import xgrammar as xg

        if importlib.metadata.version("xgrammar") != "0.2.6+phala.union1":
            raise RuntimeError(
                "native tests require the single pinned union1 candidate"
            )
        cls.xg = xg
        cls.info = xg.TokenizerInfo(
            [bytes([i]) for i in range(256)] + [b"<eos>"],
            xg.VocabType.RAW,
            stop_token_ids=[256],
        )
        namespace = adapter_namespace(
            lambda **kwargs: xg.GrammarCompiler(max_threads=2, **kwargs)
        )
        namespace.update(
            StructuralTag=xg.StructuralTag, StructuralTagItem=xg.StructuralTagItem
        )
        cls.backend_type = namespace["XGrammarGrammarBackend"]

    def backend(self, limit=None):
        return self.backend_type(
            SimpleNamespace(init_xgrammar=lambda: (self.info, None)),
            257,
            max_whitespace_cnt=limit,
        )

    def accepts(self, compiled, text):
        self.assertNotIsInstance(compiled, InvalidGrammar)
        matcher = self.xg.GrammarMatcher(compiled)
        return matcher.accept_string(text) and matcher.accept_token(256)

    def test_direct_schema_bound_and_string_whitespace(self):
        grammar = self.backend(2).dispatch_json(
            json.dumps(
                {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                }
            )
        )
        self.assertTrue(self.accepts(grammar, '{"x":  "a     b"}'))
        self.assertFalse(self.accepts(grammar, '{"x":   "a"}'))

    def test_bounded_any_retains_arbitrary_json(self):
        grammar = self.backend(2).dispatch_json("$$ANY$$")
        for text in ("[1,  2]", '{"unknown":  2}', '"a    b"', "true", "2"):
            self.assertTrue(self.accepts(grammar, text), text)
        self.assertFalse(self.accepts(grammar, "[1,   2]"))
        self.assertTrue(
            self.accepts(self.backend().dispatch_json("$$ANY$$"), "[1,     2]")
        )

    def test_direct_pattern_length_fail_closed(self):
        for keyword in ("minLength", "maxLength"):
            result = self.backend().dispatch_json(
                json.dumps(
                    {
                        "type": "string",
                        "pattern": "^a+$",
                        keyword: 2,
                    }
                )
            )
            self.assertIsInstance(result, InvalidGrammar)

    def test_legacy_rejects_unsupported_schema(self):
        result = self.backend().dispatch_structural_tag(
            json.dumps(
                {
                    "type": "structural_tag",
                    "triggers": ["<call>"],
                    "structures": [
                        {
                            "begin": "<call>",
                            "end": "</call>",
                            "schema": {"type": "array", "uniqueItems": True},
                        }
                    ],
                }
            )
        )
        self.assertIsInstance(result, InvalidGrammar)

    def test_legacy_order_and_required_native(self):
        result = self.backend().dispatch_structural_tag(
            json.dumps(
                {
                    "type": "structural_tag",
                    "triggers": ["<call>"],
                    "at_least_one": True,
                    "structures": [
                        {
                            "begin": "<call>",
                            "end": "</call>",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer"},
                                    "b": {"type": "integer"},
                                },
                                "required": ["a", "b"],
                                "additionalProperties": False,
                            },
                        }
                    ],
                }
            )
        )
        self.assertTrue(self.accepts(result, '<call>{"b":2,"a":1}</call>'))
        self.assertFalse(self.accepts(result, '<call>{"b":2}</call>'))

    def test_nested_json_and_xml_reject_ignored_constraints(self):
        for fmt_type in ("json_schema", "qwen_xml_parameter"):
            result = self.backend().dispatch_structural_tag(
                json.dumps(
                    {
                        "type": "structural_tag",
                        "format": {
                            "type": "optional",
                            "content": {
                                "type": "tag",
                                "begin": "<call>",
                                "end": "</call>",
                                "content": {
                                    "type": fmt_type,
                                    "json_schema": {
                                        "properties": {
                                            "x": {"pattern": "^a+$", "maxLength": 2}
                                        }
                                    },
                                },
                            },
                        },
                    }
                )
            )
            self.assertIsInstance(result, InvalidGrammar)

    def test_per_format_limit_preserved_not_global_override(self):
        result = self.backend(1).dispatch_structural_tag(
            json.dumps(
                {
                    "type": "structural_tag",
                    "format": {
                        "type": "json_schema",
                        "json_schema": {"type": "array", "items": {"type": "integer"}},
                        "max_whitespace_cnt": 3,
                    },
                }
            )
        )
        self.assertTrue(self.accepts(result, "[1,   2]"))
        self.assertFalse(self.accepts(result, "[1,    2]"))

    def test_empty_xml_and_undeclared_required(self):
        for schema, text, rejected in (
            (
                {"type": "object", "properties": {}, "additionalProperties": False},
                "<call></call>",
                "<call> </call>",
            ),
            (
                {
                    "type": "object",
                    "required": ["x"],
                    "additionalProperties": {"type": "integer"},
                },
                "<call><parameter=x>2</parameter></call>",
                "<call></call>",
            ),
        ):
            result = self.backend().dispatch_structural_tag(
                json.dumps(
                    {
                        "type": "structural_tag",
                        "format": {
                            "type": "tag",
                            "begin": "<call>",
                            "end": "</call>",
                            "content": {
                                "type": "qwen_xml_parameter",
                                "json_schema": schema,
                            },
                        },
                    }
                )
            )
            self.assertTrue(self.accepts(result, text))
            self.assertFalse(self.accepts(result, rejected))

    def test_mask_rollback_eos_from_adapter_compilation(self):
        import torch

        grammar = self.backend(2).dispatch_json(
            '{"type":"array","items":{"type":"integer"}}'
        )
        matcher = self.xg.GrammarMatcher(grammar)
        self.assertTrue(matcher.accept_string("[1,  "))
        mask = self.xg.allocate_token_bitmask(1, 257)
        matcher.fill_next_token_bitmask(mask)
        self.assertFalse(int(mask[0, ord(" ") // 32]) & (1 << (ord(" ") % 32)))
        before = mask.clone()
        self.assertTrue(matcher.accept_token(ord("2")))
        matcher.rollback(1)
        matcher.fill_next_token_bitmask(mask)
        self.assertTrue(torch.equal(before, mask))
        self.assertTrue(matcher.accept_string("2]"))
        self.assertTrue(matcher.accept_token(256))
        self.assertTrue(matcher.is_terminated())


if __name__ == "__main__":
    unittest.main()
