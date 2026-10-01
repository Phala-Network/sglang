"""Native schema no-op regression and reference-preserving normalization."""

import copy
import json
import unittest

from jsonschema import Draft202012Validator
from llguidance import LLTokenizer
from llguidance.torch import allocate_token_bitmask

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.llguidance_backend import (
    GuidanceBackend,
    GuidanceGrammar,
    _normalize_llguidance_schema_noops,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLLGuidanceSchemaNoops(unittest.TestCase):
    def setUp(self):
        self.backend = GuidanceBackend.__new__(GuidanceBackend)
        self.backend.any_whitespace = False
        self.backend.whitespace_pattern = None
        self.backend.llguidance_tokenizer = LLTokenizer("byte")

    def normalize(self, schema, values=()):
        original = copy.deepcopy(schema)
        Draft202012Validator.check_schema(schema)
        normalized = _normalize_llguidance_schema_noops(schema)
        self.assertEqual(schema, original)
        for value in values:
            self.assertEqual(
                Draft202012Validator(schema).is_valid(value),
                Draft202012Validator(normalized).is_valid(value),
            )
        return normalized

    def accepts(self, schema, value):
        grammar = self.backend.dispatch_json(json.dumps(schema))
        self.assertIsInstance(grammar, GuidanceGrammar)
        text = json.dumps(value, separators=(",", ":"))
        mask = allocate_token_bitmask(1, grammar.llguidance_tokenizer.vocab_size)
        for token in [*text.encode(), next(iter(grammar.eos_tokens))]:
            grammar.fill_vocab_mask(mask, 0)
            if not ((int(mask[0, token // 32]) >> (token % 32)) & 1):
                return False
            grammar.accept_token(token)
        return grammar.is_terminated()

    @staticmethod
    def object_schema(**extra):
        return {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
            "additionalProperties": False,
            **extra,
        }

    def test_original_property_names_positive_and_strict_type(self):
        schema = self.object_schema(propertyNames=True)
        self.normalize(schema, [{"n": 1}, {"n": "x"}])
        self.assertTrue(self.accepts(schema, {"n": 1}))
        self.assertFalse(self.accepts(schema, {"n": "x"}))

    def test_empty_property_names(self):
        schema = self.object_schema(propertyNames={})
        self.normalize(schema, [{"n": 1}])
        self.assertTrue(self.accepts(schema, {"n": 1}))

    def test_original_inapplicable_uniqueness(self):
        schema = self.object_schema(uniqueItems=True)
        self.normalize(schema, [{"n": 1}])
        self.assertTrue(self.accepts(schema, {"n": 1}))

    def test_original_contains_noops_allow_duplicates(self):
        schema = {
            "type": "array",
            "items": {"type": "integer"},
            "uniqueItems": False,
            "minContains": 1,
            "maxContains": 2,
        }
        self.normalize(schema, [[], [1, 1], ["x"]])
        self.assertTrue(self.accepts(schema, [1, 1]))
        self.assertFalse(self.accepts(schema, ["x"]))

    def test_nonarray_union(self):
        schema = {"type": ["string", "null"], "uniqueItems": True}
        self.normalize(schema, ["x", None, []])
        self.assertTrue(self.accepts(schema, "x"))
        self.assertTrue(self.accepts(schema, None))

    def test_effective_array_and_union_uniqueness_rejected(self):
        for types in ("array", ["array", "null"]):
            schema = {"type": types, "items": {"type": "integer"}, "uniqueItems": True}
            self.assertTrue(self.normalize(schema)["uniqueItems"])
            self.assertIsInstance(
                self.backend.dispatch_json(json.dumps(schema)), InvalidGrammarObject
            )

    def test_untyped_uniqueness_rejected(self):
        schema = {"uniqueItems": True}
        self.assertTrue(self.normalize(schema)["uniqueItems"])
        self.assertIsInstance(
            self.backend.dispatch_json(json.dumps(schema)), InvalidGrammarObject
        )

    def test_reachable_ref_effective_uniqueness_rejected(self):
        schema = {
            "$defs": {"a": {"type": "array", "uniqueItems": True}},
            "$ref": "#/$defs/a",
        }
        self.normalize(schema)
        self.assertIsInstance(
            self.backend.dispatch_json(json.dumps(schema)), InvalidGrammarObject
        )

    def test_nested_and_ref_noops(self):
        inner = self.object_schema(propertyNames=True, uniqueItems=True)
        schema = {
            "$defs": {"x": inner},
            "type": "array",
            "items": {"$ref": "#/$defs/x"},
        }
        self.normalize(schema, [[{"n": 1}], [{"n": "x"}]])
        self.assertTrue(self.accepts(schema, [{"n": 1}]))
        self.assertFalse(self.accepts(schema, [{"n": "x"}]))

    def test_deleted_root_pointer_and_ref_siblings(self):
        for noop in (True, {}):
            schema = self.object_schema(propertyNames=noop)
            schema["properties"]["n"] = {
                "$ref": "#/propertyNames",
                "type": "integer",
                "minimum": 2,
            }
            self.normalize(schema, [{"n": 1}, {"n": 2}, {"n": "x"}])
            self.assertTrue(self.accepts(schema, {"n": 2}))
            self.assertFalse(self.accepts(schema, {"n": 1}))

    def test_deleted_nested_escaped_pointer(self):
        schema = {
            "$defs": {"a/b~c": {"propertyNames": {}}},
            "$ref": "#/$defs/a~1b~0c/propertyNames",
            "type": "integer",
        }
        self.normalize(schema, [1, "x"])
        self.assertTrue(self.accepts(schema, 1))
        self.assertFalse(self.accepts(schema, "x"))

    def test_nested_id_scope_and_absolute_reference(self):
        schema = {
            "$id": "https://example.test/root",
            "propertyNames": True,
            "$defs": {
                "child": {
                    "$id": "child",
                    "propertyNames": {"type": "string", "minLength": 2},
                    "properties": {
                        "n": {"$ref": "#/propertyNames"},
                        "outer": {
                            "$ref": "https://example.test/root#/propertyNames",
                            "type": "integer",
                        },
                    },
                }
            },
        }
        normalized = self.normalize(schema)
        props = normalized["$defs"]["child"]["properties"]
        self.assertEqual(props["n"]["$ref"], "#/propertyNames")
        self.assertEqual(props["outer"], {"type": "integer"})
        self.assertIn("propertyNames", normalized["$defs"]["child"])

    def test_nested_id_noop_ref(self):
        schema = {
            "$id": "https://example.test/root",
            "$defs": {
                "child": {
                    "$id": "child",
                    "propertyNames": {},
                    "properties": {"n": {"$ref": "#/propertyNames", "type": "integer"}},
                }
            },
        }
        normalized = self.normalize(schema)
        self.assertEqual(
            normalized["$defs"]["child"]["properties"]["n"], {"type": "integer"}
        )

    def test_const_enum_and_annotation_data_untouched(self):
        data = {
            "propertyNames": True,
            "uniqueItems": False,
            "minContains": 1,
            "$ref": "#/propertyNames",
        }
        for keyword in ("const", "default", "examples", "enum"):
            value = [data] if keyword in ("examples", "enum") else data
            schema = {keyword: value, "propertyNames": True}
            self.assertEqual(self.normalize(schema)[keyword], value)

    def test_effective_property_names_and_contains_untouched(self):
        for schema in (
            {"type": "object", "propertyNames": {"pattern": "^a$"}},
            {
                "type": "array",
                "contains": {"type": "integer"},
                "minContains": 1,
                "maxContains": 2,
            },
        ):
            self.assertEqual(self.normalize(schema), schema)
        schema = {"type": "object", "propertyNames": {"pattern": "^a$"}}
        self.assertIsInstance(
            self.backend.dispatch_json(json.dumps(schema)), InvalidGrammarObject
        )

    def test_malformed_type_rejected(self):
        self.assertIsInstance(
            self.backend.dispatch_json('{"type":"not-a-json-schema-type"}'),
            InvalidGrammarObject,
        )

    def test_legacy_and_unknown_dialects_are_unchanged(self):
        for dialect in (
            "http://json-schema.org/draft-07/schema#",
            "https://example.test/custom-dialect",
        ):
            schema = {
                "$schema": dialect,
                "definitions": {"any": {}},
                "$ref": "#/definitions/any",
                "type": "integer",
                "propertyNames": True,
            }
            self.assertEqual(_normalize_llguidance_schema_noops(schema), schema)

    def test_nested_dialect_preserves_entire_input(self):
        for dialect in (
            "http://json-schema.org/draft-07/schema#",
            "https://example.test/custom-dialect",
        ):
            schema = {
                "propertyNames": True,
                "$defs": {"child": {"$id": "child", "$schema": dialect}},
            }
            self.assertEqual(_normalize_llguidance_schema_noops(schema), schema)

    def test_literal_dialect_does_not_disable_normalization(self):
        data = {"$schema": "https://example.test/custom-dialect"}
        for keyword in ("const", "default", "examples", "enum"):
            value = [data] if keyword in ("examples", "enum") else data
            schema = {keyword: value, "propertyNames": True}
            self.assertEqual(
                _normalize_llguidance_schema_noops(schema), {keyword: value}
            )


if __name__ == "__main__":
    unittest.main()
