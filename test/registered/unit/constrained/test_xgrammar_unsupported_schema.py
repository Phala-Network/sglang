"""Regression tests for fail-closed XGrammar JSON Schema validation."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.srt.constrained.xgrammar_schema import (
    has_xgrammar_unsupported_json_features,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat


class TestHasXGrammarUnsupportedJsonFeatures(unittest.TestCase):
    def test_supported_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name"],
            "additionalProperties": False,
        }
        self.assertFalse(has_xgrammar_unsupported_json_features(schema))

    def test_unsupported_keywords(self):
        cases = [
            {"multipleOf": 2},
            {"uniqueItems": True},
            {"contains": {"type": "integer"}},
            {"minContains": 1},
            {"maxContains": 2},
            {"patternProperties": {"^x": {"type": "string"}}},
            {"propertyNames": {"pattern": "^x"}},
            {"dependentSchemas": {"mode": {"required": ["value"]}}},
        ]
        for schema in cases:
            with self.subTest(schema=schema):
                self.assertTrue(has_xgrammar_unsupported_json_features(schema))

    def test_unsupported_feature_nested_in_schema_positions(self):
        schema = {
            "type": "object",
            "$defs": {
                "row": {
                    "type": "object",
                    "properties": {
                        "values": {
                            "type": "array",
                            "items": {"type": "integer", "multipleOf": 3},
                        }
                    },
                }
            },
            "properties": {"row": {"$ref": "#/$defs/row"}},
        }
        self.assertTrue(has_xgrammar_unsupported_json_features(schema))

    def test_property_names_that_match_keywords_are_allowed(self):
        schema = {
            "type": "object",
            "properties": {
                "multipleOf": {"type": "number"},
                "uniqueItems": {"type": "boolean"},
                "contains": {"type": "string"},
                "dependentSchemas": {"type": "object"},
                "patternProperties": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            },
        }
        self.assertFalse(has_xgrammar_unsupported_json_features(schema))

    def test_dispatch_rejects_before_compile(self):
        backend = XGrammarGrammarBackend.__new__(XGrammarGrammarBackend)
        backend.grammar_compiler = MagicMock()
        backend.any_whitespace = True
        schema = {
            "type": "object",
            "dependentSchemas": {"mode": {"required": ["value"]}},
        }

        result = backend.dispatch_json(json.dumps(schema))

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertIn("unsupported by xgrammar", result.error_message)
        backend.grammar_compiler.compile_json_schema.assert_not_called()


class TestOpenAIChatFailClosedValidation(unittest.TestCase):
    @staticmethod
    def _serving(backend: str):
        serving = OpenAIServingChat.__new__(OpenAIServingChat)
        serving._grammar_backend = backend
        serving.tokenizer_manager = SimpleNamespace(
            server_args=SimpleNamespace(
                context_length=262144,
                allow_auto_truncate=False,
            )
        )
        serving._validate_media_content = lambda request: None
        serving._effective_tools = lambda request: []
        return serving

    @staticmethod
    def _request(schema):
        return SimpleNamespace(
            messages=[object()],
            return_sampling_mask=False,
            return_meta_info=False,
            tool_choice=None,
            max_completion_tokens=None,
            max_tokens=64,
            response_format=SimpleNamespace(
                type="json_schema",
                json_schema=SimpleNamespace(schema_=schema),
            ),
        )

    def test_xgrammar_response_format_rejected_before_generation(self):
        schema = {
            "type": "object",
            "dependentSchemas": {"mode": {"required": ["value"]}},
        }
        error = self._serving("xgrammar")._validate_request(self._request(schema))
        self.assertIn("unsupported by xgrammar", error)

    def test_supported_response_format_passes(self):
        schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        }
        error = self._serving("xgrammar")._validate_request(self._request(schema))
        self.assertIsNone(error)

    def test_other_backend_not_rejected_as_xgrammar(self):
        schema = {
            "type": "object",
            "dependentSchemas": {"mode": {"required": ["value"]}},
        }
        error = self._serving("llguidance")._validate_request(self._request(schema))
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
