"""Reject unsupported combined constraints; preserve valid independent uses."""

import json
from unittest.mock import MagicMock

import pytest

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.xgrammar_backend import (
    XGrammarGrammarBackend,
    has_xgrammar_unsupported_pattern_length_combination,
)

BAD = {"type": "string", "pattern": "^[a-z]+$", "minLength": 5}


@pytest.mark.parametrize(
    "schema",
    [
        BAD,
        {**BAD, "maxLength": 8},
        {"type": "object", "properties": {"value": BAD}},
        {"anyOf": [{"type": "null"}, BAD]},
        {"$defs": {"Value": BAD}},
        {"items": BAD},
        {"prefixItems": [BAD]},
        {"additionalProperties": BAD},
        {"patternProperties": {".*": BAD}},
        {"dependentSchemas": {"x": BAD}},
    ],
)
def test_unsupported_nested_schema_is_detected(schema):
    assert has_xgrammar_unsupported_pattern_length_combination(schema)
    backend = object.__new__(XGrammarGrammarBackend)
    backend.grammar_compiler = MagicMock()
    backend.max_whitespace_cnt = 64
    backend.any_whitespace = True
    result = backend.dispatch_json(json.dumps(schema))
    assert isinstance(result, InvalidGrammarObject)
    backend.grammar_compiler.compile_json_schema.assert_not_called()


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "pattern": "^[a-z]+$"},
        {"type": "string", "minLength": 5, "maxLength": 8},
        {"const": BAD},
        {"enum": [BAD]},
        {"examples": [BAD]},
        {"properties": {"pattern": {"type": "string", "minLength": 5}}},
        {},
        True,
        False,
    ],
)
def test_keyword_like_values_are_not_schemas(schema):
    assert not has_xgrammar_unsupported_pattern_length_combination(schema)


@pytest.mark.parametrize("limit", [None, 64])
def test_builtin_json_sentinel_still_compiles(limit):
    backend = object.__new__(XGrammarGrammarBackend)
    backend.grammar_compiler = MagicMock()
    backend.max_whitespace_cnt = limit
    backend.any_whitespace = True
    backend._from_context = MagicMock(return_value="compiled")
    assert backend.dispatch_json("$$ANY$$") == "compiled"


@pytest.mark.parametrize("legacy", [False, True])
def test_structural_tags_do_not_bypass_schema_guard(legacy):
    backend = object.__new__(XGrammarGrammarBackend)
    backend.grammar_compiler = MagicMock()
    if legacy:
        tag = {
            "structures": [{"begin": "<tool>", "schema": BAD, "end": "</tool>"}],
            "triggers": ["<tool>"],
        }
    else:
        tag = {
            "type": "structural_tag",
            "format": {
                "type": "tag",
                "begin": "<tool>",
                "content": {"type": "json_schema", "json_schema": BAD},
                "end": "</tool>",
            },
        }
    assert isinstance(
        backend.dispatch_structural_tag(json.dumps(tag)), InvalidGrammarObject
    )
    backend.grammar_compiler.compile_structural_tag.assert_not_called()
