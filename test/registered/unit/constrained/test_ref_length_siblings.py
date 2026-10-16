"""Synthetic reference siblings: validation oracle and installed native grammar."""
import copy
import json

import pytest
import xgrammar as xgr
from jsonschema import Draft202012Validator
from sglang.srt.constrained.json_schema_validation import normalize_json_schema_resources


CASES = [
    ({"type": "string"}, {"minLength": 2, "maxLength": 3}),
    ({"type": ["string", "null"]}, {"minLength": 2, "type": "string"}),
    ({"type": "string", "maxLength": 5}, {"minLength": 2, "maxLength": 3, "type": "string"}),
    ({"anyOf": [{"type": "string"}, {"type": "null"}]}, {"minLength": 2, "maxLength": 3}),
    ({"enum": ["a", "abc", "abcd", None, 1]}, {"minLength": 2, "maxLength": 3}),
    ({"type": "object", "additionalProperties": False}, {"minLength": 0, "type": "object"}),
    ({"type": "string", "pattern": "^[a-z]+$"}, {"minLength": 0}),
    ({"type": "string"}, {"minLength": 0, "maxLength": 3}),
    ({"type": "string", "minLength": 2}, {"maxLength": 3}),
    ({"type": ["string", "null"]}, {"minLength": 3, "maxLength": 2}),
    ({"type": ["string", "null"]}, {"minLength": 2, "type": ["string", "null"]}),
    ({"type": "number"}, {"minLength": 2, "type": "integer"}),
    ({"const": "abc"}, {"minLength": 2, "maxLength": 3}),
    ({"type": "string", "pattern": "^[a-z]+$"}, {"minLength": 2, "maxLength": 3}),
    ({"oneOf": [{"type": "string"}, {"type": "null"}]}, {"minLength": 2, "maxLength": 3}),
]


@pytest.mark.parametrize("target,siblings", CASES)
def test_native_reference_siblings(target, siblings):
    schema = {"$ref": "#/$defs/S", "$defs": {"S": target}, **siblings}
    old = copy.deepcopy(schema)
    normalized = normalize_json_schema_resources(schema)
    assert schema == old
    assert normalize_json_schema_resources(normalized) == normalized
    grammar = xgr.Grammar.from_json_schema(normalized, strict_mode=False)
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo(["x", "EOS"], stop_token_ids=1)).compile_grammar(grammar)
    oracle = Draft202012Validator(schema)
    for value in [None, True, 0, 1, 1.5, "", "a", "ab", "abc", "abcd", "abcde", "ABC", "é", "你好", [], {}]:
        matcher = xgr.GrammarMatcher(compiled)
        actual = matcher.accept_string(json.dumps(value, ensure_ascii=False)) and matcher.is_completed()
        assert actual == oracle.is_valid(value), (target, siblings, value)


@pytest.mark.parametrize("siblings", [
    {"minLength": False}, {"minLength": -1}, {"minLength": 1.5},
    {"maxLength": True}, {"maxLength": -1}, {"maxLength": 1.5},
    {"minLength": 2, "type": []}, {"minLength": 2, "type": ["invalid"]},
    {"minLength": 2, "pattern": "^a"},
])
def test_invalid_or_unproved_ref_siblings_remain_explicit(siblings):
    schema = {"$ref": "#/$defs/S", "$defs": {"S": {"type": "string"}}, **siblings}
    with pytest.raises(ValueError):
        normalize_json_schema_resources(schema)


def test_chain_bounds_and_derived_cache_identity():
    schema = {"type": "object", "properties": {
        "a": {"$ref": "#/$defs/B", "minLength": 2, "maxLength": 3},
        "b": {"$ref": "#/$defs/B", "minLength": 2, "maxLength": 4},
        "c": {"$ref": "#/$defs/B", "minLength": 2, "maxLength": 3},
    }, "$defs": {
        "A": {"type": ["string", "null"], "maxLength": 5},
        "B": {"$ref": "#/$defs/A", "minLength": 1, "type": "string"},
    }}
    normalized = normalize_json_schema_resources(schema)
    properties = normalized["properties"]
    assert properties["a"] == properties["c"]
    assert properties["a"] != properties["b"]
    assert len(normalized["$defs"]) == 2
    oracle = Draft202012Validator(schema)
    grammar = xgr.Grammar.from_json_schema(normalized, strict_mode=False)
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo(["x", "EOS"], stop_token_ids=1)).compile_grammar(grammar)
    for a in ["a", "ab", "abc", "abcd", None]:
        for b in ["a", "ab", "abc", "abcd", "abcde", None]:
            value = {"a": a, "b": b, "c": "ab"}
            matcher = xgr.GrammarMatcher(compiled)
            assert (matcher.accept_string(json.dumps(value)) and matcher.is_completed()) == oracle.is_valid(value)


def test_actual_k3_tool_path_preserves_length_siblings():
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.kimik3_structural_tag import _tool_arguments_format
    from xgrammar.structural_tag import StructuralTag, TagFormat
    schema = {"type": "object", "properties": {
        "value": {"$ref": "#/$defs/S", "type": "string", "minLength": 2, "maxLength": 3},
    }, "required": ["value"], "additionalProperties": False,
        "$defs": {"S": {"type": ["string", "null"]}}}
    tool = Tool(type="function", function={"name": "synthetic", "parameters": schema, "strict": True})
    tag = StructuralTag(format=TagFormat(begin="[", end="]", content=_tool_arguments_format(tool)))
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo(["x", "EOS"], stop_token_ids=1)).compile_structural_tag(tag)
    for value in ["", "a", "ab", "abc", "abcd", "你好", "éé"]:
        wire = '[<|open|>argument key="value" type="json-string"<|sep|>' + json.dumps(value, ensure_ascii=False) + '<|close|>argument<|sep|>]'
        matcher = xgr.GrammarMatcher(compiled)
        assert (matcher.accept_string(wire) and matcher.is_completed()) == (2 <= len(value) <= 3)
    matcher = xgr.GrammarMatcher(compiled)
    assert not matcher.accept_string('[<|open|>argument key="value" type="string"<|sep|>ab<|close|>argument<|sep|>]')
