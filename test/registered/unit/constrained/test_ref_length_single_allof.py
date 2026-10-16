"""Synthetic transparent allOf/ref-length cases; no captured request schemas."""
import copy
import json

import pytest
import xgrammar as xgr
from jsonschema import validators

from sglang.srt.constrained.json_schema_intersections import intersect_ref_min_length
from sglang.srt.constrained.json_schema_validation import (
    _iter_subschemas,
    normalize_json_schema_resources as normalize,
    validate_xgrammar_json_schema,
)


VALUES = [None, True, 0, 1, 1.5, "", "a", "ab", "abc", "abcd", "abcde", "ABC", "甲乙", [], {}, {"child": "a"}, {"child": "abc"}, {"child": 1}]


def wrap(branch, **metadata):
    return {"allOf": [branch], **metadata}


def reference(target, **siblings):
    return {"$ref": "#/$defs/Value", "$defs": {"Value": target}, **siblings}


def assert_oracle_native(schema, values=VALUES):
    before = copy.deepcopy(schema)
    normalized = normalize(schema)
    assert schema == before
    assert normalize(normalized) == normalized
    oracle = validators.validator_for(schema)(schema)
    normalized_oracle = validators.validator_for(normalized)(normalized)
    validate_xgrammar_json_schema(normalized)
    grammar = xgr.Grammar.from_json_schema(normalized, strict_mode=False)
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo(["x", "EOS"], stop_token_ids=1)).compile_grammar(grammar)
    for value in values:
        expected = oracle.is_valid(value)
        assert normalized_oracle.is_valid(value) == expected, value
        matcher = xgr.GrammarMatcher(compiled)
        actual = matcher.accept_string(json.dumps(value, ensure_ascii=False, separators=(",", ":"))) and matcher.is_completed()
        assert actual == expected, value
    return normalized


@pytest.mark.parametrize("target", [
    wrap({"type": "string"}),
    wrap({"type": "string", "pattern": "^[a-z]*$", "maxLength": 5}),
    wrap({"type": ["string", "null"]}),
    wrap({"enum": ["a", "abc", "abcd", None, 1]}),
    wrap({"anyOf": [{"type": "string", "maxLength": 4}, {"type": "integer"}]}),
    wrap({"oneOf": [{"type": "string"}, {"type": "null"}]}),
    wrap(True),
    wrap(wrap({"type": "string", "maxLength": 3}, title="inner"), title="outer", examples=[{"$ref": "literal data"}]),
])
def test_transparent_wrapper_oracle_and_native(target):
    assert_oracle_native(reference(target, minLength=2, maxLength=4))


@pytest.mark.parametrize("branch", [False, {"type": "string", "maxLength": 1}])
def test_proven_empty_wrapper_is_not_a_generic_failure(branch):
    schema = reference(wrap(branch), minLength=2)
    normalized = normalize(schema)
    assert normalize(normalized) == normalized
    assert intersect_ref_min_length(wrap(branch), 2) is False
    oracle = validators.validator_for(schema)(schema)
    assert not any(oracle.is_valid(value) for value in VALUES)


def test_nested_ref_constraints_remain_explicit_with_resource_scope():
    # The inner fragment names the child resource's Leaf, not the root's Leaf.
    schema = reference(wrap({
        "type": "object", "properties": {"child": {"$ref": "#/$defs/Leaf"}},
        "required": ["child"], "additionalProperties": False,
    }, **{"$id": "child", "$defs": {"Leaf": {"type": "string", "minLength": 3}}}), minLength=2)
    schema["$defs"]["Leaf"] = {"type": "integer"}
    before = copy.deepcopy(schema)
    oracle = validators.validator_for(schema)(schema)
    assert oracle.is_valid({"child": "abc"}) and not oracle.is_valid({"child": 1})
    with pytest.raises(ValueError, match="unsupported_reference_intersection"):
        normalize(schema)
    assert schema == before


def test_inherited_dialect_does_not_drop_active_assertions():
    schema = reference(wrap({
        "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "dependencies": {"a": ["b"]}, "additionalProperties": False,
    }, **{"$schema": "http://json-schema.org/draft-07/schema#"}), minLength=2)
    before = copy.deepcopy(schema)
    oracle = validators.validator_for(schema)(schema)
    assert oracle.is_valid({"a": 1, "b": 2}) and not oracle.is_valid({"a": 1})
    with pytest.raises(ValueError, match="unsupported_reference_intersection"):
        normalize(schema)
    assert schema == before


def test_bound_resource_id_and_unused_definitions_are_safe():
    target = wrap({"type": "string", "maxLength": 3}, **{
        "$id": "child", "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": {"unused": {"$ref": "#/definitions/missing"}},
    })
    normalized = assert_oracle_native(reference(target, minLength=2))
    assert any(node.get("$schema") == "http://json-schema.org/draft-07/schema#" for node, _ in _iter_subschemas(normalized))


def test_explicit_branch_dialect_overrides_wrapper_dialect():
    schema = reference(wrap({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "string", "maxLength": 3,
    }, **{"$schema": "http://json-schema.org/draft-07/schema#"}), minLength=2)
    assert_oracle_native(schema)


def test_legacy_ref_siblings_still_follow_the_legacy_dialect():
    schema = {"$schema": "http://json-schema.org/draft-07/schema#", "$ref": "#/definitions/Value",
              "minLength": 5, "definitions": {"Value": wrap({"type": "string", "maxLength": 3})}}
    assert_oracle_native(schema)


@pytest.mark.parametrize("target", [
    {"allOf": []},
    {"allOf": [{"type": "string"}, {"maxLength": 3}]},
    {"allOf": [{"type": "string"}], "maxLength": 3},
    {"allOf": [{"type": "string"}], "type": "string"},
    wrap({"oneOf": [{"type": "string"}, {"type": "string", "pattern": "^a"}]}),
    wrap({"type": "string", "format": "date"}),
])
def test_unproven_intersections_are_explicit(target):
    with pytest.raises(ValueError, match="unsupported_reference_intersection"):
        normalize(reference(target, minLength=2))


def test_direct_nested_ref_and_raw_resource_wrapper_are_not_silently_flattened():
    schema = reference(wrap({"$ref": "#/$defs/Leaf"}, **{
        "$id": "child", "$defs": {"Leaf": {"type": "string", "maxLength": 3}},
    }), minLength=2)
    schema["$defs"]["Leaf"] = {"type": "integer"}
    oracle = validators.validator_for(schema)(schema)
    assert oracle.is_valid("abc") and not oracle.is_valid("a")
    with pytest.raises(ValueError, match="unsupported_reference_intersection"):
        normalize(schema)
    # Direct callers must bind resources first; $id is not a harmless annotation.
    with pytest.raises(ValueError):
        intersect_ref_min_length(wrap({"type": "string"}, **{"$id": "child"}), 2)


def test_wrapper_expansion_consumes_the_existing_depth_and_node_budgets():
    target = {"type": "string"}
    for _ in range(32):
        target = wrap(target)
    assert intersect_ref_min_length(target, 2)["minLength"] == 2
    with pytest.raises(ValueError, match="depth limit"):
        intersect_ref_min_length(wrap(target), 2)
    with pytest.raises(ValueError, match="node limit"):
        intersect_ref_min_length(wrap(wrap({"type": "string"})), 2, _budget=[2])


def test_instance_literals_in_a_wrapped_finite_domain_are_not_schemas():
    literal = {"$ref": "#/$defs/missing", "minLength": 99}
    assert_oracle_native(reference(wrap({"const": literal}), minLength=2), [literal, {}, "ab", None])


@pytest.mark.parametrize("target", [
    wrap({"type": "string", "pattern": "^[a-z]*$", "maxLength": 3}),
    wrap({"enum": ["a", "abc", None, 1]}),
    wrap({"type": "integer", "const": 1}),
])
def test_actual_tool_path_uses_the_shared_wrapper_reducer(target):
    from xgrammar.structural_tag import StructuralTag, TagFormat
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.kimik3_structural_tag import _tool_arguments_format
    schema = {"type": "object", "properties": {"v": {"$ref": "#/$defs/Value", "minLength": 2}},
              "required": ["v"], "additionalProperties": False,
              "$defs": {"Value": target}}
    old = copy.deepcopy(schema)
    tool = Tool(type="function", function={"name": "synthetic", "parameters": schema, "strict": True})
    tag = StructuralTag(format=TagFormat(begin="[", end="]", content=_tool_arguments_format(tool)))
    assert schema == old
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo(["x", "EOS"], stop_token_ids=1)).compile_structural_tag(tag)
    oracle = validators.validator_for(schema)(schema)
    for value in ["", "a", "ab", "abc", "abcd", "ABC", "甲乙", 0, 1, 2, None, True]:
        kind = "json-string" if isinstance(value, str) else "null" if value is None else "boolean" if isinstance(value, bool) else "number"
        wire = f'[<|open|>argument key="v" type="{kind}"<|sep|>' + json.dumps(value, ensure_ascii=False) + '<|close|>argument<|sep|>]'
        matcher = xgr.GrammarMatcher(compiled)
        assert (matcher.accept_string(wire) and matcher.is_completed()) == oracle.is_valid({"v": value}), value


@pytest.mark.parametrize("leaf", [
    {"type": "integer", "multipleOf": 2},
    {"type": "integer", "minimum": 0},
    {"enum": [1, 2], "minimum": 2},
    {"type": "string", "enum": ["ab"], "pattern": "^a"},
    {"type": "string", "const": "abc", "pattern": "^a"},
    {"type": "array", "items": {"type": "integer", "multipleOf": 2}},
])
def test_unproved_wrapper_values_refuse_in_normalizer_and_tool(leaf):
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.kimik3_structural_tag import _tool_arguments_format
    schema = reference(wrap(leaf), minLength=2)
    with pytest.raises(ValueError, match="unsupported_reference_intersection"):
        normalize(schema)
    parameters = {"type": "object", "properties": {"v": {"$ref": "#/$defs/Value", "minLength": 2}},
                  "required": ["v"], "additionalProperties": False, "$defs": schema["$defs"]}
    with pytest.raises(ValueError, match="unsupported_reference_intersection"):
        _tool_arguments_format(Tool(type="function", function={"name": "synthetic", "strict": True, "parameters": parameters}))


@pytest.mark.parametrize("leaf", [
    {"type": "integer", "const": "abc"},
    {"type": "integer", "enum": ["abc"]},
    {"type": "string", "enum": [1, None]},
    {"type": "null", "const": True},
])
def test_conflicting_finite_types_are_proven_empty_instead_of_native_const_precedence(leaf):
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.kimik3_structural_tag import _tool_arguments_format
    target = wrap(leaf)
    schema = reference(target, minLength=2)
    normalized = normalize(schema)
    assert normalize(normalized) == normalized
    assert intersect_ref_min_length(target, 2) is False
    assert not any(validators.validator_for(schema)(schema).is_valid(value) for value in VALUES)
    parameters = {"type": "object", "properties": {"v": {"$ref": "#/$defs/Value", "minLength": 2}},
                  "required": ["v"], "additionalProperties": False, "$defs": schema["$defs"]}
    with pytest.raises(ValueError, match="accepts no values"):
        _tool_arguments_format(Tool(type="function", function={"name": "synthetic", "strict": True, "parameters": parameters}))
