"""Synthetic strict K3 finite-object constraints against an independent validator."""
import copy
import itertools
import json

import pytest
from jsonschema import Draft202012Validator
import xgrammar as xgr
from xgrammar.base import _core
from xgrammar.structural_tag import StructuralTag, TagFormat

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.kimik3_structural_tag import _tool_arguments_format


def schema(**constraints):
    return dict(type="object", properties={name: {"type": "integer", "minimum": 1} for name in "cabd"},
                additionalProperties=False, **constraints)


def build(parameters):
    return _tool_arguments_format(Tool(type="function", function={
        "name": "synthetic", "parameters": parameters, "strict": True}))


def compile_schema(parameters):
    original = copy.deepcopy(parameters)
    tag = StructuralTag(format=TagFormat(begin="[", end="]", content=build(parameters)))
    assert parameters == original
    return xgr.GrammarCompiler(xgr.TokenizerInfo(["x", "[EOS]"], stop_token_ids=1)).compile_structural_tag(tag)


def accepts(compiled, names, value=1):
    wire = "[" + "".join(f'<|open|>argument key="{key}" type="number"<|sep|>{value}<|close|>argument<|sep|>' for key in names) + "]"
    matcher = xgr.GrammarMatcher(compiled)
    return matcher.accept_string(wire) and matcher.is_completed()


@pytest.mark.parametrize("constraints", [
    {"minProperties": 2}, {"maxProperties": 2}, {"maxProperties": 0},
    {"minProperties": 1, "maxProperties": 2, "required": ["a"]},
    {"dependentRequired": {"c": ["a"], "a": ["b"]}, "minProperties": 1},
    {"dependentRequired": {"a": ["b"], "b": ["a"]}, "minProperties": 2},
    {"dependentRequired": {"c": ["a"], "a": ["b"]}, "required": ["c"]},
    {"dependentRequired": {"c": ["missing"], "a": ["c"]}},
    {"dependentRequired": {"unknown": ["a"], "a": ["a"]}},
])
def test_independent_schema_truth_table(constraints):
    parameters = schema(**constraints)
    compiled = compile_schema(parameters)
    validator = Draft202012Validator(parameters)
    for size in range(5):
        for names in itertools.permutations("cabd", size):
            assert accepts(compiled, names) == validator.is_valid(dict.fromkeys(names, 1)), names
    assert not accepts(compiled, ["a", "a"])
    assert not accepts(compiled, ["a", "b", "c", "d"], value=0)


def test_false_dependencies_and_empty_objects():
    parameters = schema(dependentRequired={"a": ["b"], "c": ["a"]}, minProperties=1)
    parameters["properties"]["b"] = False
    compiled = compile_schema(parameters)
    assert accepts(compiled, ["d"])
    for names in [[], ["a"], ["c"], ["a", "b", "c", "d"]]:
        assert not accepts(compiled, names)
    empty = {"type": "object", "properties": {}, "additionalProperties": False, "maxProperties": 0}
    assert accepts(compile_schema(empty), [])
    empty["minProperties"] = 1
    with pytest.raises(ValueError, match="accept no values"):
        build(empty)


@pytest.mark.parametrize("changes", [
    {"minProperties": -1}, {"maxProperties": True}, {"minProperties": 1.0},
    {"minProperties": 5}, {"minProperties": 3, "maxProperties": 2},
    {"maxProperties": 0, "required": ["a"]},
    {"dependentRequired": {"a": ["missing"]}, "required": ["a"]},
    {"dependentRequired": {"a": "b"}}, {"dependentRequired": []},
    {"dependentRequired": {"a": ["b"]}, "maxProperties": 2},
    {"minProperties": 1, "additionalProperties": True},
])
def test_invalid_or_unsupported_are_explicit(changes):
    parameters = schema()
    parameters.update(changes)
    with pytest.raises(ValueError):
        build(parameters)


def test_older_native_gate(monkeypatch):
    monkeypatch.delattr(_core, "SupportsUnorderedConstraintsV1")
    with pytest.raises(ValueError, match="native unordered constraints"):
        build(schema(minProperties=1))


def test_large_optional_properties_no_presence_enumeration():
    names = [f"p{i:03}" for i in range(65)]
    parameters = dict(type="object", additionalProperties=False,
        properties={key: {"type": "integer"} for key in names}, minProperties=32,
        dependentRequired={names[i]: [names[(i+1) % 65]] for i in range(65)})
    fmt = build(parameters)
    assert len(fmt.elements) == 65 and len(fmt.dependencies) == 65
    assert len(fmt.model_dump_json()) < 100000
    compiled = compile_schema(parameters)
    assert accepts(compiled, names[::-1])
    assert not accepts(compiled, names[:-1])


@pytest.mark.parametrize("target", [
    True, False, {},
    {"type": "string"}, {"type": "string", "minLength": 3, "maxLength": 5},
    {"type": "string", "pattern": "^[a-z]+$"},
    {"type": ["string", "null"], "maxLength": 4},
    {"type": "integer", "minimum": 2},
    {"enum": ["a", "abc", 2, None]}, {"const": "abc"}, {"const": "x"},
    {"anyOf": [{"type": "string", "maxLength": 4}, {"type": "null"}]},
    {"oneOf": [{"type": "string", "pattern": "^a"}, {"type": "null"}]},
    {"type": "string", "maxLength": 1},
    {"type": ["string", "null"], "maxLength": 1},
])
def test_ref_minlength_preserves_intersection(target):
    parameters = {"type": "object", "properties": {"v": {"$ref": "#/$defs/V", "minLength": 2}},
                  "$defs": {"V": target}, "additionalProperties": False}
    compiled = compile_schema(parameters)
    validator = Draft202012Validator(parameters)
    for value in ["", "a", "ab", "abc", "abcde", "abcdef", "ABC", "甲乙", 1, 2, None]:
        wire_type = "json-string" if isinstance(value, str) else "null" if value is None else "number"
        wire = f'[<|open|>argument key="v" type="{wire_type}"<|sep|>{json.dumps(value)}<|close|>argument<|sep|>]'
        matcher = xgr.GrammarMatcher(compiled)
        actual = matcher.accept_string(wire) and matcher.is_completed()
        assert actual == validator.is_valid({"v": value}), (target, value)


def test_ref_chain_length_maximum_and_unproven_shapes():
    parameters = {"type": "object", "properties": {"v": {"$ref": "#/$defs/A", "minLength": 2}},
                  "$defs": {"A": {"$ref": "#/$defs/V", "minLength": 4}, "V": {"type": "string"}},
                  "additionalProperties": False}
    fmt = build(parameters)
    assert '"minLength":4' in fmt.model_dump_json()
    for target in [
        {"type": "string", "format": "date"},
        {"allOf": [{"type": "string"}, {"pattern": "^a"}]},
        {"type": "string", "enum": ["ab"], "pattern": "^b"},
        {"oneOf": [{"type": "string"}, {"type": "string", "pattern": "^a"}]},
    ]:
        parameters["$defs"]["V"] = target
        with pytest.raises(ValueError, match="unsupported_reference_intersection"):
            build(parameters)
