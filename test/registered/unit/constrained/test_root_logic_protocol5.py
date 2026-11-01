"""Bounded K3 root logic acceptance, independent from native presence internals."""
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


def build(schema):
    before = copy.deepcopy(schema)
    result = _tool_arguments_format(Tool(type="function", function={"name": "synthetic", "strict": True, "parameters": schema}))
    assert schema == before
    return result


def compile_schema(schema):
    return xgr.GrammarCompiler(xgr.TokenizerInfo(["x", "[EOS]"], stop_token_ids=1)).compile_structural_tag(
        StructuralTag(format=TagFormat(begin="[", end="]", content=build(schema))))


def accepts(compiled, values):
    wire = "["
    for key, value in values.items():
        kind = ("json-string" if isinstance(value, str) else "null" if value is None else
                "boolean" if isinstance(value, bool) else "array" if isinstance(value, list) else
                "object" if isinstance(value, dict) else "number")
        wire += f'<|open|>argument key="{key}" type="{kind}"<|sep|>{json.dumps(value)}<|close|>argument<|sep|>'
    matcher = xgr.GrammarMatcher(compiled)
    return matcher.accept_string(wire + "]") and matcher.is_completed()


def obj(**kwargs):
    return dict(type="object", additionalProperties=False, properties={key: {"type": "integer"} for key in "cab"}, **kwargs)


@pytest.mark.parametrize("logic", [
    {"not": {"required": ["a", "b"]}},
    {"not": {"type": "object", "required": ["a"]}},
    {"oneOf": [{"required": ["a"]}, {"required": ["b"]}]},
    {"oneOf": [{"required": ["c"]}]},
])
@pytest.mark.parametrize("dependencies", [{}, {"c": ["b"]}, {"a": ["b"], "b": ["a"]}])
def test_presence_against_schema_oracle(logic, dependencies):
    schema = obj(minProperties=1, dependentRequired=dependencies, **logic)
    validator = Draft202012Validator(schema)
    samples = [{key: 1 for key in order} for size in range(4) for order in itertools.permutations("cab", size)]
    if not any(validator.is_valid(value) for value in samples):
        with pytest.raises((ValueError, RuntimeError), match="no values|presence_unsatisfiable"):
            compile_schema(schema)
        return
    compiled = compile_schema(schema)
    for value in samples:
        assert accepts(compiled, value) == validator.is_valid(value), (schema, value)


@pytest.mark.parametrize("negative", [False, {"type": "string"}, {"type": ["null", "array"]}])
def test_not_constants_preserve_base(negative):
    compiled = compile_schema(obj(required=["a"], **{"not": negative}))
    assert accepts(compiled, {"a": 1})
    assert not accepts(compiled, {})
    assert not accepts(compiled, {"a": "wrong"})


@pytest.mark.parametrize("negative", [True, {}, {"type": "object"}, {"type": ["object", "null"]}, {"required": []}])
def test_not_constants_unsatisfiable(negative):
    with pytest.raises(ValueError, match="no.*values"):
        build(obj(**{"not": negative}))


def test_missing_false_and_composite_empty_values():
    schema = obj(**{"not": {"required": ["missing"]}})
    assert accepts(compile_schema(schema), {"a": 1, "b": 1})
    schema = obj(oneOf=[{"required": ["a"]}, {"required": ["b"]}])
    schema["properties"]["a"] = {"type": "array", "items": False, "minItems": 1}
    compiled = compile_schema(schema)
    assert accepts(compiled, {"b": 1})
    assert not accepts(compiled, {})
    schema["properties"]["b"] = {"type": "integer", "minimum": 5, "maximum": 2}
    with pytest.raises(ValueError, match="no values"):
        build(schema)


def branch(label, extra=None):
    properties = {"kind": {"type": "string", "const": label}, "value": extra or {"type": "integer", "minimum": 1}}
    return {"type": "object", "properties": properties, "required": ["kind", "value"], "additionalProperties": False}


def test_disjoint_closed_branches_and_intersections():
    schema = {"type": "object", "oneOf": [branch("left"), branch("right", {"type": "string", "minLength": 2})]}
    compiled = compile_schema(schema)
    oracle = Draft202012Validator(schema)
    for label, value in itertools.product(["left", "right", "unknown"], [0, 1, "a", "ab"]):
        for order in [("kind", "value"), ("value", "kind")]:
            data = {key: (label if key == "kind" else value) for key in order}
            assert accepts(compiled, data) == oracle.is_valid(data)
    for parent in [{"required": ["kind"]}, {"additionalProperties": False}, {"properties": {"x": True}}]:
        with pytest.raises(ValueError, match="parent assertions"):
            build(dict(schema, **parent))
    overlapping = {"oneOf": [branch("same"), branch("same")]}
    with pytest.raises(ValueError, match="discriminator"):
        build(overlapping)


def test_finite_domains_keep_sibling_assertions_and_nested_values():
    left = branch("left", {"type": "string", "enum": ["x", "long"], "minLength": 2})
    right = branch("right", {"type": "object", "properties": {"nested": {"type": "array", "items": {"type": "integer"}, "minItems": 1}},
                             "required": ["nested"], "additionalProperties": False})
    compiled = compile_schema({"oneOf": [left, right]})
    assert accepts(compiled, {"kind": "left", "value": "long"})
    assert not accepts(compiled, {"kind": "left", "value": "x"})
    assert accepts(compiled, {"kind": "right", "value": {"nested": [1]}})
    assert not accepts(compiled, {"kind": "right", "value": {"nested": []}})


def test_branch_bound_and_overlap_at_eight():
    branches = [branch(str(i)) for i in range(8)]
    compiled = compile_schema({"oneOf": branches})
    assert accepts(compiled, {"value": 1, "kind": "7"})
    with pytest.raises(ValueError, match="at most 8"):
        build({"oneOf": branches + [branch("8")]})
    branches[-1] = branch("0")
    with pytest.raises(ValueError, match="discriminator"):
        build({"oneOf": branches})


@pytest.mark.parametrize("change", [
    {"not": {"properties": {"a": {"const": 1}}}},
    {"not": {"required": ["a", "a"]}},
    {"oneOf": [{"required": ["a"]}, {"required": ["a"]}]},
    {"oneOf": [{"required": ["a", "b"]}, {"required": ["b"]}]},
    {"not": {"required": ["a"]}, "oneOf": [{"required": ["b"]}]},
    {"not": {"required": ["a"]}, "maxProperties": 1},
    {"not": {"required": ["a"]}, "additionalProperties": True},
])
def test_unproven_root_shapes_refuse(change):
    schema = obj()
    schema.update(change)
    with pytest.raises(ValueError):
        build(schema)


def test_unproven_value_is_not_deleted_and_old_native_gate(monkeypatch):
    schema = obj(**{"not": {"required": ["a", "b"]}})
    schema["properties"]["a"] = {"type": "string", "pattern": "^a", "minLength": 2}
    with pytest.raises(ValueError, match="unsupported active"):
        build(schema)
    monkeypatch.delattr(_core, "SupportsUnorderedConstraintsV2")
    with pytest.raises(ValueError, match="v2"):
        build(obj(**{"not": {"required": ["a"]}}))


def test_request_local_witness_dedup_keeps_distinct_languages(monkeypatch):
    from sglang.srt.function_call import kimik3_root_constraints as proof
    original = proof.verify_native_witnesses
    records = []
    def verify(arguments, witnesses, schemas, verified=None):
        before = set(verified or ())
        original(arguments, witnesses, schemas, verified)
        records.append((len(before), len(verified or ())))
    monkeypatch.setattr(proof, "verify_native_witnesses", verify)
    branches = []
    for i in range(3):
        item = branch(str(i))
        item["properties"].update({"a": {"type": "integer", "minimum": 0},
                                   "b": {"type": "integer", "minimum": -1}})
        branches.append(item)
    build({"oneOf": branches})
    assert records == [(0, 4), (4, 5), (5, 6)]
    records.clear()
    build({"oneOf": branches})
    assert records[0] == (0, 4)  # No state survives another request.


@pytest.mark.parametrize("schema", [
    {"type": "number", "minimum": 1e308, "maximum": 1.7e308},
    {"type": "number", "exclusiveMinimum": 1e308},
    {"type": "number", "exclusiveMaximum": -1e308},
    {"type": "number", "exclusiveMinimum": -1e308, "exclusiveMaximum": 1e308},
    {"type": "integer", "exclusiveMinimum": 10**25, "maximum": 10**25 + 3},
])
def test_numeric_witness_is_independently_valid(schema):
    from sglang.srt.function_call.kimik3_root_constraints import EMPTY, prove_value
    reduced, witness = prove_value(schema, schema)
    assert witness is not EMPTY
    assert Draft202012Validator(schema).is_valid(witness)
    assert Draft202012Validator(reduced).is_valid(witness)


def test_new_value_oneof_type_proof_is_conservative():
    from sglang.srt.function_call.kimik3_root_constraints import prove_value
    schema = {"oneOf": [{"maxLength": 2}, {"type": "null"}]}
    with pytest.raises(ValueError, match="disjointness"):
        prove_value(schema, schema)
