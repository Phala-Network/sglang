"""Synthetic presence conditionals through the real K3/native grammar path.

No production schemas or inference are used. An independent JSON Schema oracle
checks every presence subset/order; unsupported conditions must still refuse.
"""
import copy
import itertools
import json

import pytest
import xgrammar as xgr
from jsonschema import Draft6Validator, Draft7Validator, Draft202012Validator
from xgrammar.base import _core
from xgrammar.structural_tag import StructuralTag, TagFormat

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call import kimik3_structural_tag as k3


def conditional(trigger="a", targets=("b",)):
    return {"if": {"required": [trigger]}, "then": {"required": list(targets)}}


def closed(**changes):
    return dict(
        type="object", additionalProperties=False,
        properties={key: {"type": "integer", "minimum": 1} for key in "cabd"},
        **changes,
    )


def build(schema):
    before = copy.deepcopy(schema)
    result = k3._tool_arguments_format(Tool(type="function", function={
        "name": "synthetic", "strict": True, "parameters": schema,
    }))
    assert schema == before
    return result


def compile_schema(schema):
    return xgr.GrammarCompiler(
        xgr.TokenizerInfo(["x", "EOS"], stop_token_ids=1), max_threads=1,
    ).compile_structural_tag(StructuralTag(
        format=TagFormat(begin="[", end="]", content=build(schema)),
    ))


def accepts(compiled, data):
    pieces = ["["]
    for key, value in data.items():
        kind = (
            "json-string" if isinstance(value, str) else
            "null" if value is None else "boolean" if isinstance(value, bool) else
            "object" if isinstance(value, dict) else "array" if isinstance(value, list)
            else "number"
        )
        pieces.append(
            f'<|open|>argument key="{key}" type="{kind}"<|sep|>'
            + json.dumps(value, ensure_ascii=False)
            + "<|close|>argument<|sep|>"
        )
    matcher = xgr.GrammarMatcher(compiled)
    return matcher.accept_string("".join(pieces) + "]") and matcher.is_completed()


@pytest.mark.parametrize("changes,condition", [
    ({}, conditional()),
    ({}, conditional("c", ("a", "b"))),
    ({"dependentRequired": {"c": ["d"], "a": ["b"]}}, conditional("c", ("a", "b"))),
    ({"dependentRequired": {"a": ["a", "c", "c"], "c": ["d"]}}, conditional()),
    ({"dependentRequired": {"b": ["c"]}, "required": ["a"]}, conditional()),
    ({}, conditional("a", ("a",))),
    ({}, conditional("a", ())),
    ({}, conditional("missing", ("a",))),
    ({}, conditional("a", ("missing",))),
    ({"minProperties": 1}, conditional()),
    ({"not": {"required": ["a", "b"]}}, conditional()),
    ({"oneOf": [{"required": ["a"]}, {"required": ["c"]}]}, conditional()),
])
def test_native_conditionals_preserve_all_presence_subsets_and_orders(changes, condition):
    schema = closed(**changes, **condition)
    compiled = compile_schema(schema)
    oracle = Draft202012Validator(schema)
    for size in range(5):
        for names in itertools.permutations("cabd", size):
            value = dict.fromkeys(names, 1)
            assert accepts(compiled, value) == oracle.is_valid(value), (schema, value)
    assert not accepts(compiled, {"a": 1, "b": 0})
    assert not accepts(compiled, {"a": 1, "b": "wrong"})
    assert not accepts(compiled, {"a": 1, "b": 1, "extra": 1})


@pytest.mark.parametrize("dialect,validator", [
    ("http://json-schema.org/draft-06/schema#", Draft6Validator),
    ("http://json-schema.org/draft-07/schema#", Draft7Validator),
    ("https://json-schema.org/draft/2020-12/schema", Draft202012Validator),
])
def test_condition_type_annotations_and_selected_dialect(dialect, validator):
    schema = closed(**conditional())
    schema["$schema"] = dialect
    schema["if"].update(type="object", description="synthetic trigger")
    schema["then"].update(type="object", title="synthetic consequence")
    compiled = compile_schema(schema)
    oracle = validator(schema)
    for value in [{}, {"a": 1}, {"b": 1}, {"a": 1, "b": 1}]:
        assert accepts(compiled, value) == oracle.is_valid(value)


def test_no_type_root_and_object_domain_boundary():
    schema = closed(**conditional())
    schema.pop("type")
    # Tool arguments are always an object. JSON Schema's vacuous required
    # behavior on a scalar does not add scalar outputs to the XTML protocol.
    compiled = compile_schema(schema)
    oracle = Draft202012Validator(schema)
    for value in [{}, {"a": 1}, {"b": 1}, {"a": 1, "b": 1}]:
        assert accepts(compiled, value) == oracle.is_valid(value)
    for kind in ["null", "string", "number", "boolean", "array"]:
        with pytest.raises(ValueError, match="object schema"):
            build(dict(schema, type=kind))


def test_draft4_remains_unsupported_at_root_and_referenced_resource():
    schema = closed(**conditional())
    schema["$schema"] = "http://json-schema.org/draft-04/schema#"
    with pytest.raises(ValueError, match="unsupported_schema_dialect"):
        build(schema)
    with pytest.raises(ValueError, match="unsupported_schema_dialect"):
        build({"$ref": "#/$defs/Params", "$defs": {"Params": schema}})


def test_resource_dialects_and_ref_rehoming_preserve_conditional_values():
    target = closed(**conditional())
    target["properties"]["b"] = {
        "type": "object", "properties": {"inner": {"$ref": "#/$defs/Integer"}},
        "required": ["inner"], "additionalProperties": False,
    }
    # A modern resource lifts a root ref and a descendant value ref. Lowering
    # changes only root presence; extraction must preserve the value's scope.
    schema = {"$ref": "#/$defs/Params", "$defs": {
        "Params": target, "Integer": {"type": "integer", "minimum": 1},
    }}
    compiled = compile_schema(schema)
    oracle = Draft202012Validator(schema)
    for value in [
        {}, {"a": 1}, {"b": {"inner": 1}}, {"a": 1, "b": {"inner": 1}},
        {"a": 1, "b": {"inner": 0}}, {"a": 1, "b": {}},
        {"a": 1, "b": {"inner": 1, "extra": 1}},
    ]:
        assert accepts(compiled, value) == oracle.is_valid(value), value
    # The referenced resource explicitly selects draft 7 under a modern root;
    # its if/then remains active while a modern-only dependency is annotation.
    target["$id"] = "https://synthetic.invalid/params"
    target["$schema"] = "http://json-schema.org/draft-07/schema#"
    target["dependentRequired"] = {"d": ["c"]}
    target["properties"]["b"] = {"type": "integer", "minimum": 1}
    compiled = compile_schema(schema)
    for value in [{"d": 1}, {"a": 1}, {"a": 1, "b": 1}]:
        assert accepts(compiled, value) == Draft202012Validator(schema).is_valid(value)


def test_conditional_dependency_and_ref_length_wrapper_combine_without_weakening():
    schema = closed(**conditional())
    schema["properties"]["b"] = {
        "$ref": "#/$defs/Text", "minLength": 2, "maxLength": 3,
    }
    schema["$defs"] = {
        "Text": {"allOf": [{"type": "string", "maxLength": 4}]},
    }
    compiled = compile_schema(schema)
    oracle = Draft202012Validator(schema)
    samples = [{}, {"a": 1}]
    for text in ["", "a", "ab", "abc", "abcd", "甲乙", None, 3]:
        samples.extend([{"b": text}, {"a": 1, "b": text}, {"b": text, "a": 1}])
    for value in samples:
        assert accepts(compiled, value) == oracle.is_valid(value), value


def test_false_and_missing_consequences_are_never_dropped():
    schema = closed(**conditional())
    schema["properties"]["b"] = False
    compiled = compile_schema(schema)
    assert accepts(compiled, {"d": 1})
    for value in [{"a": 1}, {"a": 1, "b": 1}, {"b": 1}]:
        assert not accepts(compiled, value)
    schema["required"] = ["a"]
    with pytest.raises(ValueError, match="no values"):
        build(schema)


@pytest.mark.parametrize("wrapper", [{}, {"not": False}, {"not": {"type": "string"}}])
def test_conditional_value_proof_cannot_be_bypassed_by_root_wrappers(wrapper):
    schema = closed(**conditional(), **wrapper)
    schema["properties"]["b"] = {"type": "integer", "multipleOf": 2}
    with pytest.raises(ValueError, match="multipleOf|ignored"):
        build(schema)
    schema["properties"]["b"] = {"type": "string", "pattern": "^a", "minLength": 2}
    with pytest.raises(ValueError, match="unsupported active"):
        build(schema)


def test_conditional_const_siblings_remain_assertions_in_actual_native_grammar():
    schema = closed(**conditional())
    schema["properties"]["b"] = {"type": "integer", "const": 2, "minimum": 2}
    compiled = compile_schema(schema)
    oracle = Draft202012Validator(schema)
    for value in [{}, {"a": 1}, {"b": 2}, {"a": 1, "b": 1}, {"a": 1, "b": 2}]:
        assert accepts(compiled, value) == oracle.is_valid(value), value
    schema["properties"]["b"]["const"] = 1
    compiled = compile_schema(schema)
    assert accepts(compiled, {"d": 1})
    for value in [{"a": 1}, {"a": 1, "b": 1}, {"a": 1, "b": 2}, {"b": 1}]:
        assert not accepts(compiled, value)
    schema["required"] = ["a"]
    with pytest.raises(ValueError, match="no values"):
        build(schema)


def test_ordinary_no_conditional_path_does_not_add_value_proof(monkeypatch):
    from sglang.srt.function_call import kimik3_root_constraints as proof
    original = proof.prove_value
    calls = []

    def counted(schema, root):
        calls.append(schema)
        return original(schema, root)

    monkeypatch.setattr(proof, "prove_value", counted)
    build(closed())
    build(closed(dependentRequired={"a": ["b"]}))
    build(closed(**{"not": False}))
    assert calls == []
    build(closed(**conditional()))
    assert len(calls) == 4


def branch(label, trigger="a", targets=("b",)):
    result = closed(required=["kind"], **conditional(trigger, targets))
    result["properties"]["kind"] = {"type": "string", "const": label}
    return result


def test_disjoint_oneof_branch_conditions_preserve_xor_and_siblings():
    schema = {"type": "object", "oneOf": [
        branch("left"), branch("right", "b", ("c",)),
    ]}
    compiled = compile_schema(schema)
    oracle = Draft202012Validator(schema)
    for label in ["left", "right", "unknown"]:
        for size in range(4):
            for names in itertools.combinations("cab", size):
                for order in itertools.permutations(("kind", *names)):
                    value = {key: label if key == "kind" else 1 for key in order}
                    assert accepts(compiled, value) == oracle.is_valid(value), value
    assert not accepts(compiled, {"kind": "left", "a": 1, "b": 0})
    assert not accepts(compiled, {"kind": "right", "b": 1, "c": "wrong"})
    with pytest.raises(ValueError, match="discriminator"):
        build({"oneOf": [branch("same"), branch("same", "b", ("c",))]})


def test_empty_conditional_branch_does_not_widen_oneof():
    left = branch("left", targets=("missing",))
    left["required"].append("a")
    schema = {"oneOf": [left, branch("right")]}
    compiled = compile_schema(schema)
    assert accepts(compiled, {"kind": "right"})
    assert not accepts(compiled, {"kind": "left", "a": 1})
    assert not accepts(compiled, {"kind": "right", "a": 1})


@pytest.mark.parametrize("condition", [
    {"if": {"required": ["a", "b"]}, "then": {"required": ["c"]}},
    {"if": {"required": []}, "then": {"required": ["b"]}},
    {"if": {"properties": {"a": {"const": 1}}}, "then": {"required": ["b"]}},
    {"if": {"required": ["a"]}, "then": {"properties": {"b": {"const": 1}}}},
    {"if": {"required": ["a"]}, "then": True},
    {"if": True, "then": {"required": ["b"]}},
    {"if": {"required": ["a"]}, "then": {"required": ["b"]}, "else": {}},
    {"if": {"required": ["a"]}},
    {"then": {"required": ["b"]}},
    {"else": {"required": ["b"]}},
    {"if": {"type": "string", "required": ["a"]}, "then": {"required": ["b"]}},
    {"if": {"required": ["a", "a"]}, "then": {"required": ["b"]}},
    {"if": {"required": "a"}, "then": {"required": ["b"]}},
    {"if": {"required": ["a"]}, "then": {"required": ["b", "b"]}},
])
def test_unproven_conditions_remain_explicit_refusals(condition):
    with pytest.raises(ValueError, match="conditional|presence"):
        build(closed(**condition))
    left = branch("left")
    left.pop("if")
    left.pop("then")
    left.update(condition)
    with pytest.raises(ValueError, match="conditional|presence"):
        build({"oneOf": [left, branch("right")]})


@pytest.mark.parametrize("changes", [
    {"additionalProperties": True},
    {"maxProperties": 2},
    {"dependentRequired": []},
    {"dependentRequired": {"a": "c"}},
    {"dependentRequired": {"a": [1]}},
    {"patternProperties": {"^a": {"type": "integer"}}},
])
def test_unsupported_siblings_are_not_lost(changes):
    schema = closed(**conditional())
    schema.update(changes)
    with pytest.raises(ValueError):
        build(schema)


def test_parent_oneof_intersection_and_native_capability_stay_guarded(monkeypatch):
    with pytest.raises(ValueError, match="parent assertions"):
        build({"oneOf": [branch("left"), branch("right")], **conditional()})
    monkeypatch.delattr(_core, "SupportsUnorderedConstraintsV1")
    with pytest.raises(ValueError, match="native unordered constraints"):
        build(closed(**conditional()))


def test_conditional_resource_limits_and_existing_edge_merge():
    schema = closed(dependentRequired={"a": ["a", "c", "c"], "c": ["d"]}, **conditional())
    before = copy.deepcopy(schema)
    lowered = k3._lower_presence_conditional(schema)
    assert lowered["dependentRequired"] == {"a": ["a", "c", "b"], "c": ["d"]}
    assert schema == before
    names = [f"p{i}" for i in range(64)]
    schema = dict(
        type="object", additionalProperties=False,
        properties={key: {"type": "integer"} for key in names},
        dependentRequired={key: list(names) for key in names},
        **conditional("p0", ("p1",)),
    )
    assert len(build(schema).elements) == 64
    compiled = compile_schema(schema)
    assert accepts(compiled, {})
    assert accepts(compiled, dict.fromkeys(names[::-1], 1))
    assert not accepts(compiled, {"p0": 1})
    schema["properties"]["p64"] = {"type": "integer"}
    with pytest.raises(ValueError, match="resource_limit"):
        build(schema)
    del schema["properties"]["p64"]
    schema["dependentRequired"]["p0"].append("missing")
    with pytest.raises(ValueError, match="resource_limit"):
        build(schema)
    schema["dependentRequired"]["p0"] = names[1:]
    schema["then"] = {"required": ["missing1", "missing2"]}
    with pytest.raises(ValueError, match="resource_limit"):
        build(schema)
    schema = closed(**conditional(targets=tuple(f"p{i}" for i in range(65))))
    with pytest.raises(ValueError, match="resource_limit"):
        build(schema)
