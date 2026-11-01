"""Shared ref/length reducer: independent semantics and actual native grammar."""
import copy
import json

import pytest
import xgrammar as xgr
from jsonschema import Draft202012Validator
from sglang.srt.constrained.json_schema_intersections import intersect_ref_min_length
from sglang.srt.constrained.json_schema_validation import normalize_json_schema_resources as normalize


@pytest.mark.parametrize("target", [
    {"type": "string", "maxLength": 4},
    {"type": ["string", "null"], "maxLength": 2},
    {"enum": ["a", "abc", "abcd", 1, None]},
    {"const": "a"},
    {"anyOf": [{"type": "string"}, {"type": "integer"}]},
    {"oneOf": [{"type": "string"}, {"type": "null"}]},
    {"maxLength": 2},
])
def test_oracle_native_chain_and_immutability(target):
    schema = {"$ref": "#/$defs/B", "minLength": 3, "$defs": {
        "A": target, "B": {"$ref": "#/$defs/A", "minLength": 2}}}
    before = copy.deepcopy(schema)
    normalized = normalize(schema)
    assert schema == before
    assert normalize(normalized) == normalized
    oracle = Draft202012Validator(schema)
    # Native explicitly rejects an empty root language. Prove emptiness with
    # the reducer instead of treating the compiler refusal as successful parse.
    reduced = intersect_ref_min_length(target, 3)
    if reduced is False:
        assert not any(oracle.is_valid(v) for v in [None, 1, "", "a", "abc", "abcd"])
        return
    grammar = xgr.Grammar.from_json_schema(normalized, strict_mode=False)
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo(["x", "EOS"], stop_token_ids=1)).compile_grammar(grammar)
    for value in [None, True, 1, 1.5, "", "a", "ab", "abc", "abcd", "abcde", [], {}]:
        matcher = xgr.GrammarMatcher(compiled)
        actual = matcher.accept_string(json.dumps(value)) and matcher.is_completed()
        assert actual == oracle.is_valid(value), (target, value)


@pytest.mark.parametrize("branches", [
    [{"maxLength": 2}, {"type": "null"}],
    [{"type": "integer", "const": 1.0}, {"type": "number", "const": 1.0}],
])
def test_unproved_oneof_overlap_is_never_anyof(branches):
    with pytest.raises(ValueError, match="disjoint"):
        intersect_ref_min_length({"oneOf": branches}, 3)


def test_shared_target_deduplicates_and_canonicalizes():
    schema = {"type": "object", "properties": {
        str(i): {"$ref": "#/$defs/S", "minLength": 3} for i in range(100)
    }, "$defs": {"S": {"type": "string"}}}
    normalized = normalize(schema)
    assert len(normalized["$defs"]) == 1
    assert len({p["$ref"] for p in normalized["properties"].values()}) == 1
    assert normalize(normalized) == normalized


def test_instance_data_is_not_rewritten():
    literal = {"$ref": "#/$defs/missing", "minLength": 3}
    schema = {"const": literal, "$defs": {"unused": {"type": "string"}}}
    assert normalize(schema)["const"] == literal
