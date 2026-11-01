"""Pure synthetic resource compatibility and optional actual native acceptance."""

import copy
import importlib.util
import json
import os
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

try:
    from sglang.srt.constrained.json_schema_validation import (
        UnsupportedJSONSchemaFeature,
    )
    from sglang.srt.constrained.json_schema_validation import (
        normalize_json_schema_resources as normalize,
    )
    from sglang.srt.constrained.json_schema_validation import (
        validate_xgrammar_json_schema as validate,
    )
except ModuleNotFoundError:
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/constrained/json_schema_validation.py"
    )
    spec = importlib.util.spec_from_file_location("normalization_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    normalize = module.normalize_json_schema_resources
    validate = module.validate_xgrammar_json_schema
    UnsupportedJSONSchemaFeature = module.UnsupportedJSONSchemaFeature


def fixtures():
    integer = {"type": "integer", "minimum": 2, "maximum": 5}

    def obj(value, **extra):
        return dict(
            type="object",
            properties={"v": value},
            required=["v"],
            additionalProperties=False,
            **extra,
        )

    yield "plain", obj(integer), {"v": 3}, {"v": 1}
    yield (
        "no_ref_id",
        obj(dict(integer, **{"$id": "https://example.invalid/inner"})),
        {"v": 3},
        {"v": "3"},
    )
    yield (
        "unused_root",
        obj(integer, **{"$defs": {"unused": {"$id": "inner", "$ref": "remote.json"}}}),
        {"v": 3},
        {"v": 6},
    )
    yield (
        "unused_property",
        obj(
            dict(
                integer,
                **{"$defs": {"unused": {"$id": "inner", "$ref": "remote.json"}}},
            )
        ),
        {"v": 3},
        {"v": 1},
    )
    yield (
        "ref_id_leaf",
        obj(
            {"$ref": "#/$defs/X"}, **{"$defs": {"X": dict(integer, **{"$id": "inner"})}}
        ),
        {"v": 3},
        {"v": 6},
    )
    yield (
        "child_jump",
        obj(
            {"$ref": "#/$defs/R/properties/X"},
            **{"$defs": {"R": {"$id": "inner", "properties": {"X": integer}}}},
        ),
        {"v": 3},
        {"v": 1},
    )
    yield (
        "escaped",
        obj({"$ref": "#%2F$defs%2Fa~1b"}, **{"$defs": {"a/b": integer}}),
        {"v": 3},
        {"v": 1},
    )
    yield (
        "annotation_target",
        obj({"$ref": "#/default/X"}, default={"$id": "literal", "X": integer}),
        {"v": 3},
        {"v": 6},
    )
    literal = {"$id": "data", "$ref": "data", "$defs": {"X": False}}
    yield "literal", obj({"const": literal}), {"v": literal}, {"v": {}}
    yield (
        "id_member",
        {
            "type": "object",
            "properties": {"$id": integer},
            "required": ["$id"],
            "additionalProperties": False,
        },
        {"$id": 3},
        {"$id": "3"},
    )
    yield (
        "consuming_cycle",
        obj(
            {"$ref": "#/$defs/N"},
            **{
                "$defs": {
                    "N": {
                        "anyOf": [
                            {"type": "null"},
                            {"type": "array", "items": {"$ref": "#/$defs/N"}},
                        ]
                    }
                }
            },
        ),
        {"v": [[None]]},
        {"v": [1]},
    )

    for identifier in ["inner", "https://example.invalid/root", ""]:
        resource = {
            "$id": identifier,
            "type": "array",
            "$defs": {"X": {"type": "string"}},
            "items": {"$ref": "#/$defs/X"},
        }
        schema = obj(
            resource,
            **{
                "$id": "https://example.invalid/root",
                "$defs": {"X": {"type": "integer"}},
            },
        )
        different = identifier == "inner"
        yield (
            "resource_" + repr(identifier),
            schema,
            {"v": ["text"] if different else [3]},
            {"v": [3] if different else ["text"]},
        )
    inner = {
        "$id": "inner",
        "type": "array",
        "$defs": {"X": {"type": "string"}},
        "items": {"$ref": "#/$defs/X"},
    }
    yield (
        "resource_child_jump",
        obj(
            {"$ref": "#/$defs/R/items"},
            **{"$defs": {"R": inner, "X": {"type": "integer"}}},
        ),
        {"v": "text"},
        {"v": 3},
    )
    recursive = {
        "$id": "inner",
        "type": "object",
        "properties": {"next": {"$ref": "#"}},
        "additionalProperties": False,
    }
    yield "resource_self_cycle", obj(recursive), {"v": {"next": {}}}, {"v": {"next": 3}}
    yield (
        "root_id",
        obj(integer, **{"$id": "https://example.invalid/root"}),
        {"v": 3},
        {"v": 1},
    )
    yield "root_allof", {"allOf": [obj(integer)]}, {"v": 3}, {"v": 1}
    yield (
        "root_allof_ref",
        {"allOf": [{"$ref": "#/$defs/P"}], "$defs": {"P": obj(integer)}},
        {"v": 3},
        {"v": "3"},
    )
    yield (
        "root_typed_allof",
        {"type": "object", "allOf": [obj(integer)]},
        {"v": 3},
        {"v": 6},
    )

    yield (
        "root_id_ref",
        {
            "$id": "https://example.invalid/root",
            "$ref": "#/$defs/P",
            "$defs": {"P": obj(integer)},
        },
        {"v": 3},
        {"v": 6},
    )

    root_cycle = {
        "$id": "https://example.invalid/root",
        "type": "object",
        "properties": {"v": {"anyOf": [{"type": "null"}, {"$ref": "#"}]}},
        "required": ["v"],
        "additionalProperties": False,
    }
    yield "root_id_consuming_cycle", root_cycle, {"v": {"v": None}}, {"v": {"v": 3}}
    scoped_escape = {
        "$id": "inner",
        "type": "array",
        "$defs": {"a/b": integer},
        "items": {"$ref": "#%2F$defs%2Fa~1b"},
    }
    yield "nested_escaped", obj(scoped_escape), {"v": [3]}, {"v": [1]}
    deepest = {
        "$id": "child",
        "type": "array",
        "$defs": {"X": integer},
        "items": {"$ref": "#/$defs/X"},
    }
    middle = {
        "$id": "https://example.invalid/middle/",
        "type": "array",
        "$defs": {"X": {"type": "string"}},
        "items": deepest,
    }
    yield (
        "nested_two_resources",
        obj(middle, **{"$defs": {"X": {"type": "boolean"}}}),
        {"v": [[3]]},
        {"v": [["3"]]},
    )

    named_property = {
        "type": "object",
        "properties": {"$id": {"type": "integer"}, "x": {"$ref": "#/properties/$id"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    yield "properties_member_id_ref", named_property, {"x": 3}, {"x": "3"}
    yield (
        "defs_member_id_ref",
        obj({"$ref": "#/$defs/$id"}, **{"$defs": {"$id": integer}}),
        {"v": 3},
        {"v": 1},
    )
    active_literal = {
        "$id": "inner",
        "type": "array",
        "$defs": {"X": {"type": "string"}},
        "items": {"$ref": "#/$defs/X"},
    }
    yield (
        "literal_target_becomes_resource",
        obj(
            {"$ref": "#/$defs/holder/const/target"},
            **{
                "$defs": {
                    "X": {"type": "integer"},
                    "holder": {"const": {"target": active_literal}},
                }
            },
        ),
        {"v": ["text"]},
        {"v": [3]},
    )


class ResourceNormalization(unittest.TestCase):
    def test_preservation(self):
        from jsonschema import Draft202012Validator

        for name, schema, good, bad in fixtures():
            with self.subTest(name=name):
                original = copy.deepcopy(schema)
                normalized = normalize(schema)
                self.assertEqual(schema, original)
                self.assertEqual(normalize(normalized), normalized)
                validate(schema)
                self.assertTrue(Draft202012Validator(normalized).is_valid(good))
                self.assertFalse(Draft202012Validator(normalized).is_valid(bad))
                self.assertLess(len(json.dumps(normalized)), 5000)

    def test_real_resource_ref_is_not_retargeted(self):
        from jsonschema import Draft202012Validator

        for identifier in [
            "inner",
            "https://example.invalid/inner",
            "",
            "https://example.invalid/root",
        ]:
            inner = {
                "$id": identifier,
                "type": "array",
                "$defs": {"X": {"type": "string"}},
                "items": {"$ref": "#/$defs/X"},
            }
            schema = {
                "$id": "https://example.invalid/root",
                "type": "object",
                "$defs": {"X": {"type": "integer"}},
                "properties": {"v": inner},
            }
            normalized = normalize(schema)
            for value in [{"v": ["text"]}, {"v": [3]}]:
                with self.subTest(identifier=identifier, value=value):
                    self.assertEqual(
                        Draft202012Validator(schema).is_valid(value),
                        Draft202012Validator(normalized).is_valid(value),
                    )

    def test_child_jump_retains_resource_context(self):
        from jsonschema import Draft202012Validator

        schema = {
            "$defs": {
                "X": {"type": "integer"},
                "R": {
                    "$id": "inner",
                    "$defs": {"X": {"type": "string"}},
                    "properties": {"X": {"$ref": "#/$defs/X"}},
                },
            },
            "$ref": "#/$defs/R/properties/X",
        }
        normalized = normalize(schema)
        self.assertTrue(Draft202012Validator(normalized).is_valid("text"))
        self.assertFalse(Draft202012Validator(normalized).is_valid(3))

    def test_duplicate_active_resource_ids_fail_closed(self):
        first = {
            "$id": "shared",
            "$defs": {"X": {"type": "integer"}},
            "items": {"$ref": "#/$defs/X"},
        }
        second = {
            "$id": "shared",
            "$defs": {"X": {"type": "string"}},
            "items": {"$ref": "#/$defs/X"},
        }
        with self.assertRaises(UnsupportedJSONSchemaFeature) as caught:
            normalize({"type": "object", "properties": {"a": first, "b": second}})
        self.assertEqual(caught.exception.reason_code, "ambiguous_resource_identifier")

    def test_fragment_ids_do_not_become_pointer_resources(self):
        with self.assertRaises(UnsupportedJSONSchemaFeature):
            normalize({"type": "array", "items": {"$id": "#anchor", "type": "integer"}})

    def test_pointer_scope_controls_pure(self):
        from jsonschema import Draft202012Validator

        schema = {
            "properties": {
                "$id": {"type": "integer"},
                "x": {"$ref": "#/properties/$id"},
            }
        }
        normalized = normalize(schema)
        self.assertTrue(Draft202012Validator(normalized).is_valid({"x": 3}))
        self.assertFalse(Draft202012Validator(normalized).is_valid({"x": "3"}))
        # Once the whole map is selected as a schema, its own $id is a keyword,
        # not a property name: the non-string identifier must fail explicitly.
        for container in ["default", "const"]:
            document = {
                "$ref": "#/$defs/holder/" + container,
                "$defs": {"holder": {container: {"$id": {"type": "integer"}}}},
            }
            with self.subTest(container=container):
                with self.assertRaises(UnsupportedJSONSchemaFeature) as caught:
                    normalize(document)
                self.assertEqual(
                    caught.exception.reason_code, "invalid_resource_identifier"
                )

    def test_nonlocal_and_dynamic_fail(self):
        for node in [
            {"$ref": "remote.json"},
            {"$ref": "#anchor"},
            {"$ref": "#/missing"},
            {"$dynamicRef": "#"},
            {
                "$schema": "https://json-schema.org/draft/2019-09/schema",
                "$recursiveRef": "#",
            },
            {"$ref": "#%FF"},
        ]:
            with (
                self.subTest(node=node),
                self.assertRaises(UnsupportedJSONSchemaFeature),
            ):
                normalize(node)

    def test_pure_reference_cycles_rejected_early(self):
        for schema in [
            {"$ref": "#"},
            {
                "$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"$ref": "#/$defs/a"}},
                "$ref": "#/$defs/a",
            },
        ]:
            with self.subTest(schema=schema):
                with self.assertRaises(UnsupportedJSONSchemaFeature) as caught:
                    normalize(schema)
                self.assertEqual(
                    caught.exception.reason_code, "nonconsuming_reference_cycle"
                )

    def test_unused_assertions_do_not_reject(self):
        validate({"type": "integer", "$defs": {"unused": {"uniqueItems": True}}})
        with self.assertRaises(UnsupportedJSONSchemaFeature):
            validate({"$defs": {"used": {"uniqueItems": True}}, "$ref": "#/$defs/used"})

    def test_root_id_and_literal_data_preserved(self):
        schema = {
            "$id": "https://example.invalid/root",
            "type": "integer",
            "default": {"$id": "data", "$ref": "data"},
        }
        self.assertEqual(normalize(schema), schema)

    def test_graph_depth_bound(self):
        schema = {"type": "integer"}
        for _ in range(130):
            schema = {"type": "array", "items": schema}
        with self.assertRaisesRegex(UnsupportedJSONSchemaFeature, "schema_graph_limit"):
            normalize(schema)

    def test_shared_target_deduplicated(self):
        schema = {
            "type": "object",
            "properties": {str(i): {"$ref": "#/$defs/X"} for i in range(100)},
            "$defs": {"X": {"type": "integer"}, "unused": {"enum": ["x" * 100000]}},
        }
        result = normalize(schema)
        self.assertEqual(len(result["$defs"]), 1)
        self.assertLess(len(json.dumps(result)), 10000)


@unittest.skipUnless(
    os.environ.get("PROTOCOL3_NATIVE") == "1", "native gate runs in isolated CPU image"
)
class NativeResourceNormalization(unittest.TestCase):
    def test_pointer_scope_controls_native(self):
        import xgrammar as xgr
        from xgrammar.testing import _is_grammar_accept_string

        from sglang.srt.entrypoints.openai.protocol import Tool
        from sglang.srt.function_call.kimik3_structural_tag import (
            _tool_arguments_format,
        )

        selected = {
            "properties_member_id_ref",
            "defs_member_id_ref",
            "literal_target_becomes_resource",
        }
        for name, schema, good, bad in fixtures():
            if name not in selected:
                continue
            with self.subTest(name=name):
                grammar = xgr.Grammar.from_json_schema(
                    normalize(schema), strict_mode=False
                )
                self.assertTrue(_is_grammar_accept_string(grammar, json.dumps(good)))
                self.assertFalse(_is_grammar_accept_string(grammar, json.dumps(bad)))
                tool = Tool(
                    type="function",
                    function={
                        "name": "synthetic",
                        "parameters": schema,
                        "strict": True,
                    },
                )
                fmt = _tool_arguments_format(tool)
                grammar = xgr.Grammar.from_structural_tag(
                    json.dumps(
                        {
                            "type": "structural_tag",
                            "format": fmt.model_dump(exclude_none=True),
                        }
                    )
                )

                def wire(value):
                    key, content = next(iter(value.items()))
                    kind = (
                        "array"
                        if isinstance(content, list)
                        else "number"
                        if isinstance(content, int)
                        else "json-string"
                    )
                    return (
                        '<|open|>argument key="'
                        + key
                        + '" type="'
                        + kind
                        + '"<|sep|>'
                        + json.dumps(content)
                        + "<|close|>argument<|sep|>"
                    )

                self.assertTrue(_is_grammar_accept_string(grammar, wire(good)))
                self.assertFalse(_is_grammar_accept_string(grammar, wire(bad)))

    def test_actual_native_and_json_format(self):
        import xgrammar as xgr
        from xgrammar.testing import _is_grammar_accept_string

        from sglang.srt.constrained.kimik3_json_schema import json_response_format

        for name, schema, good, bad in fixtures():
            with self.subTest(name=name):
                result = normalize(schema)
                grammar = xgr.Grammar.from_json_schema(
                    json.dumps(result), strict_mode=False, separators=(",", ":")
                )
                self.assertTrue(
                    _is_grammar_accept_string(
                        grammar, json.dumps(good, separators=(",", ":"))
                    )
                )
                self.assertFalse(
                    _is_grammar_accept_string(
                        grammar, json.dumps(bad, separators=(",", ":"))
                    )
                )
                self.assertFalse(_is_grammar_accept_string(grammar, "{}"))
                self.assertFalse(
                    _is_grammar_accept_string(
                        grammar, json.dumps(dict(good, extra=1), separators=(",", ":"))
                    )
                )
                native = xgr.Grammar.from_structural_tag(
                    json.dumps(json_response_format(schema, False))
                )
                self.assertTrue(
                    _is_grammar_accept_string(
                        native, json.dumps(good, separators=(",", ":"))
                    )
                )
                self.assertFalse(
                    _is_grammar_accept_string(
                        native, json.dumps(bad, separators=(",", ":"))
                    )
                )

    def test_allof_unproven_siblings_and_intersections_refused(self):
        from sglang.srt.entrypoints.openai.protocol import Tool
        from sglang.srt.function_call.kimik3_structural_tag import (
            _tool_arguments_format,
        )

        for schema in [
            {"allOf": [{"type": "object"}, {"required": ["v"]}]},
            {"allOf": [{"type": "object"}], "required": ["v"]},
            {"type": "object", "allOf": [{"type": "string"}]},
            {"allOf": [{"$ref": "#"}]},
        ]:
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                _tool_arguments_format(
                    Tool(
                        type="function",
                        function={
                            "name": "synthetic",
                            "parameters": schema,
                            "strict": True,
                        },
                    )
                )

    def test_actual_tool_builder_and_native(self):
        import xgrammar as xgr
        from xgrammar.testing import _is_grammar_accept_string

        from sglang.srt.entrypoints.openai.protocol import Tool
        from sglang.srt.function_call.kimik3_structural_tag import (
            _tool_arguments_format,
        )

        for name, schema, good, bad in fixtures():
            with self.subTest(name=name):
                tool = Tool(
                    type="function",
                    function={
                        "name": "synthetic",
                        "parameters": schema,
                        "strict": True,
                    },
                )
                fmt = _tool_arguments_format(tool)
                grammar = xgr.Grammar.from_structural_tag(
                    json.dumps(
                        {
                            "type": "structural_tag",
                            "format": fmt.model_dump(exclude_none=True),
                        }
                    )
                )

                def wire(value):
                    key, content = next(iter(value.items()))
                    kind = (
                        "object"
                        if isinstance(content, dict)
                        else "array"
                        if isinstance(content, list)
                        else "null"
                        if content is None
                        else "number"
                        if isinstance(content, int)
                        else "json-string"
                    )
                    return (
                        '<|open|>argument key="'
                        + key
                        + '" type="'
                        + kind
                        + '"<|sep|>'
                        + json.dumps(content, separators=(",", ":"))
                        + "<|close|>argument<|sep|>"
                    )

                self.assertTrue(_is_grammar_accept_string(grammar, wire(good)))
                self.assertFalse(_is_grammar_accept_string(grammar, wire(bad)))


if __name__ == "__main__":
    unittest.main()
