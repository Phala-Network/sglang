"""Synthetic dialect/annotation contracts, with opt-in actual native/API gates."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

try:
    from sglang.srt.constrained import json_schema_dialect as dialects
    from sglang.srt.constrained.json_schema_validation import (
        UnsupportedJSONSchemaFeature,
        normalize_json_schema_resources as normalize,
        validate_xgrammar_json_schema as validate,
    )
except ModuleNotFoundError:
    # Load the real pure-CPU modules without importing the serving package's
    # torch/CUDA dependencies. No grammar/compiler is simulated by this loader.
    source = Path(__file__).resolve().parents[4] / "python/sglang/srt/constrained"
    for name in (
        "json_schema_dialect",
        "json_schema_intersections",
        "json_schema_validation",
    ):
        qualified = "sglang.srt.constrained." + name
        spec = importlib.util.spec_from_file_location(
            qualified, source / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        if name == "json_schema_dialect":
            dialects = module
        elif name == "json_schema_validation":
            normalize = module.normalize_json_schema_resources
            validate = module.validate_xgrammar_json_schema
            UnsupportedJSONSchemaFeature = module.UnsupportedJSONSchemaFeature


DECLARATIONS = (
    "http://json-schema.org/draft-06/schema#",
    "https://json-schema.org/draft-07/schema#",
    "https://json-schema.org/draft/2019-09/schema",
    "https://json-schema.org/draft/2020-12/schema",
)


def object_schema(**extra):
    return dict(
        type="object",
        properties={"v": {"type": "string", "minLength": 2}},
        required=["v"],
        additionalProperties=False,
        **extra
    )


class DialectNormalization(unittest.TestCase):
    def assert_reason(self, schema, reason):
        with self.assertRaises(UnsupportedJSONSchemaFeature) as caught:
            normalize(schema)
        self.assertEqual(caught.exception.reason_code, reason)
        return caught.exception

    def test_fixed_default_and_exact_local_aliases(self):
        self.assertEqual(
            dialects.resolve_json_schema_dialect().uri,
            "https://json-schema.org/draft/2020-12/schema",
        )
        for declaration in DECLARATIONS:
            with self.subTest(declaration=declaration):
                schema = object_schema(**{"$schema": declaration})
                validate(schema)
                self.assertEqual(normalize(schema), schema)
        for declaration in (None, 12, {}, False):
            self.assert_reason({"$schema": declaration}, "invalid_schema_dialect")

    def test_unavailable_dialect_and_no_retrieval(self):
        with patch(
            "urllib.request.urlopen", side_effect=AssertionError("network retrieval")
        ), patch(
            "socket.create_connection", side_effect=AssertionError("network retrieval")
        ):
            for uri in (
                "https://example.invalid/private-meta",
                "file:///private-meta.json",
                "https://json-schema.org/draft/2020-12/schema#anchor",
                "",
            ):
                with self.subTest(uri=uri):
                    error = self.assert_reason(
                        {"$schema": uri}, "unsupported_schema_dialect"
                    )
                    self.assertNotIn("private", str(error))

    def test_unknown_annotations_all_prefixes_and_arbitrary_data(self):
        poison = {
            "not": {},
            "$id": 9,
            "$ref": "https://invalid.invalid/schema",
            "$schema": "https://invalid.invalid/dialect",
            "properties": {"x": None},
        }
        for declaration in (None, *DECLARATIONS):
            schema = object_schema(
                **{
                    name: copy.deepcopy(poison)
                    for name in ("x-note", "ordinaryNote", "$unregistered")
                }
            )
            if declaration:
                schema["$schema"] = declaration
            original = copy.deepcopy(schema)
            result = normalize(schema)
            for key in ("x-note", "ordinaryNote", "$unregistered"):
                self.assertNotIn(key, result)
            validate(schema)
            self.assertEqual(schema, original)
            self.assertEqual(normalize(result), result)

    def test_property_names_and_instance_data_are_not_keyword_maps(self):
        names = ("not", "$schema", "$vocabulary", "x-note", "ordinaryNote", "$id")
        literal = {name: {"$schema": "unavailable", "not": {}} for name in names}
        schema = {
            "type": "object",
            "properties": {name: {"const": literal} for name in names},
            "default": literal,
            "examples": [literal],
            "enum": [literal],
        }
        self.assertEqual(normalize(schema), schema)

    def test_content_and_identifier_annotations_do_not_assert(self):
        from jsonschema import Draft202012Validator

        schema = object_schema(
            **{
                "$anchor": "unused",
                "$dynamicAnchor": "dynamicUnused",
                "$recursiveAnchor": "recursiveUnused",
                "contentEncoding": "base64",
                "contentMediaType": "application/json",
                "contentSchema": {
                    "not": {},
                    "$schema": "https://example.invalid/inactive-meta",
                },
                "$vocabulary": {"https://example.invalid/ordinary-schema": True},
            }
        )
        Draft202012Validator.check_schema(schema)
        self.assertEqual(normalize(schema), object_schema())
        validate(schema)
        self.assert_reason(
            dict(schema, **{"$ref": "#unused"}), "unresolved_or_nonlocal_reference"
        )

    def test_known_unsupported_assertions_remain_active(self):
        for declaration in DECLARATIONS:
            for constraint in (
                {"not": {"required": ["v"]}},
                {"multipleOf": 2},
                {"uniqueItems": True},
            ):
                schema = dict(constraint, **{"$schema": declaration})
                result = normalize(schema)
                self.assertTrue(set(constraint).issubset(result))
                with self.assertRaises(UnsupportedJSONSchemaFeature):
                    validate(schema)

    def test_deprecated_names_follow_the_selected_dialect(self):
        from jsonschema import (
            Draft6Validator,
            Draft7Validator,
            Draft201909Validator,
            Draft202012Validator,
        )

        for declaration, validator in zip(
            DECLARATIONS,
            (
                Draft6Validator,
                Draft7Validator,
                Draft201909Validator,
                Draft202012Validator,
            ),
        ):
            schema = {
                "$schema": declaration,
                "type": "object",
                "dependencies": {"v": ["w"]},
            }
            legacy = declaration in DECLARATIONS[:2]
            self.assertEqual(validator(schema).is_valid({"v": 1}), not legacy)
            result = normalize(schema)
            self.assertEqual("dependencies" in result, legacy)
            if legacy:
                with self.assertRaises(UnsupportedJSONSchemaFeature):
                    validate(schema)
            else:
                validate(schema)
                self.assertTrue(validator(result).is_valid({"v": 1}))
        self.assert_reason(
            {"$schema": DECLARATIONS[2], "$recursiveRef": "#"},
            "unsupported_reference_kind",
        )
        validate({"$schema": DECLARATIONS[3], "type": "integer", "$recursiveRef": "#"})

    def test_unknown_in_selected_draft_is_not_a_new_draft_assertion(self):
        for declaration in DECLARATIONS[:2]:
            schema = object_schema(
                **{
                    "$schema": declaration,
                    "dependentRequired": {"v": ["w"]},
                    "prefixItems": [{"not": {}}],
                }
            )
            result = normalize(schema)
            self.assertNotIn("dependentRequired", result)
            self.assertNotIn("prefixItems", result)
            validate(schema)
        schema = object_schema(
            **{
                "$schema": DECLARATIONS[0],
                "if": {"not": {}},
                "then": False,
                "else": False,
            }
        )
        validate(schema)  # if/then/else are annotations in draft 6.

    def test_associated_meta_schema_required_vocabulary_gate(self):
        uri = dialects.DEFAULT_JSON_SCHEMA_DIALECT
        original = dialects._LOCAL_META_SCHEMAS[uri]
        vocabularies = dict(original["$vocabulary"])
        unknown = "https://example.invalid/private-required-vocabulary"
        for required in (False, True):
            metadata = dict(
                original, **{"$vocabulary": dict(vocabularies, **{unknown: required})}
            )
            registry = dict(dialects._LOCAL_META_SCHEMAS, **{uri: metadata})
            with patch.object(dialects, "_LOCAL_META_SCHEMAS", registry):
                if required:
                    self.assert_reason(object_schema(), "unknown_required_vocabulary")
                    self.assert_reason(True, "unknown_required_vocabulary")
                else:
                    validate(object_schema())
        # No implicit inheritance of declarations through meta-schema composition.
        registry = dict(
            dialects._LOCAL_META_SCHEMAS, **{uri: {"allOf": [{"$ref": uri}]}}
        )
        with patch.object(dialects, "_LOCAL_META_SCHEMAS", registry):
            self.assert_reason(object_schema(), "missing_vocabulary_declaration")
        for required in ("yes", 1, None):
            with self.assertRaises(dialects.JSONSchemaDialectError):
                dialects.validate_required_vocabularies(
                    {"$vocabulary": {unknown: required}}
                )

    def test_ref_into_removed_unknown_annotation_stays_assertive(self):
        schema = {
            "$ref": "#/ordinaryNote/value",
            "ordinaryNote": {
                "$schema": "unavailable-data",
                "$id": 1,
                "value": {"type": "string", "minLength": 2, "x-note": {"not": {}}},
            },
        }
        result = normalize(schema)
        self.assertNotIn("ordinaryNote", result)
        target = next(iter(result["$defs"].values()))
        self.assertEqual(target, {"type": "string", "minLength": 2})
        self.assertEqual(normalize(result), result)
        validate(schema)
        with self.assertRaises(UnsupportedJSONSchemaFeature):
            validate({"$ref": "#/note", "note": {"not": {"type": "string"}}})
        self.assert_reason(
            {"$ref": "#/note", "note": {"$schema": "unavailable"}},
            "unsupported_schema_dialect",
        )

    def test_legacy_ref_siblings_are_ignored_but_modern_are_preserved(self):
        from jsonschema import (
            Draft6Validator,
            Draft7Validator,
            Draft201909Validator,
            Draft202012Validator,
        )

        for declaration, validator in zip(
            DECLARATIONS,
            (
                Draft6Validator,
                Draft7Validator,
                Draft201909Validator,
                Draft202012Validator,
            ),
        ):
            schema = {
                "$schema": declaration,
                "$ref": "#/definitions/S",
                "minLength": 3,
                "definitions": {"S": {"type": "string"}},
            }
            result = normalize(schema)
            for value in ("x", "abcd", 4):
                self.assertEqual(
                    validator(schema).is_valid(value), validator(result).is_valid(value)
                )
            self.assertEqual(normalize(result), result)
        legacy = {
            "$schema": DECLARATIONS[1],
            "$ref": "#/definitions/S",
            "not": {},
            "definitions": {"S": {"type": "string"}},
        }
        validate(legacy)

    def test_resource_dialect_survives_reference_lifting_and_repeat_normalization(self):
        from jsonschema import Draft202012Validator

        legacy = {
            "$id": "legacy",
            "$schema": DECLARATIONS[1],
            "definitions": {"S": {"type": "string"}},
            "properties": {"v": {"$ref": "#/definitions/S", "minLength": 3}},
        }
        schema = {
            "$schema": DECLARATIONS[3],
            "$ref": "#/$defs/R/properties/v",
            "$defs": {"R": legacy},
        }
        result = normalize(schema)
        target = result["$defs"]["__sglang_ref_1"]
        self.assertEqual(target["$schema"], "http://json-schema.org/draft-07/schema")
        self.assertNotIn("minLength", target)
        self.assertEqual(normalize(result), result)
        validator = Draft202012Validator(result)
        self.assertTrue(validator.is_valid("x"))
        self.assertFalse(validator.is_valid(1))
        # The same legacy-looking annotation is active under a modern resource.
        legacy["$schema"] = DECLARATIONS[2]
        modern = normalize(schema)
        self.assertEqual(modern["$defs"]["__sglang_ref_1"]["$schema"], DECLARATIONS[2])
        self.assertFalse(Draft202012Validator(modern).is_valid("x"))
        self.assertEqual(normalize(modern), modern)

    def test_legacy_array_bounds_are_explicit_without_changing_scalar_items(self):
        for declaration in DECLARATIONS[:3]:
            schema = {
                "$schema": declaration,
                "type": "array",
                "items": {"type": "integer"},
                "additionalItems": {"not": {}, "$schema": "unavailable"},
            }
            validate(schema)  # additionalItems does not apply to schema-valued items.
            self.assertNotIn("additionalItems", normalize(schema))
            self.assert_reason(
                dict(schema, items=[{"type": "integer"}]), "unsupported_tuple_items"
            )
        self.assert_reason(
            {"type": "array", "items": [{"type": "integer"}]}, "invalid_schema"
        )


@unittest.skipUnless(
    os.environ.get("DIALECT_NATIVE") == "1", "actual native CPU gate is opt-in"
)
class NativeDialectIntegration(unittest.TestCase):
    def test_actual_k3_helper_and_native_annotation_controls(self):
        import xgrammar as xgr
        from xgrammar.testing import _is_grammar_accept_string
        from sglang.srt.entrypoints.openai.protocol import Tool
        from sglang.srt.function_call.kimik3_structural_tag import (
            _tool_arguments_format,
        )

        for declaration in (None, *DECLARATIONS):
            schema = object_schema(
                **{
                    "ordinaryNote": {"not": {}, "$schema": "unavailable"},
                    "$vocabulary": {"ordinary": True},
                    "contentSchema": {"not": {}},
                }
            )
            if declaration:
                schema["$schema"] = declaration
            original = copy.deepcopy(schema)
            fmt = _tool_arguments_format(
                Tool(
                    type="function",
                    function={
                        "name": "synthetic",
                        "parameters": schema,
                        "strict": True,
                    },
                )
            )
            native = xgr.Grammar.from_structural_tag(
                json.dumps(
                    {
                        "type": "structural_tag",
                        "format": fmt.model_dump(exclude_none=True),
                    }
                )
            )

            def wire(value):
                return (
                    '<|open|>argument key="v" type="json-string"<|sep|>'
                    + json.dumps(value)
                    + "<|close|>argument<|sep|>"
                )

            self.assertTrue(_is_grammar_accept_string(native, wire("ok")))
            self.assertFalse(_is_grammar_accept_string(native, wire("x")))
            self.assertFalse(_is_grammar_accept_string(native, ""))
            self.assertEqual(schema, original)

    def test_actual_native_mixed_resource_dialect_assertions(self):
        import xgrammar as xgr
        from xgrammar.testing import _is_grammar_accept_string

        for declaration in DECLARATIONS:
            schema = object_schema(**{"$schema": declaration})
            native = xgr.Grammar.from_json_schema(normalize(schema), strict_mode=False)
            self.assertTrue(_is_grammar_accept_string(native, '{"v":"ok"}'))
            self.assertFalse(_is_grammar_accept_string(native, '{"v":"x"}'))
            self.assertFalse(_is_grammar_accept_string(native, '{"v":2}'))
        for declaration in DECLARATIONS:
            resource = {
                "$id": "inner",
                "$schema": declaration,
                "definitions": {"S": {"type": "string"}},
                "properties": {"v": {"$ref": "#/definitions/S", "minLength": 3}},
            }
            schema = {"$ref": "#/$defs/R/properties/v", "$defs": {"R": resource}}
            native = xgr.Grammar.from_json_schema(normalize(schema), strict_mode=False)
            self.assertTrue(_is_grammar_accept_string(native, '"abcd"'))
            self.assertFalse(_is_grammar_accept_string(native, "4"))
            self.assertEqual(
                _is_grammar_accept_string(native, '"x"'),
                declaration in DECLARATIONS[:2],
            )

    def test_actual_k3_helper_checks_associated_vocabulary(self):
        from sglang.srt.entrypoints.openai.protocol import Tool
        from sglang.srt.function_call.kimik3_structural_tag import (
            _tool_arguments_format,
        )

        tool = Tool(
            type="function",
            function={
                "name": "synthetic",
                "parameters": object_schema(),
                "strict": True,
            },
        )
        uri = dialects.DEFAULT_JSON_SCHEMA_DIALECT
        for required in (False, True):
            vocabularies = dict(dialects._LOCAL_META_SCHEMAS[uri]["$vocabulary"])
            vocabularies["https://example.invalid/unknown"] = required
            registry = dict(
                dialects._LOCAL_META_SCHEMAS, **{uri: {"$vocabulary": vocabularies}}
            )
            with patch.object(dialects, "_LOCAL_META_SCHEMAS", registry):
                if required:
                    with self.assertRaises(UnsupportedJSONSchemaFeature) as caught:
                        _tool_arguments_format(tool)
                    self.assertEqual(
                        caught.exception.reason_code, "unknown_required_vocabulary"
                    )
                else:
                    _tool_arguments_format(tool)


@unittest.skipUnless(
    os.environ.get("DIALECT_SERVING") == "1", "actual serving CPU preflight is opt-in"
)
class ServingDialectIntegration(unittest.TestCase):
    def test_actual_serving_preflight_dialect_status_and_annotation_control(self):
        from types import SimpleNamespace
        from fastapi import HTTPException
        from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase

        serving = SimpleNamespace(chat_encoding_spec="kimi_k3")

        def preflight(schema):
            OpenAIServingBase._validate_active_json_constraint(
                serving, {"json_schema": json.dumps(schema)}
            )

        for declaration in DECLARATIONS:
            preflight(
                object_schema(**{"$schema": declaration, "ordinaryNote": {"not": {}}})
            )
        for schema, reason in [
            (
                object_schema(**{"$schema": "https://example.invalid/private-dialect"}),
                "unsupported_schema_dialect",
            ),
            (
                object_schema(**{"not": {"required": ["v"]}}),
                "unsupported_schema_constraint",
            ),
        ]:
            with self.assertRaises(HTTPException) as caught:
                preflight(schema)
            self.assertEqual(caught.exception.status_code, 400)
            self.assertIn(reason, caught.exception.detail)
            self.assertNotIn("private", caught.exception.detail)
        uri = dialects.DEFAULT_JSON_SCHEMA_DIALECT
        vocabularies = dict(dialects._LOCAL_META_SCHEMAS[uri]["$vocabulary"])
        vocabularies["https://example.invalid/unknown"] = True
        registry = dict(
            dialects._LOCAL_META_SCHEMAS, **{uri: {"$vocabulary": vocabularies}}
        )
        with patch.object(dialects, "_LOCAL_META_SCHEMAS", registry):
            with self.assertRaises(HTTPException) as caught:
                preflight(object_schema())
            self.assertEqual(caught.exception.status_code, 400)
            self.assertIn("unknown_required_vocabulary", caught.exception.detail)


if __name__ == "__main__":
    unittest.main()
