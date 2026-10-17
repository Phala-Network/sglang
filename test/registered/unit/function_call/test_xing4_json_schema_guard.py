"""CPU-only checks of the installed schema guard."""

import unittest

from sglang.srt.constrained import json_schema_validation as _GUARD
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1, "base-a-test-cpu")


def tagged(tag, required=True):
    schema = {"type": "object", "properties": {"kind": {"const": tag}}}
    if required:
        schema["required"] = ["kind"]
    return schema


class TestXing4JSONSchemaGuard(unittest.TestCase):
    def accepts(self, schema):
        _GUARD.validate_xgrammar_json_schema(schema)

    def rejects(self, schema, pointer):
        with self.assertRaises(_GUARD.UnsupportedJSONSchemaFeature) as caught:
            self.accepts(schema)
        self.assertIn(pointer, str(caught.exception))

    def test_integer_number_overlap(self):
        self.rejects({"oneOf": [{"type": "integer"}, {"type": "number"}]}, "#/oneOf")

    def test_other_unproved_overlaps(self):
        for branches in (
            [{"type": "string"}, {"type": "string", "minLength": 1}],
            [tagged("a", False), tagged("b", False)],
            [tagged("a"), tagged("a")],
            [True, {"type": "string"}],
            [{"$ref": "#/$defs/a"}, {"$ref": "#/$defs/b"}],
            [{"type": ["string", "number"]}, {"type": "integer"}],
            [{"const": 1}, {"const": 1.0}],
            [{"enum": ["a", "b"]}, {"enum": ["b", "c"]}],
        ):
            with self.subTest(branches=branches):
                self.rejects({"oneOf": branches}, "#/oneOf")

    def test_provably_disjoint_unions(self):
        for branches in (
            [tagged("a"), tagged("b")],
            [{"type": "integer"}, {"type": "string"}, {"type": "null"}],
            [{"type": ["string", "null"]}, {"type": "number"}],
            [{"const": True}, {"const": 1}],
            [{"enum": ["a", "b"]}, {"const": "c"}],
            [False, {"type": "string"}],
            [{"type": "string"}],
        ):
            with self.subTest(branches=branches):
                self.accepts({"oneOf": branches})
        left, right = tagged("a"), tagged("b")
        left["properties"]["kind"] = {"enum": ["a", "c"]}
        self.accepts({"oneOf": [left, right]})

    def test_discriminator_must_exclude_nonobjects(self):
        left, right = tagged("a"), tagged("b")
        del left["type"]
        self.rejects({"oneOf": [left, right]}, "#/oneOf")

    def test_dependencies_all_forms_fail_closed(self):
        for dependencies in ({"a": ["b"]}, {"a": {"required": ["b"]}}, {"a": True}):
            with self.subTest(dependencies=dependencies):
                self.rejects({"dependencies": dependencies}, "dependencies")

    def test_nested_and_escaped_definition_paths(self):
        overlap = {"oneOf": [{"type": "integer"}, {"type": "number"}]}
        for schema, pointer in (
            ({"properties": {"payload": overlap}}, "#/properties/payload/oneOf"),
            ({"$defs": {"a/b~c": overlap}}, "#/$defs/a~1b~0c/oneOf"),
            ({"definitions": {"item": {"dependencies": {"a": ["b"]}}}}, "#/definitions/item"),
            ({"items": [{"dependencies": {"a": ["b"]}}]}, "#/items/0"),
            ({"anyOf": [True, overlap]}, "#/anyOf/1/oneOf"),
            ({"oneOf": [tagged("a"), dict(tagged("b"), properties={"kind": {"const": "b"}, "value": overlap})]}, "#/oneOf/1/properties/value/oneOf"),
        ):
            with self.subTest(pointer=pointer):
                self.rejects(schema, pointer)

    def test_instance_literals_are_not_schemas(self):
        literal = {"dependencies": {"a": ["b"]}, "oneOf": [{}, {}], "multipleOf": 2}
        for keyword, value in (("const", literal), ("enum", [literal]), ("default", literal), ("examples", [literal])):
            with self.subTest(keyword=keyword):
                self.accepts({keyword: value})
        self.accepts({"properties": {"dependencies": {"type": "string"}, "oneOf": {"type": "number"}}})

    def test_existing_guards_and_anyof_unchanged(self):
        self.rejects({"multipleOf": 2}, "multipleOf")
        self.rejects({"type": "string", "pattern": "a", "minLength": 1}, "cannot be enforced together")
        self.accepts({"anyOf": [{"type": "integer"}, {"type": "number"}]})
        self.accepts(True)
        self.accepts(False)


if __name__ == "__main__":
    unittest.main()
