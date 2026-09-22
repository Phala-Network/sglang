"""CPU source-method regressions for named-tool wrapper reference roots.

Execute the actual pure schema-builder functions without importing the serving
stack or native dependencies. This does not qualify native grammar compilation.
"""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/function_call/utils.py"


class ToolChoice(SimpleNamespace):
    pass


def load_schema_builder():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {
        "_get_tool_schema",
        "_get_tool_schema_defs",
        "get_json_schema_constraint",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise AssertionError("The real schema builder functions were not found")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    namespace = {"ToolChoice": ToolChoice}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["get_json_schema_constraint"]


BUILD = load_schema_builder()


def tool(name, parameters):
    return SimpleNamespace(function=SimpleNamespace(name=name, parameters=parameters))


def named(name):
    return ToolChoice(function=SimpleNamespace(name=name))


class NamedToolReferenceScopeTests(unittest.TestCase):
    def test_named_local_reference_root_survives_array_wrapper(self):
        for key in ("$defs", "definitions"):
            with self.subTest(key=key):
                parameters = {
                    "type": "object",
                    key: {"city": {"type": "string", "enum": ["Paris"]}},
                    "properties": {"city": {"$ref": f"#/{key}/city"}},
                    "required": ["city"],
                }
                schema = BUILD([tool("weather", parameters)], named("weather"), False)
                reference = schema["items"]["properties"]["parameters"]["properties"][
                    "city"
                ]["$ref"]
                resolved = schema
                for part in reference.removeprefix("#/").split("/"):
                    resolved = resolved[part]
                self.assertEqual(resolved, {"type": "string", "enum": ["Paris"]})
                self.assertEqual(schema["maxItems"], 1)

    def test_named_only_retains_selected_tool_definitions(self):
        selected = {"$defs": {"value": {"type": "integer"}}}
        other = {"$defs": {"value": {"type": "string"}}}
        schema = BUILD(
            [tool("selected", selected), tool("other", other)], named("selected")
        )
        self.assertEqual(schema["$defs"], selected["$defs"])
        self.assertNotIn("maxItems", schema)

    def test_plain_named_schema_and_input_remain_unchanged(self):
        parameters = {"type": "object", "properties": {"x": {"type": "string"}}}
        before = copy.deepcopy(parameters)
        schema = BUILD([tool("plain", parameters)], named("plain"), False)
        self.assertNotIn("$defs", schema)
        self.assertNotIn("definitions", schema)
        self.assertEqual(parameters, before)
        self.assertEqual(schema["items"]["properties"]["name"]["enum"], ["plain"])

    def test_required_path_still_retains_defs(self):
        parameters = {"$defs": {"value": {"type": "integer"}}}
        schema = BUILD([tool("probe", parameters)], "required", False)
        self.assertEqual(schema["$defs"], parameters["$defs"])
        self.assertEqual(schema["maxItems"], 1)

    def test_unknown_named_tool_remains_none(self):
        self.assertIsNone(BUILD([tool("probe", {})], named("absent")))


if __name__ == "__main__":
    unittest.main()
