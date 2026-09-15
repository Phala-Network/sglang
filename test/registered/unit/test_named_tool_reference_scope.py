import copy
import unittest

from jsonschema import Draft202012Validator, ValidationError
from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.utils import get_json_schema_constraint


class NamedToolReferenceScope(unittest.TestCase):
    def test_named_tool_refs_validate_under_the_array_wrapper(self):
        for definition_key in ("$defs", "definitions"):
            schema = {"type": "object", "properties": {"location": {"$ref": f"#/{definition_key}/Location"}}, "required": ["location"], "additionalProperties": False, definition_key: {"Location": {"type": "object", "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}}, "required": ["lat", "lon"], "additionalProperties": False}}}
            original = copy.deepcopy(schema)
            selected = Tool.model_validate({"type": "function", "function": {"name": "lookup", "parameters": schema}})
            unselected = Tool.model_validate({"type": "function", "function": {"name": "unused", "parameters": {"$defs": {"Location": {"type": "string"}}}}})
            choice = ToolChoice.model_validate({"type": "function", "function": {"name": "lookup"}})
            wrapped = get_json_schema_constraint([selected, unselected], choice, parallel_tool_calls=False)
            validator = Draft202012Validator(wrapped)
            validator.validate([{"name": "lookup", "parameters": {"location": {"lat": 48.8566, "lon": 2.3522}}}])
            with self.assertRaises(ValidationError):
                validator.validate([{"name": "lookup", "parameters": {"location": "wrong-type"}}])
            with self.assertRaises(ValidationError):
                validator.validate([{"name": "lookup", "parameters": {"location": {"lat": 1, "lon": 2}}}] * 2)
            self.assertEqual(selected.function.parameters, original)


if __name__ == "__main__":
    unittest.main()
