"""Exercise the installed native compiler, not just structural-tag serialization."""

import copy
import json

import pytest
import xgrammar as xgr
from xgrammar.structural_tag import JSONSchemaFormat, StructuralTag
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.utils import normalize_json_schema_types


def tool(schema):
    # Match OpenAIServingChat's real validation order. Tool.model_validate by
    # itself deliberately does not perform the request-level null normalization.
    schema = copy.deepcopy(schema)
    normalize_json_schema_types(schema)
    return Tool.model_validate(
        {"type": "function", "function": {"name": "probe", "strict": True, "parameters": schema}}
    )


def grammar(schema, choice="named", parallel=False):
    item = tool(copy.deepcopy(schema))
    selected = (
        ToolChoice.model_validate({"type": "function", "function": {"name": "probe"}})
        if choice == "named"
        else choice
    )
    tag = Qwen3CoderDetector().get_structural_tag(
        [item], selected, thinking_mode=False, parallel_tool_calls=parallel
    )
    return xgr.Grammar.from_structural_tag(tag)


def call(body="", name="probe"):
    return f"<tool_call>\n<function={name}>\n{body}\n</function>\n</tool_call>"


def parameter(name="city", value="Paris"):
    return f"<parameter={name}>{value}</parameter>"


EMPTY_SCHEMAS = [
    {"type": "object", "additionalProperties": False},
    {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    {"type": "object", "properties": None, "required": None, "additionalProperties": False},
    {"type": "object", "properties": {}, "required": []},
    {"type": "object", "additionalProperties": True, "maxProperties": 0},
]


@pytest.mark.parametrize("schema", EMPTY_SCHEMAS)
@pytest.mark.parametrize("choice", ["named", "required"])
def test_empty_tool_has_no_whitespace_only_parameter_zone(schema, choice):
    compiled = grammar(schema, choice)
    assert _is_grammar_accept_string(compiled, call())
    for whitespace in [" ", "\n", "\t", "\t" * 64, " \n\t"]:
        assert not _is_grammar_accept_string(compiled, call(whitespace))
    assert not _is_grammar_accept_string(compiled, call(parameter()))


@pytest.mark.parametrize("properties", ["omitted", None, {}])
@pytest.mark.parametrize("additional", ["omitted", True, {"type": "string"}])
def test_undeclared_required_property_is_not_discarded(properties, additional):
    schema = {"type": "object", "required": ["city"]}
    if properties != "omitted":
        schema["properties"] = properties
    if additional != "omitted":
        schema["additionalProperties"] = additional
    compiled = grammar(schema)
    assert _is_grammar_accept_string(compiled, call(parameter()))
    assert not _is_grammar_accept_string(compiled, call())
    assert not _is_grammar_accept_string(compiled, call(parameter("country", "France")))


@pytest.mark.parametrize("schema", [
    {"type": "object", "required": ["city"], "additionalProperties": False},
    {"type": "object", "properties": {}, "required": ["city"], "additionalProperties": False},
    {"type": "object", "required": ["city"], "unevaluatedProperties": False},
    {"type": "object", "required": ["city"], "maxProperties": 0},
])
def test_unsatisfiable_required_property_rejected_before_generation(schema):
    with pytest.raises((RuntimeError, ValueError)):
        grammar(schema)


def test_typed_additional_required_property_remains_typed():
    schema = {"type": "object", "required": ["count"], "additionalProperties": {"type": "integer"}}
    compiled = grammar(schema)
    assert _is_grammar_accept_string(compiled, call(parameter("count", "7")))
    assert not _is_grammar_accept_string(compiled, call(parameter("count", "wrong")))
    assert not _is_grammar_accept_string(compiled, call())


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("keyword", ["additionalProperties", "unevaluatedProperties"])
def test_typed_additional_parameter_parser(streaming, keyword):
    schema = {"type": "object", "required": ["count"], keyword: {"type": "integer"}}
    detector = Qwen3CoderDetector()
    tools = [tool(schema)]
    text = call(parameter("count", "7"))
    if streaming:
        chunks = [detector.parse_streaming_increment(text[i:i + 3], tools) for i in range(0, len(text), 3)]
        arguments = "".join(item.parameters or "" for chunk in chunks for item in chunk.calls)
    else:
        arguments = detector.detect_and_parse(text, tools).calls[0].parameters
    assert json.loads(arguments) == {"count": 7}


def test_declared_and_undeclared_required_fields_both_enforced():
    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city", "country"]}
    compiled = grammar(schema)
    assert _is_grammar_accept_string(compiled, call(parameter() + parameter("country", "France")))
    assert not _is_grammar_accept_string(compiled, call(parameter()))


def test_optional_nonempty_tool_keeps_its_parameter_branch():
    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": [], "additionalProperties": False}
    compiled = grammar(schema)
    assert _is_grammar_accept_string(compiled, call())
    assert _is_grammar_accept_string(compiled, call(parameter()))
    assert not _is_grammar_accept_string(compiled, call(parameter("wrong")))


def test_required_parallel_empty_calls_keep_template_separator():
    compiled = grammar(EMPTY_SCHEMAS[0], "required", parallel=True)
    assert _is_grammar_accept_string(compiled, call() + "\n" + call())
    assert not _is_grammar_accept_string(grammar(EMPTY_SCHEMAS[0], "required"), call() + "\n" + call())


def test_nested_empty_object_keeps_json_braces_and_ref():
    schema = {"type": "object", "properties": {"config": {"$ref": "#/$defs/Empty"}}, "required": ["config"],
              "additionalProperties": False, "$defs": {"Empty": EMPTY_SCHEMAS[0]}}
    compiled = grammar(schema)
    assert _is_grammar_accept_string(compiled, call(parameter("config", "{}")))
    assert _is_grammar_accept_string(compiled, call(parameter("config", "{ \n\t}")))
    assert not _is_grammar_accept_string(compiled, call(parameter("config", "")))


def test_plain_json_empty_object_is_unchanged():
    compiled = xgr.Grammar.from_structural_tag(StructuralTag(format=JSONSchemaFormat(json_schema=EMPTY_SCHEMAS[0])))
    assert _is_grammar_accept_string(compiled, "{}")
    assert _is_grammar_accept_string(compiled, "{ \n\t}")
    assert not _is_grammar_accept_string(compiled, "")


def test_open_object_not_collapsed_to_no_arguments():
    compiled = grammar({"type": "object", "additionalProperties": True})
    assert _is_grammar_accept_string(compiled, call())
    assert _is_grammar_accept_string(compiled, call(parameter()))


@pytest.mark.parametrize("extra", [
    {"patternProperties": {"^city$": {"type": "string"}}},
    {"propertyNames": {"enum": ["city"]}},
])
def test_undeclared_required_complex_key_constraints_fail_closed(extra):
    with pytest.raises((RuntimeError, ValueError), match="Undeclared required"):
        grammar({"type": "object", "required": ["city"], **extra})


def test_typed_additional_ref_preserved_in_parser():
    schema = {"type": "object", "required": ["count"], "additionalProperties": {"$ref": "#/$defs/Count"},
              "$defs": {"Count": {"type": "integer", "minimum": 1}}}
    compiled = grammar(schema)
    assert _is_grammar_accept_string(compiled, call(parameter("count", "7")))
    assert not _is_grammar_accept_string(compiled, call(parameter("count", "0")))
    parsed = Qwen3CoderDetector().detect_and_parse(call(parameter("count", "7")), [tool(schema)])
    assert json.loads(parsed.calls[0].parameters) == {"count": 7}
