"""Small-object arbitrary-order grammar: no required/type/uniqueness relaxation."""
import itertools
import json
import time

import pytest
import xgrammar as xg
from xgrammar.testing import _is_grammar_accept_string


def accepts(schema, text, **kw):
    grammar = xg.Grammar.from_json_schema(schema, any_whitespace=True, max_whitespace_cnt=64, **kw)
    return _is_grammar_accept_string(grammar, text)


@pytest.mark.parametrize("required", [[], ["foo"], ["foo", "barbaz"]])
@pytest.mark.parametrize("additional", [True, False, {"type": "integer"}])
def test_all_permutations(required, additional):
    schema = {"type": "object", "properties": {"foo": {"type": "boolean"}, "barbaz": {"type": "string"}},
              "required": required, "additionalProperties": additional}
    items = [("foo", True), ("barbaz", "x")]
    if additional is not False:
        items += [("zzz", 1), ("first", 2)]
    grammar = xg.Grammar.from_json_schema(schema, any_whitespace=True, max_whitespace_cnt=64)
    for length in range(len(items) + 1):
        for subset in itertools.combinations(items, length):
            expected = set(required).issubset(dict(subset))
            for ordered in itertools.permutations(subset):
                text = json.dumps(dict(ordered), ensure_ascii=False)
                assert _is_grammar_accept_string(grammar, text) == expected, text


SCHEMA = {"type": "object", "properties": {"foo": {"type": "boolean"}, "barbaz": {"type": "string"}},
          "required": ["foo", "barbaz"], "additionalProperties": True}


@pytest.mark.parametrize("text", [
    '{"barbaz":"x"}', '{"barbaz":"x","zzz":1}',
    '{"barbaz":"x","foo":"wrong"}', '{"zzz":1,"foo":true,"barbaz":2}',
    '{"foo":true,"barbaz":"x","foo":false}', '{"barbaz":"x","barbaz":"y","foo":true}',
    '{"foo":true,"barbaz":"x",}', '{"foo":true,"barbaz":"x"',
    '{"foo":true,"barbaz":"x"} prose', '{"barbaz":"x",' + ' ' * 65 + '"foo":true}',
])
def test_invalid_and_duplicate_keys(text):
    assert not accepts(SCHEMA, text)


def test_escaped_declared_name_cannot_bypass_type_or_uniqueness():
    assert not accepts(SCHEMA, '{"foo":true,"barbaz":"x","\\u0066oo":"wrong"}')
    assert not accepts(SCHEMA, '{"foo":true,"barbaz":"x","f\\u006fo":"wrong"}')
    assert accepts(SCHEMA, '{"foo":true,"barbaz":"x","other\\u0066oo":"ok"}')


def test_additional_unicode_and_escaped_names():
    schema = {"type": "object", "properties": {"中文": {"type": "integer"}, 'a"b': {"type": "boolean"}},
              "required": ["中文", 'a"b'], "additionalProperties": True}
    assert accepts(schema, json.dumps({'a"b': True, "extra\nkey": 3, "文中": 2, "中文": 1}, ensure_ascii=False))
    assert not accepts(schema, '{"中文":1,"a\\"b":true,"中文":"wrong"}')
    assert not accepts(schema, '{"中文":1,"a\\"b":true,"\\u4e2d文":"wrong"}')
    assert not accepts(schema, '{"中文":1,"a\\"b":true,"a\\u0022b":"wrong"}')


def test_nested_refs_and_values():
    schema = {"$defs": {"item": SCHEMA}, "type": "object", "properties": {
        "item": {"$ref": "#/$defs/item"}, "n": {"const": 7},
        "union": {"type": ["integer", "null"]}}, "required": ["item", "n"], "additionalProperties": False}
    assert accepts(schema, '{"n":7,"union":null,"item":{"zzz":2,"barbaz":"x","foo":true}}')
    assert not accepts(schema, '{"n":7,"item":{"zzz":2,"barbaz":"x"}}')
    assert not accepts(schema, '{"n":"7","item":{"foo":true,"barbaz":"x"}}')


def test_unicode_escaped_values():
    schema = {"type": "object", "properties": {"中文": {"type": "string"}, 'a"b': {"const": True}},
              "required": ["中文", 'a"b'], "additionalProperties": False}
    assert accepts(schema, json.dumps({'a"b': True, "中文": 'quote" slash\\ newline\n'}, ensure_ascii=False))


def test_count_constraint_keeps_existing_strict_path():
    schema = dict(SCHEMA, minProperties=3, maxProperties=3)
    assert accepts(schema, '{"foo":true,"barbaz":"x","zzz":1}')
    assert not accepts(schema, '{"foo":true,"barbaz":"x"}')
    assert not accepts(schema, '{"foo":true,"barbaz":"x","z":1,"w":2}')


def test_size_boundary_stays_strict():
    schema = {"type": "object", "properties": {f"k{i}": {"type": "integer"} for i in range(9)},
              "required": [f"k{i}" for i in range(9)], "additionalProperties": False}
    assert accepts(schema, json.dumps({f"k{i}": i for i in range(9)}))
    assert not accepts(schema, json.dumps({f"k{i}": i for i in range(8)}))


def test_compilation_budget():
    schema = {"type": "object", "properties": {f"k{i}": {"type": "integer"} for i in range(8)},
              "required": [f"k{i}" for i in range(8)], "additionalProperties": True}
    started = time.monotonic()
    grammar = xg.Grammar.from_json_schema(schema, max_whitespace_cnt=64)
    assert time.monotonic() - started < 10
    assert _is_grammar_accept_string(grammar, json.dumps({f"k{i}": i for i in reversed(range(8))}))


def test_xml_tool_cardinality_path_unchanged():
    from xgrammar.builtin_structural_tag import get_model_structural_tag

    tools = [{"type": "function", "function": {
        "name": "get_weather", "strict": True,
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city", "unit"], "additionalProperties": False}}}]

    def call(city):
        return ("<tool_call>\n<function=get_weather>\n<parameter=city>\n" + city
                + "\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n"
                  "</function>\n</tool_call>")

    grammar = xg.Grammar.from_structural_tag(get_model_structural_tag("qwen_3_coder", tools=tools, tool_choice="required", reasoning=False))
    assert _is_grammar_accept_string(grammar, call("Paris") + "\n" + call("Tokyo"))
