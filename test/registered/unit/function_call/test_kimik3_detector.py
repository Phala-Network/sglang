import json

import pytest
import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai.protocol import Function, Tool, ToolChoice
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)

ENCLOSING_MARKERS = (CALL_CLOSE, TOOLS_CLOSE, MESSAGE_CLOSE)


def _argument(key, kind, value):
    raw = (
        value
        if kind == "string"
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    )
    return f'<|open|>argument key="{key}" type="{kind}"<|sep|>{raw}{ARGUMENT_CLOSE}'


def _call(name="fixture", *arguments, index=1):
    return (
        f'<|open|>call tool="{name}" index="{index}"<|sep|>'
        + "".join(arguments)
        + CALL_CLOSE
    )


def _tool(schema=None, strict=False, name="fixture"):
    return Tool(
        type="function",
        function=Function(
            name=name,
            strict=strict,
            parameters=schema
            or {
                "type": "object",
                "properties": {"payload": {"type": ["string", "object", "array"]}},
                "required": ["payload"],
                "additionalProperties": False,
            },
        ),
    )


def _parser(tools=None, choice="required"):
    return FunctionCallParser(tools or [_tool()], "kimi_k3", tool_choice=choice)


def _stream(text, chunks, tools=None, choice="required"):
    parser = _parser(tools, choice)
    normal, calls = [], []
    for chunk in chunks:
        n, c = parser.parse_stream_chunk(chunk)
        normal.append(n)
        calls.extend(c)
    n, c = parser.parse_stream_end()
    normal.append(n)
    calls.extend(c)
    return "".join(normal), [c.model_dump() for c in calls]


@pytest.mark.parametrize("marker", ENCLOSING_MARKERS)
@pytest.mark.parametrize(
    "choice", ["auto", "required", ToolChoice(function={"name": "fixture"})]
)
def test_marker_string_stream_nonstream_parity_at_every_split(marker, choice):
    value = f'雪 " \\ prefix {marker} suffix'
    text = (
        RESPONSE_OPEN
        + "Checking."
        + RESPONSE_CLOSE
        + TOOLS_OPEN
        + _call("fixture", _argument("payload", "string", value))
        + TOOLS_CLOSE
        + MESSAGE_CLOSE
    )
    normal, calls = _parser(choice=choice).parse_non_stream(text)
    assert normal == "Checking."
    assert len(calls) == 1
    assert json.loads(calls[0].parameters) == {"payload": value}
    expected = (normal, [c.model_dump() for c in calls])
    for split in range(len(text) + 1):
        assert (
            _stream(text, (text[:split], text[split:]), choice=choice) == expected
        ), split
    assert _stream(text, text, choice=choice) == expected


@pytest.mark.parametrize("constraint", ["const", "enum"])
@pytest.mark.parametrize("marker", ENCLOSING_MARKERS)
def test_native_constrained_string_preserves_valid_marker_value(marker, constraint):
    value = "literal " + marker + " value"
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "string",
                constraint: value if constraint == "const" else [value],
            },
        },
        "required": ["payload"],
        "additionalProperties": False,
    }
    tools = [_tool(schema, strict=True)]
    parser = _parser(tools)
    tag = parser.get_structure_constraint("required", parallel_tool_calls=False)[1]
    text = (
        TOOLS_OPEN
        + _call("fixture", _argument("payload", "json-string", value))
        + TOOLS_CLOSE
    )
    assert _is_grammar_accept_string(xgr.Grammar.from_structural_tag(tag), text)
    _, calls = parser.parse_non_stream(text)
    assert len(calls) == 1
    assert json.loads(calls[0].parameters) == {"payload": value}
    assert _stream(text, text, tools) == ("", [c.model_dump() for c in calls])


@pytest.mark.parametrize("marker", (ARGUMENT_CLOSE,) + ENCLOSING_MARKERS)
@pytest.mark.parametrize("kind", ["object", "array"])
def test_nested_json_marker_values_preserved(marker, kind):
    nested = {"雪": [None, True, 3, 'quote " \\ ' + marker], "other": {"value": marker}}
    payload = nested if kind == "object" else [nested, marker]
    text = (
        TOOLS_OPEN + _call("fixture", _argument("payload", kind, payload)) + TOOLS_CLOSE
    )
    normal, calls = _parser().parse_non_stream(text)
    assert normal == ""
    assert len(calls) == 1
    assert json.loads(calls[0].parameters) == {"payload": payload}
    assert _stream(text, text) == ("", [c.model_dump() for c in calls])


def test_parallel_calls_stable_and_argument_text_not_reparsed_as_call():
    # ARGUMENT_CLOSE is the delimiter, so the literal fake call has no arguments.
    value = TOOLS_CLOSE + _call("phantom") + CALL_CLOSE + MESSAGE_CLOSE
    text = (
        TOOLS_OPEN
        + _call("fixture", _argument("payload", "string", value), index=9)
        + _call("followup", _argument("payload", "string", "second"), index=9)
        + TOOLS_CLOSE
    )
    tools = [_tool(), _tool(name="followup")]
    _, expected = _parser(tools).parse_non_stream(text)
    assert [call.tool_index for call in expected] == [0, 1]
    assert [call.name for call in expected] == ["fixture", "followup"]
    assert json.loads(expected[0].parameters)["payload"] == value
    assert _stream(text, text, tools) == ("", [call.model_dump() for call in expected])


@pytest.mark.parametrize("marker", ENCLOSING_MARKERS)
def test_truncated_argument_never_emits_partial_call(marker):
    text = TOOLS_OPEN + _call(
        "fixture", _argument("payload", "string", "literal " + marker + " value")
    )
    for end in range(len(TOOLS_OPEN), len(text)):
        truncated = text[:end]
        assert _parser().parse_non_stream(truncated)[1] == [], end
        assert _stream(truncated, truncated) == ("", []), end


@pytest.mark.parametrize(
    "body",
    [
        '<|open|>argument key="payload" type="string"<|sep|>unfinished',
        '<|open|>argument key="payload" type="string"',
        "<|open|>argumentbroken<|sep|>value",
    ],
)
def test_malformed_argument_cannot_be_closed_by_enclosing_marker(body):
    text = TOOLS_OPEN + _call("fixture", body) + TOOLS_CLOSE
    assert _parser().parse_non_stream(text)[1] == []
    assert _stream(text, text) == ("", [])


@pytest.mark.parametrize(
    "kind,raw",
    [("object", '{"incomplete":'), ("array", '["incomplete"'), ("number", "1e")],
)
def test_malformed_or_truncated_typed_json_never_emits_partial_call(kind, raw):
    body = f'<|open|>argument key="payload" type="{kind}"<|sep|>{raw}{ARGUMENT_CLOSE}'
    text = TOOLS_OPEN + _call("fixture", body) + TOOLS_CLOSE
    assert _parser().parse_non_stream(text)[1] == []
    assert _stream(text, text) == ("", [])


def test_first_tools_close_is_terminal_outside_arguments():
    text = (
        TOOLS_OPEN
        + _call("fixture", _argument("payload", "string", "first"))
        + TOOLS_CLOSE
        + _call("fixture", _argument("payload", "string", "after-close"))
    )
    _, calls = _parser().parse_non_stream(text)
    assert len(calls) == 1
    assert json.loads(calls[0].parameters) == {"payload": "first"}
    assert _stream(text, text) == ("", [call.model_dump() for call in calls])


def test_auto_answer_after_history_and_legitimate_followup_controls():
    # History is rendered before this detector; an answerable followup may be prose.
    parser = _parser(choice="auto")
    assert parser.parse_non_stream("The tool result is complete.") == (
        "The tool result is complete.",
        [],
    )
    assert _stream(
        "The tool result is complete.", "The tool result is complete.", choice="auto"
    ) == ("The tool result is complete.", [])
    text = (
        TOOLS_OPEN
        + _call("fixture", _argument("payload", "string", "needed followup"))
        + TOOLS_CLOSE
    )
    assert len(_parser(choice="auto").parse_non_stream(text)[1]) == 1


def test_complete_call_kept_before_truncated_next_call():
    first = _call("fixture", _argument("payload", "string", "complete"))
    partial = '<|open|>call tool="fixture" index="2"<|sep|><|open|>argument key="payload" type="string"<|sep|>unfinished'
    text = TOOLS_OPEN + first + partial
    _, calls = _parser().parse_non_stream(text)
    assert len(calls) == 1
    assert json.loads(calls[0].parameters) == {"payload": "complete"}
    assert _stream(text, text) == ("", [call.model_dump() for call in calls])


def test_escaped_attributes_and_empty_call_controls():
    text = (
        TOOLS_OPEN
        + _call("f&amp;&quot;", _argument("a&amp;&quot;", "string", "value"))
        + TOOLS_CLOSE
    )
    _, calls = _parser().parse_non_stream(text)
    assert calls[0].name == 'f&"'
    assert json.loads(calls[0].parameters) == {'a&"': "value"}
    empty = TOOLS_OPEN + _call("fixture") + TOOLS_CLOSE
    assert json.loads(_parser().parse_non_stream(empty)[1][0].parameters) == {}


def test_combined_live_fixture_preserves_raw_and_nested_markers():
    enclosing = (TOOLS_CLOSE, CALL_CLOSE, MESSAGE_CLOSE)
    label = "synthetic-prefix:" + ":".join(enclosing) + ":suffix"
    nested = {
        "markers": [*enclosing, ARGUMENT_CLOSE],
        "tail": "synthetic-after-markers",
    }
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "const": label},
            "nested": {
                "type": "object",
                "properties": {
                    "markers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "const": nested["markers"],
                    },
                    "tail": {"type": "string", "const": nested["tail"]},
                },
                "required": ["markers", "tail"],
                "additionalProperties": False,
            },
        },
        "required": ["label", "nested"],
        "additionalProperties": False,
    }
    tools = [_tool(schema, strict=True)]
    text = (
        TOOLS_OPEN
        + _call(
            "fixture",
            _argument("label", "json-string", label),
            _argument("nested", "object", nested),
        )
        + TOOLS_CLOSE
    )
    parser = _parser(tools)
    tag = parser.get_structure_constraint("required", parallel_tool_calls=False)[1]
    assert _is_grammar_accept_string(xgr.Grammar.from_structural_tag(tag), text)
    normal, calls = parser.parse_non_stream(text)
    assert len(calls) == 1
    assert json.loads(calls[0].parameters) == {"label": label, "nested": nested}
    expected = (normal, [call.model_dump() for call in calls])
    for split in range(len(text) + 1):
        assert _stream(text, (text[:split], text[split:]), tools) == expected, split


@pytest.mark.parametrize(
    "body",
    [
        "unexpected body text",
        _argument("payload", "string", "a" + ARGUMENT_CLOSE + " b"),
        _argument("payload", "string", "ok") + "unexpected tail",
    ],
)
def test_unparsed_call_body_text_rejected_instead_of_silently_dropped(body):
    text = TOOLS_OPEN + _call("fixture", body) + TOOLS_CLOSE
    assert _parser().parse_non_stream(text)[1] == []
    assert _stream(text, text) == ("", [])


def test_whitespace_between_complete_arguments_remains_valid():
    body = (
        " \n\t"
        + _argument("a", "string", "first")
        + "\r\n "
        + _argument("b", "number", 7)
        + " \n"
    )
    text = TOOLS_OPEN + _call("fixture", body) + TOOLS_CLOSE
    _, calls = _parser().parse_non_stream(text)
    assert len(calls) == 1
    assert json.loads(calls[0].parameters) == {"a": "first", "b": 7}
    assert _stream(text, text) == ("", [call.model_dump() for call in calls])
