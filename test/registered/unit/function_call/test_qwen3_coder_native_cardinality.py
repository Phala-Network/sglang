"""Native XML cardinality regressions; no model output or GPU required."""
import json

import pytest
import xgrammar as xg
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai.protocol import Function, Tool, ToolChoice
from sglang.srt.function_call.function_call_parser import FunctionCallParser


TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="lookup",
            strict=True,
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        ),
    )
]


def call(value="first"):
    return (
        "<tool_call>\n<function=lookup>\n<parameter=key>\n"
        + value
        + "\n</parameter>\n</function>\n</tool_call>"
    )


def grammar(mode, parallel, reasoning=False, strict=True):
    tools = [tool.model_copy(deep=True) for tool in TOOLS]
    tools[0].function.strict = strict
    choice = (
        ToolChoice(type="function", function={"name": "lookup"})
        if mode == "named"
        else mode
    )
    kind, tag = FunctionCallParser(tools, "qwen3_coder").get_structure_constraint(
        choice, parallel_tool_calls=parallel, thinking_mode=reasoning
    )
    assert kind == "structural_tag"
    return xg.Grammar.from_structural_tag(tag)


@pytest.mark.parametrize("mode", ["required", "named"])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("reasoning", [False, True])
def test_native_cardinality(mode, parallel, reasoning):
    g = grammar(mode, parallel, reasoning)
    prefix = "I need independent lookups.</think>\n\n" if reasoning else ""
    accepts = lambda text: _is_grammar_accept_string(g, prefix + text)
    assert accepts(call())
    assert accepts("\n" + call() + "\n")
    assert accepts(call() + call("second")) is parallel
    assert accepts(call() + "\n\n" + call("second")) is parallel
    assert not accepts("")
    assert not accepts("Only prose")
    assert not accepts(call() + "\nContinue narrating")
    assert not accepts(call() + "\nProse between\n" + call("second"))
    assert not accepts(" " * 65 + call())
    assert not accepts(call()[:-5])
    assert not accepts(call().replace("lookup", "unknown"))


@pytest.mark.parametrize("strict", [False, True])
def test_auto_false_still_allows_text_but_only_one_call(strict):
    g = grammar("auto", False, strict=strict)
    assert _is_grammar_accept_string(g, "No lookup is needed.")
    assert _is_grammar_accept_string(g, call())
    assert not _is_grammar_accept_string(g, call() + "\n" + call("second"))


def test_auto_true_can_introduce_multiple_calls_with_text():
    g = grammar("auto", True)
    assert _is_grammar_accept_string(g, "No lookup is needed.")
    assert _is_grammar_accept_string(g, "Looking up.\n" + call() + "\n" + call("second"))


@pytest.mark.parametrize("strict", [False, True])
def test_auto_enters_one_final_call_phase(strict):
    g = grammar("auto", True, strict=strict)
    assert _is_grammar_accept_string(g, "No lookup is needed.")
    assert _is_grammar_accept_string(g, "Looking up.\n" + call() + "\n" + call("second"))
    # Identical calls can be legitimate; cardinality remains the model's choice.
    assert _is_grammar_accept_string(g, call() + "\n" + call())
    assert not _is_grammar_accept_string(g, call() + "\nBut need to check the format.")
    assert not _is_grammar_accept_string(g, call() + "\nNow another call\n" + call())
    assert not _is_grammar_accept_string(g, call() + "\n" * 65)
    bare = call().removeprefix("<tool_call>\n").removesuffix("\n</tool_call>")
    assert _is_grammar_accept_string(g, bare + "\n" + call("second"))


@pytest.mark.parametrize("closing_wrapper", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_bare_function_is_constrained_and_parses_like_streaming(closing_wrapper, strict):
    bare = call().removeprefix("<tool_call>\n")
    if not closing_wrapper:
        bare = bare.removesuffix("\n</tool_call>")
    g = grammar("auto", True, strict=strict)
    assert _is_grammar_accept_string(g, bare)
    assert not _is_grammar_accept_string(g, bare.replace("lookup", "unknown"))
    # Preserve upstream strict=False's unconstrained argument schema, while
    # enforcing names in both modes. Bare and wrapped spellings must agree.
    bad_arg = bare.replace("<parameter=key>", "<parameter=unknown>")
    assert _is_grammar_accept_string(g, bad_arg) is (not strict)
    assert _is_grammar_accept_string(
        g, call().replace("<parameter=key>", "<parameter=unknown>")
    ) is (not strict)
    parser = FunctionCallParser(TOOLS, "qwen3_coder")
    assert parser.has_tool_call(bare)
    normal, calls = parser.parse_non_stream(bare + "\n" + call("second"))
    assert not normal.strip()
    assert [json.loads(c.parameters) for c in calls] == [{"key": "first"}, {"key": "second"}]


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_auto_triggers_at_tool_marker_not_after_function_header(strict, parallel):
    g = grammar("auto", parallel, strict=strict)
    assert _is_grammar_accept_string(g, "No lookup is needed.")
    assert _is_grammar_accept_string(g, call())
    # These previously bypassed the trigger '<tool_call>\n<function=' and
    # were accepted as plain text, despite already emitting a tool marker.
    for malformed in [
        "<tool_call><function=lookup>anything</function></tool_call>",
        '<tool_call>\n<function name="lookup">anything</function></tool_call>',
        "<tool_call>\nI will invoke lookup now.",
        "<tool_call>\n<function=unknown>anything</function></tool_call>",
    ]:
        assert not _is_grammar_accept_string(g, malformed)


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 31, 10000])
def test_parallel_parser_nonstream_stream_parity(chunk_size):
    text = call("first") + "\n\n" + call("second")
    parser = FunctionCallParser(TOOLS, "qwen3_coder")
    normal, calls = parser.parse_non_stream(text)
    assert not normal
    assert [json.loads(c.parameters) for c in calls] == [{"key": "first"}, {"key": "second"}]
    stream_parser = FunctionCallParser(TOOLS, "qwen3_coder")
    reconstructed = {}
    for start in range(0, len(text), chunk_size):
        normal, calls = stream_parser.parse_stream_chunk(text[start : start + chunk_size])
        assert not normal.strip()
        for c in calls:
            row = reconstructed.setdefault(c.tool_index, {"name": "", "arguments": ""})
            if c.name:
                row["name"] = c.name
            if c.parameters:
                assert row["name"]
                row["arguments"] += c.parameters
    assert sorted(reconstructed) == [0, 1]
    assert [json.loads(reconstructed[i]["arguments"]) for i in [0, 1]] == [{"key": "first"}, {"key": "second"}]
