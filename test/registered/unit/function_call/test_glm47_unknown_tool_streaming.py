"""Regression tests for GLM47 streaming unknown-tool filtering."""

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def known_tool():
    return Tool(
        function=Function(name="known", parameters={"type": "object", "properties": {}})
    )


def _parse_character_by_character(text, tools, *, forward_unknown):
    detector = Glm47MoeDetector()
    calls = []
    normal_text = []
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(forward_unknown):
        for character in text:
            result = detector.parse_streaming_increment(character, tools)
            normal_text.append(result.normal_text)
            calls.extend(result.calls)
    return calls, "".join(normal_text)


def _named_calls(calls):
    return [(call.tool_index, call.name) for call in calls if call.name is not None]


def test_unknown_call_is_dropped_before_following_known_call(known_tool):
    calls, normal_text = _parse_character_by_character(
        "<tool_call>unknown</tool_call><tool_call>known</tool_call>",
        [known_tool],
        forward_unknown=False,
    )

    assert _named_calls(calls) == [(0, "known")]
    assert [call.parameters for call in calls if call.name is None] == ["{}"]
    assert normal_text == ""


def test_unknown_arguments_do_not_leak_before_following_known_call(known_tool):
    calls, normal_text = _parse_character_by_character(
        (
            "<tool_call>unknown"
            "<arg_key>q</arg_key><arg_value>secret</arg_value>"
            "</tool_call><tool_call>known</tool_call>"
        ),
        [known_tool],
        forward_unknown=False,
    )

    assert _named_calls(calls) == [(0, "known")]
    assert [call.parameters for call in calls if call.name is None] == ["{}"]
    assert "secret" not in normal_text


def test_unknown_call_is_forwarded_when_explicitly_enabled(known_tool):
    calls, normal_text = _parse_character_by_character(
        "<tool_call>unknown</tool_call>",
        [known_tool],
        forward_unknown=True,
    )

    assert _named_calls(calls) == [(0, "unknown")]
    assert [call.parameters for call in calls if call.name is None] == ["{}"]
    assert normal_text == ""
