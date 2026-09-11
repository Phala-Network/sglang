"""A tool example inside reasoning is not an executable tool call."""

import pytest

from sglang.srt.parser.reasoning_parser import Nemotron3Detector


CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>"
EXAMPLE = "<tool_call>\n<function=FUNCTION_NAME>\n</function>\n</tool_call>"


def collect(detector, text, width):
    reasoning, content = [], []
    for start in range(0, len(text), width):
        part = detector.parse_streaming_increment(text[start : start + width])
        reasoning.append(part.reasoning_text)
        content.append(part.normal_text)
    part = detector.finish()
    reasoning.append(part.reasoning_text)
    content.append(part.normal_text)
    return "".join(reasoning), "".join(content)


@pytest.mark.parametrize("width", [1, 3, 7, 31, 10000])
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("stream_reasoning", [False, True])
def test_quoted_tool_before_explicit_reasoning_end(width, implicit, stream_reasoning):
    reasoning = "The format example is " + EXAMPLE + ". Now issue one actual call."
    text = ("" if implicit else "<think>") + reasoning + "</think>" + CALL
    detector = Nemotron3Detector(force_reasoning=implicit, stream_reasoning=stream_reasoning)
    assert collect(detector, text, width) == (reasoning, CALL)
    expected = Nemotron3Detector(force_reasoning=implicit).detect_and_parse(text)
    assert (expected.reasoning_text, expected.normal_text) == (reasoning, CALL)


@pytest.mark.parametrize("width", [1, 7, 10000])
@pytest.mark.parametrize("stream_reasoning", [False, True])
def test_missing_end_tool_fallback_is_preserved_at_eof(width, stream_reasoning):
    detector = Nemotron3Detector(stream_reasoning=stream_reasoning)
    assert collect(detector, "<think>Need current weather." + CALL, width) == (
        "Need current weather.", CALL
    )
    assert detector.finish().normal_text == ""


@pytest.mark.parametrize("width", [1, 7, 10000])
def test_nonreasoning_tools_are_not_deferred(width):
    detector = Nemotron3Detector(force_reasoning=False)
    assert collect(detector, CALL, width) == ("", CALL)


def test_ambiguous_marker_is_not_published_early():
    detector = Nemotron3Detector(force_reasoning=True)
    detector.parse_streaming_increment("Let me inspect this example: ")
    part = detector.parse_streaming_increment(EXAMPLE)
    assert part.normal_text == ""
    assert detector._in_reasoning
    part = detector.parse_streaming_increment(".</think>" + CALL)
    assert part.reasoning_text == EXAMPLE + "."
    assert part.normal_text == CALL
