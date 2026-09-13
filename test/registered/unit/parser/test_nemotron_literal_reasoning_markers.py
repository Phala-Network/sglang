"""Reasoning delimiters inside the final answer are data, not a new channel."""

import json

import pytest

from sglang.srt.parser.reasoning_parser import Nemotron3Detector

ANSWER = json.dumps({"text": "<think>literal not reasoning</think>", "end": "after"})


@pytest.mark.parametrize("prefix", ["", "analysis</think>", "<think>analysis</think>"])
@pytest.mark.parametrize("stream_reasoning", [False, True])
def test_literal_markers_preserved_at_every_split(prefix, stream_reasoning):
    raw = prefix + ANSWER
    force = prefix == "analysis</think>"
    expected_reasoning = "analysis" if prefix else ""
    detector = Nemotron3Detector(force_reasoning=force)
    result = detector.detect_and_parse(raw)
    assert (result.reasoning_text, result.normal_text) == (expected_reasoning, ANSWER)

    for cut in range(len(raw) + 1):
        detector = Nemotron3Detector(
            force_reasoning=force, stream_reasoning=stream_reasoning
        )
        parts = [
            detector.parse_streaming_increment(raw[:cut]),
            detector.parse_streaming_increment(raw[cut:]),
            detector.finish(),
        ]
        assert "".join(p.reasoning_text for p in parts) == expected_reasoning, cut
        assert "".join(p.normal_text for p in parts) == ANSWER, cut


@pytest.mark.parametrize("width", [1, 3, 7, 19, 10000])
@pytest.mark.parametrize("stream_reasoning", [False, True])
def test_answer_does_not_reopen_reasoning(width, stream_reasoning):
    detector = Nemotron3Detector(
        force_reasoning=True, stream_reasoning=stream_reasoning
    )
    raw = "analysis</think>" + ANSWER
    parts = [
        detector.parse_streaming_increment(raw[i : i + width])
        for i in range(0, len(raw), width)
    ]
    parts.append(detector.finish())
    assert "".join(p.reasoning_text for p in parts) == "analysis"
    assert "".join(p.normal_text for p in parts) == ANSWER


@pytest.mark.parametrize("previous", ['{"text":"', '<think>analysis</think>{"text":"'])
def test_continued_final_answer_preserves_a_leading_literal(previous):
    tail = '<think>literal</think>"}'
    kwargs = dict(continue_final_message=True, previous_content=previous)
    result = Nemotron3Detector(**kwargs).detect_and_parse(tail)
    assert (result.reasoning_text, result.normal_text) == ("", tail)
    detector = Nemotron3Detector(**kwargs)
    parts = [detector.parse_streaming_increment(c) for c in tail]
    parts.append(detector.finish())
    assert "".join(p.reasoning_text for p in parts) == ""
    assert "".join(p.normal_text for p in parts) == tail
