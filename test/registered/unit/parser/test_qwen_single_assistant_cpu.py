"""Real Qwen parser regressions; no engine, model or tokenizer load."""

import pytest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.parser.reasoning_parser import (
    BaseReasoningFormatDetector,
    Qwen3Detector,
    ReasoningParser,
)

CASES = [
    (
        False,
        '{"key":"value"}\nuser\nassistant\n<think>later unfinished',
        "",
        '{"key":"value"}\nuser\nassistant\n<think>later unfinished',
    ),
    (False, "answer<think>literal</think>tail", "", "answer<think>literal</think>tail"),
    (
        False,
        'A literal "<think>" is not a new assistant.',
        "",
        'A literal "<think>" is not a new assistant.',
    ),
    (
        False,
        '{"key":"value"}\nuser\nassistant\n<think>later</think>tail',
        "",
        '{"key":"value"}\nuser\nassistant\n<think>later</think>tail',
    ),
    (
        False,
        "<think>thought</think>answer<think>literal</think>",
        "thought",
        "answer<think>literal</think>",
    ),
    (False, "<think></think><think>literal</think>", "", "<think>literal</think>"),
    (False, "<think><think>thought</think>answer", "thought", "answer"),
    (False, "\n<think>thought</think>\nanswer", "\nthought", "\nanswer"),
    (False, "<think>thought</thi", "thought</thi", ""),
    (False, "<think>thought<", "thought<", ""),
    (False, "<thi", "", "<thi"),
    (False, "answer<thi", "", "answer<thi"),
    (
        False,
        "\r\n🚀 answer字<think>literal</think>尾",
        "",
        "\r\n🚀 answer字<think>literal</think>尾",
    ),
    (False, "</think>answer", "", "</think>answer"),
    (False, " \n\t", "", " \n\t"),
    (
        True,
        "prefilled thought</think>answer<think>literal</think>",
        "prefilled thought",
        "answer<think>literal</think>",
    ),
    (True, "<think>thought</think>answer", "thought", "answer"),
    (True, "<thi", "<thi", ""),
    (True, "thought </thi", "thought </thi", ""),
    (True, "</think>answer", "", "answer"),
]


def streamed(detector, chunks):
    reasoning, content = [], []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk)
        reasoning.append(result.reasoning_text)
        content.append(result.normal_text)
    result = detector.finish()
    reasoning.append(result.reasoning_text)
    content.append(result.normal_text)
    # A second final flush cannot duplicate content or a partial marker.
    final = detector.finish()
    assert final.normal_text == final.reasoning_text == ""
    return "".join(reasoning), "".join(content)


@pytest.mark.parametrize("force,text,reasoning,content", CASES)
@pytest.mark.parametrize("stream_reasoning", [False, True])
def test_stream_and_nonstream_match_at_every_split(
    force, text, reasoning, content, stream_reasoning
):
    expected = (reasoning, content)
    result = Qwen3Detector(force_reasoning=force).detect_and_parse(text)
    assert (result.reasoning_text, result.normal_text) == expected
    for split in range(len(text) + 1):
        detector = Qwen3Detector(
            force_reasoning=force, stream_reasoning=stream_reasoning
        )
        assert streamed(detector, [text[:split], text[split:]]) == expected
    detector = Qwen3Detector(force_reasoning=force, stream_reasoning=stream_reasoning)
    assert streamed(detector, list(text)) == expected


@pytest.mark.parametrize("stream_reasoning", [False, True])
def test_visible_content_is_never_reclassified_or_removed(stream_reasoning):
    detector = Qwen3Detector(stream_reasoning=stream_reasoning)
    first = detector.parse_streaming_increment("visible prefix")
    assert first.normal_text == "visible prefix"
    assert first.reasoning_text == ""
    tail = "\nuser\nassistant\n<think>body literal</think>tail<thi"
    assert streamed(detector, list(tail)) == ("", tail)


def test_disabled_mode_does_not_hide_a_real_leading_thought():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Answer directly."}],
        chat_template_kwargs={"enable_thinking": False},
    )
    parser = ReasoningParser("qwen3", force_reasoning=False, request=request)
    assert parser.parse_non_stream("<think>actual thought</think>answer") == (
        "actual thought",
        "answer",
    )
    parser = ReasoningParser("qwen3", force_reasoning=False, request=request)
    text = "answer<think>literal</think>"
    assert parser.parse_non_stream(text) == ("", text)


@pytest.mark.parametrize(
    "previous,text,expected",
    [
        ("prior content", "<think>literal</think>", ("", "<think>literal</think>")),
        ("visible prefix<think>literal", " suffix", ("", " suffix")),
        (
            "<think>thought<tool_call>payload",
            " suffix<think>literal</think>",
            ("", " suffix<think>literal</think>"),
        ),
        ("<think>prior thought", "more</think>answer", ("more", "answer")),
        ("<think>prior thought", "<think>literal", ("<think>literal", "")),
        (
            "<think>prior thought",
            'use "<think>" literally</think>answer',
            ('use "<think>" literally', "answer"),
        ),
        (
            "<think>prior</think>",
            "<think>literal</think>",
            ("", "<think>literal</think>"),
        ),
    ],
)
@pytest.mark.parametrize("stream_reasoning", [False, True])
def test_continuation_preserves_the_existing_assistant_phase(
    previous, text, expected, stream_reasoning
):
    create = lambda: Qwen3Detector(
        continue_final_message=True,
        previous_content=previous,
        stream_reasoning=stream_reasoning,
    )
    result = create().detect_and_parse(text)
    assert (result.reasoning_text, result.normal_text) == expected
    assert streamed(create(), list(text)) == expected


@pytest.mark.parametrize("stream_reasoning", [False, True])
def test_tool_interruption_still_closes_real_initial_reasoning(stream_reasoning):
    text = "<think>thought<tool_call>payload</tool_call><think>literal</think>"
    expected = ("thought", "<tool_call>payload</tool_call><think>literal</think>")
    result = Qwen3Detector().detect_and_parse(text)
    assert (result.reasoning_text, result.normal_text) == expected
    assert (
        streamed(Qwen3Detector(stream_reasoning=stream_reasoning), list(text))
        == expected
    )


def test_reusing_nonstream_parser_does_not_retain_a_previous_choice_phase():
    parser = ReasoningParser("qwen3", force_reasoning=False)
    for text, expected in [
        ("answer<think>literal</think>", ("", "answer<think>literal</think>")),
        ("<thi", ("", "<thi")),
        ("<think>real thought</think>answer", ("real thought", "answer")),
        ("another answer", ("", "another answer")),
    ]:
        assert parser.parse_non_stream(text) == expected


def test_real_chat_stream_choice_parsers_do_not_share_terminal_content_state():
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    chat = object.__new__(OpenAIServingChat)
    chat.reasoning_parser = "qwen3"
    chat.template_manager = SimpleNamespace(force_reasoning=False)
    chat.tokenizer_manager = SimpleNamespace(tokenizer=None)
    chat._get_reasoning_from_request = lambda request: False
    chat._tool_call_parsing_active = lambda request: False
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Answer directly."}],
        n=2,
        stream=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    parsers = {}
    outputs = {0: ["", ""], 1: ["", ""]}
    for index, chunk, finished in [
        (0, "visible", None),
        (1, "<thi", None),
        (0, "<think>literal</think>", "stop"),
        (1, "nk>real thought</think>answer", "stop"),
    ]:
        reasoning, content = chat._process_reasoning_stream(
            index, chunk, parsers, {}, request, finished
        )
        outputs[index][0] += reasoning or ""
        outputs[index][1] += content or ""
    assert outputs == {
        0: ["", "visible<think>literal</think>"],
        1: ["real thought", "answer"],
    }
    assert parsers[0] is not parsers[1]


@pytest.mark.parametrize(
    "name",
    ["dots", "minimax", "nanbeige", "interns1", "deepseek-v3", "mimo", "poolside_v1"],
)
def test_non_qwen_aliases_and_subclasses_preserve_actual_parse_behavior(name):
    parser = ReasoningParser(name)
    detector = parser.detector
    for text in (
        "prefix<think>thought</think>answer",
        "<think>thought</think>answer",
        "plain",
    ):
        legacy = BaseReasoningFormatDetector(
            detector.think_start_token,
            detector.think_end_token,
            tool_start_token=detector.tool_start_token,
            force_reasoning=detector.force_reasoning,
        ).detect_and_parse(text)
        assert parser.parse_non_stream(text) == (
            legacy.reasoning_text,
            legacy.normal_text,
        )
