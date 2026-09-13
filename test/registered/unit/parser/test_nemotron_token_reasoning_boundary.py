"""Only a generated control token may close Nemotron reasoning."""

import json
from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.parser.reasoning_parser import (
    BaseReasoningFormatDetector,
    ReasoningParser,
)


class LiteralTokenizer:
    """Same visible marker can be ordinary pieces or an added control ID."""

    @staticmethod
    def convert_tokens_to_ids(text):
        return {"<think>": 12, "</think>": 13}[text]

    @staticmethod
    def decode(ids, **kwargs):
        return "".join(
            {12: "<think>", 13: "</think>", 2: ""}.get(
                i, chr(i - 1000) if i >= 1000 else ""
            )
            for i in ids
        )


def ordinary(text):
    return [1000 + ord(c) for c in text]


THOUGHT = 'Quoted JSON {"text":"<think>中文 😀</think>"} is only an example. Continue reasoning.'
ANSWER = json.dumps({"text": "<think>literal</think>", "ok": True}, ensure_ascii=False)


def test_generic_stream_requires_complete_self_labeled_opener():
    detector = BaseReasoningFormatDetector("<think>", "</think>")
    detector.think_start_self_label = "analysis"
    result = detector.parse_streaming_increment("<think>literal content")
    assert result.normal_text == "<think>literal content"
    assert result.reasoning_text == ""


@pytest.mark.parametrize("width", [1, 3, 7, 29, 10000])
@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("stream_reasoning", [False, True])
@pytest.mark.parametrize("opening", [False, True])
def test_chat_stream_uses_control_id_not_quoted_text(
    width, incremental, stream_reasoning, opening
):
    tokenizer = LiteralTokenizer()
    serving = object.__new__(OpenAIServingChat)
    serving.reasoning_parser = "nemotron_3"
    serving.template_manager = SimpleNamespace(force_reasoning=False)
    serving.tokenizer_manager = SimpleNamespace(
        tokenizer=tokenizer,
        server_args=SimpleNamespace(incremental_streaming_output=incremental),
    )
    serving._get_reasoning_from_request = lambda _: True
    serving._tool_call_parsing_active = lambda _: False
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "Return JSON"}],
        stream=True,
        stream_reasoning=stream_reasoning,
    )
    ids = ([12] if opening else []) + ordinary(THOUGHT) + [13] + ordinary(ANSWER)
    states = {}
    results = []
    for start in range(0, len(ids), width):
        end = min(start + width, len(ids))
        delta = tokenizer.decode(ids[start:end])
        results.append(
            serving._process_reasoning_stream(
                0,
                delta,
                states,
                {"output_ids": ids[start:end] if incremental else ids[:end]},
                request,
                "stop" if end == len(ids) else None,
            )
        )
    assert "".join(r or "" for r, _ in results) == THOUGHT
    assert "".join(c or "" for _, c in results) == ANSWER


@pytest.mark.parametrize("opening", [False, True])
@pytest.mark.parametrize("finish", ["stop", "length"])
def test_nonstream_token_boundary_and_truncation(opening, finish):
    tokenizer = LiteralTokenizer()
    ids = ([12] if opening else []) + ordinary(THOUGHT)
    if finish == "stop":
        ids += [13] + ordinary(ANSWER)
    parser = ReasoningParser("nemotron_3", force_reasoning=True, tokenizer=tokenizer)
    reasoning, content = parser.parse_non_stream(
        tokenizer.decode(ids), finish_reason_type=finish, output_ids=ids
    )
    assert reasoning == THOUGHT
    assert content == (ANSWER if finish == "stop" else "")


def test_disabled_reasoning_keeps_a_literal_opening_marker():
    text = "<think>literal</think>"
    parser = ReasoningParser(
        "nemotron_3", force_reasoning=False, tokenizer=LiteralTokenizer()
    )
    assert parser.parse_non_stream(
        text, finish_reason_type="stop", output_ids=ordinary(text)
    ) == ("", text)


@pytest.mark.parametrize("stream_reasoning", [False, True])
@pytest.mark.parametrize("finish", ["stop", "length"])
def test_reasoning_can_start_with_literal_marker_data(stream_reasoning, finish):
    thought = "<think>quoted literal</think> remains reasoning"
    ids = ordinary(thought) + ([13] + ordinary(ANSWER) if finish == "stop" else [])
    tokenizer = LiteralTokenizer()
    parser = ReasoningParser(
        "nemotron_3",
        force_reasoning=True,
        tokenizer=tokenizer,
        stream_reasoning=stream_reasoning,
    )
    assert parser.parse_non_stream(
        tokenizer.decode(ids), finish_reason_type=finish, output_ids=ids
    ) == (thought, ANSWER if finish == "stop" else "")
    parser = ReasoningParser(
        "nemotron_3",
        force_reasoning=True,
        tokenizer=tokenizer,
        stream_reasoning=stream_reasoning,
    )
    results = [
        parser.parse_stream_chunk(
            tokenizer.decode([token_id]), output_ids=[token_id], incremental_output=True
        )
        for token_id in ids
    ]
    results.append(parser.parse_stream_end(finish_reason_type=finish))
    assert "".join(r or "" for r, _ in results) == thought
    assert "".join(c or "" for _, c in results) == (ANSWER if finish == "stop" else "")
