"""Budget-cut reasoning examples must not become executable tool calls."""
from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.parser.reasoning_parser import ReasoningParser

CALL = '<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>'


def parse_stream(text, width, stream_reasoning, finish_type):
    service = SimpleNamespace(
        template_manager=SimpleNamespace(force_reasoning=True),
        reasoning_parser='nemotron_3',
        tokenizer_manager=SimpleNamespace(tokenizer=None),
        _get_reasoning_from_request=lambda request: True,
        _tool_call_parsing_active=lambda request: True,
    )
    request = ChatCompletionRequest(model='test', messages=[{'role': 'user', 'content': 'test'}],
                                    stream=True, stream_reasoning=stream_reasoning)
    state = {}
    reasoning, content = [], []
    for start in range(0, len(text), width):
        delta = text[start:start + width]
        last = finish_type if start + width >= len(text) else None
        r, c = OpenAIServingChat._process_reasoning_stream(service, 0, delta, state, {}, request, last)
        reasoning.append(r or '')
        content.append(c or '')
    return ''.join(reasoning), ''.join(content)


@pytest.mark.parametrize('width', [1, 7, 10000])
@pytest.mark.parametrize('stream_reasoning', [False, True])
def test_length_does_not_execute_unclosed_reasoning(width, stream_reasoning):
    thought = 'Need weather. Here is an example: ' + CALL + '. Still thinking.'
    assert parse_stream(thought, width, stream_reasoning, 'length') == (thought, '')


@pytest.mark.parametrize('width', [1, 7, 10000])
@pytest.mark.parametrize('stream_reasoning', [False, True])
def test_normal_stop_preserves_missing_closer_fallback(width, stream_reasoning):
    assert parse_stream('Need weather.' + CALL, width, stream_reasoning, 'stop') == ('Need weather.', CALL)


@pytest.mark.parametrize('width', [1, 7, 10000])
@pytest.mark.parametrize('stream_reasoning', [False, True])
def test_length_after_closed_reasoning_preserves_actual_calls(width, stream_reasoning):
    thought = 'Example: ' + CALL + '. Now call.'
    assert parse_stream(thought + '</think>' + CALL, width, stream_reasoning, 'length') == (thought, CALL)


@pytest.mark.parametrize('finish_type', ['length', 'abort'])
@pytest.mark.parametrize('force_content', [False, True])
def test_nonstream_budget_cut_is_reasoning_even_with_force_content(finish_type, force_content):
    request = ChatCompletionRequest(model='test', messages=[{'role': 'user', 'content': 'test'}],
        chat_template_kwargs={'force_nonempty_content': force_content})
    parser = ReasoningParser('nemotron_3', force_reasoning=True, request=request)
    thought = 'Example: ' + CALL + '. Not finished.'
    assert parser.parse_non_stream(thought, finish_type) == (thought, '')


@pytest.mark.parametrize('finish_type', [None, 'stop'])
def test_nonstream_normal_completion_keeps_fallback(finish_type):
    parser = ReasoningParser('nemotron_3', force_reasoning=True)
    assert parser.parse_non_stream('Need weather.' + CALL, finish_type) == ('Need weather.', CALL)
