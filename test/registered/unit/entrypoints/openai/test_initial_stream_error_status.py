"""An initial error is an HTTP error; mid-stream errors remain SSE events."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat


@pytest.mark.parametrize("status", [400, 422, 429, 500, 503])
def test_initial_stream_error_uses_real_http_status_and_closes_generator(status):
    serving = OpenAIServingChat.__new__(OpenAIServingChat)
    serving.tokenizer_manager = SimpleNamespace(create_abort_task=MagicMock())
    closed = []

    async def generate(*_args):
        try:
            yield serving.create_streaming_error_response(
                "Invalid schema", status_code=status
            ).join(["data: ", "\n\n"])
            raise AssertionError("Must not consume a fake usage event after the error")
        finally:
            closed.append(True)

    serving._generate_chat_stream = generate
    response = asyncio.run(serving._handle_streaming_request(None, None, None))
    assert isinstance(response, ORJSONResponse)
    assert response.status_code == status
    assert json.loads(response.body)["message"] == "Invalid schema"
    assert closed == [True]
    serving.tokenizer_manager.create_abort_task.assert_not_called()


def test_successful_first_event_is_preserved_and_later_error_stays_streamed():
    serving = OpenAIServingChat.__new__(OpenAIServingChat)
    serving.tokenizer_manager = SimpleNamespace(
        create_abort_task=MagicMock(return_value=None)
    )
    role = 'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
    error = 'data: {"error":{"message":"later failure","code":500}}\n\n'

    async def generate(*_args):
        yield role
        yield error

    serving._generate_chat_stream = generate

    async def run():
        response = await serving._handle_streaming_request(None, None, None)
        assert isinstance(response, StreamingResponse)
        assert response.status_code == 200
        return [chunk async for chunk in response.body_iterator]

    assert asyncio.run(run()) == [role, error]
