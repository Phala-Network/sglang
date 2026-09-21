from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import asyncio
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

import orjson
import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from utils import engine_chunk, make_serving

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _openai_error(response, status, message):
    assert response.status_code == status
    assert orjson.loads(response.body) == {
        "object": "error",
        "message": message,
        "type": str(status),
        "param": None,
        "code": status,
    }


@pytest.fixture(autouse=True)
def isolated_runtime_context():
    reset_context()
    yield
    reset_context()


def _entrypoint_serving(serving_cls, generator_name, generator):
    serving = object.__new__(serving_cls)
    serving.tokenizer_manager = Mock()
    serving.tokenizer_manager.request_logger = Mock(
        log_requests=False, log_requests_level=0
    )
    serving.tokenizer_manager.create_abort_task.return_value = None
    serving._validate_request = Mock(return_value=None)
    adapted = SimpleNamespace(background=False)
    request = SimpleNamespace(stream=True)
    serving._convert_to_internal_request = Mock(return_value=(adapted, request))
    setattr(serving, generator_name, generator)
    return serving, request


@pytest.mark.parametrize(
    "serving_cls,generator_name",
    [
        (OpenAIServingChat, "_generate_chat_stream"),
        (OpenAIServingCompletion, "_generate_completion_stream"),
    ],
    ids=["chat-completions", "completions"],
)
def test_first_stream_item_429_returns_openai_http_error(
    serving_cls, generator_name
):
    async def reject(*args, **kwargs):
        raise HTTPException(status_code=429, detail="server overloaded")
        yield  # pragma: no cover

    serving, request = _entrypoint_serving(serving_cls, generator_name, reject)
    response = asyncio.run(serving.handle_request(request, raw_request=None))

    _openai_error(response, HTTPStatus.TOO_MANY_REQUESTS, "server overloaded")
    serving.tokenizer_manager.create_abort_task.assert_not_called()


@pytest.mark.parametrize(
    "serving_cls,generator_name",
    [
        (OpenAIServingChat, "_generate_chat_stream"),
        (OpenAIServingCompletion, "_generate_completion_stream"),
    ],
    ids=["chat-completions", "completions"],
)
@pytest.mark.parametrize(
    "chunks",
    [
        ["data: {\"id\":\"ok\"}\n\n", "data: [DONE]\n\n"],
        [
            'data: {"error":{"message":"worker unavailable","code":503}}\n\n',
            "data: [DONE]\n\n",
        ],
    ],
    ids=["normal", "503-error-chunk"],
)
def test_stream_still_starts_as_sse_and_replays_first_item(
    serving_cls, generator_name, chunks
):
    async def generate(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    serving, request = _entrypoint_serving(serving_cls, generator_name, generate)

    async def run():
        response = await serving.handle_request(request, raw_request=None)
        body = [chunk async for chunk in response.body_iterator]
        return response, body

    response, body = asyncio.run(run())
    assert isinstance(response, StreamingResponse)
    assert response.status_code == HTTPStatus.OK
    assert response.media_type == "text/event-stream"
    assert body == chunks
    serving.tokenizer_manager.create_abort_task.assert_called_once()


def _responses_serving(generator):
    serving = make_serving()
    serving.use_harmony = False
    serving.default_chat_template_kwargs = {}
    serving.template_manager.chat_template_name = None
    serving.template_manager.jinja_template_content_format = "string"
    serving.tokenizer_manager.tokenizer.apply_chat_template.return_value = [1, 2, 3]
    serving.tokenizer_manager.generate_request = Mock(side_effect=generator)
    serving.reasoning_parser = None
    serving.tool_call_parser = None
    return serving


def test_responses_first_stream_item_429_returns_openai_http_error():
    async def reject(*args, **kwargs):
        raise HTTPException(status_code=429, detail="server overloaded")
        yield  # pragma: no cover

    serving = _responses_serving(reject)
    response = asyncio.run(
        serving.create_responses(
            ResponsesRequest(model="x", input="hi", stream=True, store=False)
        )
    )

    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    error = orjson.loads(response.body)["error"]
    assert error["message"] == "server overloaded"
    assert error["code"] == HTTPStatus.TOO_MANY_REQUESTS


def test_responses_normal_stream_remains_200_sse():
    async def generate(*args, **kwargs):
        yield engine_chunk("ok", finish=True)

    serving = _responses_serving(generate)

    async def run():
        response = await serving.create_responses(
            ResponsesRequest(model="x", input="hi", stream=True, store=False)
        )
        body = [chunk async for chunk in response.body_iterator]
        return response, body

    response, body = asyncio.run(run())
    assert response.status_code == HTTPStatus.OK
    assert response.media_type == "text/event-stream"
    assert any("response.completed" in chunk for chunk in body)


@pytest.mark.parametrize(
    "status,raises",
    [(HTTPStatus.TOO_MANY_REQUESTS, True), (HTTPStatus.SERVICE_UNAVAILABLE, False)],
    ids=["429-before-headers", "503-stream-chunk"],
)
def test_tokenizer_abort_mapping_for_streams(status, raises):
    manager = object.__new__(TokenizerManager)
    manager.rid_to_state = {"rid": Mock()}
    manager.enable_lora = False
    state = SimpleNamespace(obj=SimpleNamespace(rid="rid", lora_path=None))
    output = {
        "meta_info": {
            "finish_reason": {
                "type": "abort",
                "status_code": status,
                "message": "overloaded",
            }
        }
    }

    async def run():
        return await manager._handle_abort_finish_reason(output, state, is_stream=True)

    if raises:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(run())
        assert exc_info.value.status_code == status
        assert exc_info.value.detail == "overloaded"
    else:
        assert asyncio.run(run()) is output
    assert "rid" not in manager.rid_to_state
