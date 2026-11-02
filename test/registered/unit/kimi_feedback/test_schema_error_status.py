"""Synthetic HTTP/manager status contracts, no inference traffic."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import responses_cpu_helpers  # noqa: F401 - initialize CPU dependency stubs
from fastapi import HTTPException
from test_stream_outcomes import Generation, events, run_case

from sglang.srt.entrypoints.openai import serving_chat as module
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

KINDS = {
    400: "invalid_request_error",
    500: "internal_server_error",
    503: "service_unavailable_error",
    504: "timeout_error",
}


def abort(code):
    return {
        "type": "abort",
        "status_code": code,
        "err_type": KINDS[code],
        "message": "Synthetic grammar failure.",
    }


def config():
    return SimpleNamespace(
        stream_response_default_include_usage=False,
        return_input_ids=False,
        return_output_ids=False,
        enable_cache_report=False,
        incremental_streaming_output=False,
    )


@pytest.mark.parametrize("code", KINDS)
def test_chat_first_abort_real_http_status(code):
    generation = Generation(finish={0: abort(code)})
    serving = object.__new__(module.OpenAIServingChat)
    serving.chat_encoding_spec = None
    serving.tokenizer_manager = SimpleNamespace(
        generate_request=lambda *a: generation, create_abort_task=lambda *a: None
    )
    request = ChatCompletionRequest(
        model="synthetic",
        messages=[{"role": "user", "content": "synthetic"}],
        stream=True,
    )
    with patch.object(module, "get_serving", return_value=config()):
        response = asyncio.run(
            serving._handle_streaming_request(SimpleNamespace(), request, None)
        )
    assert response.status_code == code
    assert json.loads(response.body)["code"] == code
    assert json.loads(response.body)["type"] == KINDS[code]
    assert generation.closed == 1


@pytest.mark.parametrize("code", KINDS)
def test_chat_late_abort_preserves_code_without_success(code):
    generation = Generation(
        finish={0: {"type": "stop"}, 1: abort(code)}, indices=(0, 1)
    )
    # The fork's main line scopes this contract to the K3 encoding.
    wire, error = run_case(generation, n=2, encoding_spec="kimi_k3")
    assert error is None
    payload = events(wire)
    assert [e["error"]["code"] for e in payload if "error" in e] == [code]
    assert [e["error"]["type"] for e in payload if "error" in e] == [KINDS[code]]
    assert not [
        c for e in payload for c in e.get("choices", []) if c.get("finish_reason")
    ]
    assert wire.count("[DONE]") == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("code", KINDS)
def test_tokenizer_abort_contract(code, stream):
    state = SimpleNamespace(obj=SimpleNamespace(rid="synthetic"))
    # Main only drops its own state and releases LoRA through one helper.
    manager = SimpleNamespace(
        rid_to_state={"synthetic": state},
        enable_lora=False,
        _release_lora_once=lambda state: None,
    )
    output = {"meta_info": {"finish_reason": abort(code)}}
    call = TokenizerManager._handle_abort_finish_reason(manager, output, state, stream)
    if stream:
        assert asyncio.run(call) is output
    elif code == 400:
        with pytest.raises(ValueError) as caught:
            asyncio.run(call)
        assert caught.value.error_type == KINDS[code]
    else:
        with pytest.raises(HTTPException) as caught:
            asyncio.run(call)
        assert caught.value.status_code == code
        assert caught.value.error_type == KINDS[code]
    if code != 400:
        assert not manager.rid_to_state


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "number", "multipleOf": 2},
        {"type": "array", "uniqueItems": True},
        {"type": "invalid-private-type"},
    ],
)
def test_selected_unsupported_schema_early_400_no_values(schema):
    serving = object.__new__(module.OpenAIServingChat)
    serving.chat_encoding_spec = "kimi_k3"
    with pytest.raises(HTTPException) as caught:
        serving._validate_active_json_constraint({"json_schema": json.dumps(schema)})
    assert caught.value.status_code == 400
    assert "private" not in caught.value.detail


@pytest.mark.parametrize(
    "schema",
    [{"type": "string"}, {"type": "object", "properties": {"x": {"type": "integer"}}}],
)
def test_selected_supported_schema_passes(schema):
    serving = object.__new__(module.OpenAIServingChat)
    serving.chat_encoding_spec = "kimi_k3"
    serving._validate_active_json_constraint({"json_schema": json.dumps(schema)})
    serving._validate_active_json_constraint({"temperature": 1})


@pytest.mark.parametrize("code", KINDS)
def test_responses_prefetch_abort_before_created(code):
    serving = object.__new__(OpenAIServingResponses)
    closed = []

    async def generation():
        try:
            yield {"meta_info": {"finish_reason": abort(code)}}
        finally:
            closed.append(True)

    with pytest.raises(HTTPException) as caught:
        asyncio.run(serving._prefetch_response_generation(generation()))
    assert caught.value.status_code == code
    assert caught.value.error_type == KINDS[code]
    assert closed == [True]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_http_selected_schema_preflight_before_generation(stream, active):
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import ORJSONResponse

    app = FastAPI()
    serving = object.__new__(module.OpenAIServingChat)
    serving.chat_encoding_spec = "kimi_k3"
    serving._validate_request = lambda request: None
    calls = []
    serving.tokenizer_manager = SimpleNamespace(
        request_logger=SimpleNamespace(log_requests=False)
    )
    params = (
        {"json_schema": json.dumps({"type": "number", "multipleOf": 2})}
        if active
        else {}
    )
    serving._convert_to_internal_request = lambda request, raw: (
        SimpleNamespace(sampling_params=params),
        request,
    )

    async def generate(*args):
        calls.append(True)
        return ORJSONResponse({"ok": True})

    serving._handle_streaming_request = generate
    serving._handle_non_streaming_request = generate

    @app.post("/v1/chat/completions")
    async def route(raw: Request):
        return await serving.handle_request(
            ChatCompletionRequest(**await raw.json()), raw
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://fixture"
        ) as client:
            return await client.post(
                "/v1/chat/completions",
                json={
                    "model": "synthetic",
                    "messages": [{"role": "user", "content": "synthetic"}],
                    "stream": stream,
                },
            )

    response = asyncio.run(run())
    assert response.status_code == (400 if active else 200)
    assert len(calls) == (0 if active else 1)


@pytest.mark.parametrize("fault,code", [(RuntimeError, 500), (TimeoutError, 504)])
def test_chat_prestart_service_failure_http(fault, code):
    generation = Generation(empty=True, fault=fault)
    serving = object.__new__(module.OpenAIServingChat)
    serving.chat_encoding_spec = None
    serving.tokenizer_manager = SimpleNamespace(
        generate_request=lambda *a: generation, create_abort_task=lambda *a: None
    )
    request = ChatCompletionRequest(
        model="synthetic",
        messages=[{"role": "user", "content": "synthetic"}],
        stream=True,
    )
    with patch.object(module, "get_serving", return_value=config()):
        response = asyncio.run(
            serving._handle_streaming_request(SimpleNamespace(), request, None)
        )
    assert response.status_code == code
    assert "PRIVATE" not in response.body.decode()
    assert generation.closed == 1


@pytest.mark.parametrize("code", KINDS)
def test_responses_late_abort_failed_event(code):
    from responses_cpu_helpers import (
        StreamFixture,
        engine_chunk,
        event_payloads,
        make_serving,
    )

    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

    serving = make_serving()
    serving.chat_encoding_spec = "kimi_k3"  # main scopes this contract to K3
    request = ResponsesRequest(model="x", input="synthetic", store=False)
    chunks = [
        engine_chunk("hello", completion_tokens=1, finish=False),
        engine_chunk("hello", completion_tokens=1, finish=False),
    ]
    chunks[-1]["meta_info"]["finish_reason"] = abort(code)
    payloads = event_payloads(
        StreamFixture(serving, request, require_reasoning=False).run(chunks)
    )
    assert payloads[-1]["type"] == "response.failed"
    assert payloads[-1]["response"]["error"]["code"] == (
        "invalid_prompt" if code == 400 else "server_error"
    )
    assert [p["code"] for p in payloads if p["type"] == "error"] == [str(code)]
    assert not any(x["type"] == "response.completed" for x in payloads)


def test_responses_missing_finish_is_not_completed():
    from responses_cpu_helpers import (
        StreamFixture,
        engine_chunk,
        event_payloads,
        make_serving,
    )

    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

    serving = make_serving()
    request = ResponsesRequest(model="x", input="synthetic", store=False)
    payloads = event_payloads(
        StreamFixture(serving, request, require_reasoning=False).run(
            [engine_chunk("hello", completion_tokens=1, finish=False)]
        )
    )
    assert payloads[-1]["type"] == "response.failed"
    assert not any(x["type"] == "response.completed" for x in payloads)


@pytest.mark.parametrize("code", [None, 400, 500, 503, 504])
def test_responses_actual_handler_prefetch_and_cleanup(code):
    from responses_cpu_helpers import engine_chunk, event_payloads, make_serving

    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
    from sglang.srt.entrypoints.openai.serving_chat import MessageProcessingResult

    serving = make_serving()
    rendered = MessageProcessingResult(
        prompt="synthetic",
        prompt_ids=[1, 2, 3],
        image_data=None,
        audio_data=None,
        video_data=None,
        modalities=[],
        stop=[],
    )
    closed = []
    calls = []

    async def generate(*args):
        try:
            calls.append(True)
            chunk = engine_chunk("hello", finish=code is None)
            if code is not None:
                chunk["meta_info"]["finish_reason"] = abort(code)
            yield chunk
        finally:
            closed.append(True)

    serving.tokenizer_manager.generate_request = generate
    request = ResponsesRequest(model="x", input="synthetic", stream=True, store=False)

    async def run():
        response = await serving.create_responses(request)
        if code is not None:
            return response
        return [chunk async for chunk in response.body_iterator]

    with patch.object(serving, "_apply_conversation_template", return_value=rendered):
        response = asyncio.run(run())
    assert closed == [True] and calls == [True]
    if code is not None:
        assert response.status_code == code
        assert json.loads(response.body)["error"]["code"] == code
    else:
        payloads = event_payloads(response)
        assert [p["type"] for p in payloads[:2]] == [
            "response.created",
            "response.in_progress",
        ]
        assert payloads[-1]["type"] == "response.completed"


@pytest.mark.parametrize(
    "strict,renderer,active",
    [(True, True, True), (False, True, False), (False, False, True)],
)
def test_preflight_respects_actual_strict_false_selection(strict, renderer, active):
    request = ChatCompletionRequest(
        model="synthetic",
        messages=[{"role": "user", "content": "synthetic"}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "synthetic",
                "strict": strict,
                "schema": {"type": "number", "multipleOf": 2},
            },
        },
    )
    sampling = request.to_sampling_params(
        [], {}, renderer_handles_response_format=renderer
    )
    serving = object.__new__(module.OpenAIServingChat)
    serving.chat_encoding_spec = "kimi_k3"
    if active:
        with pytest.raises(HTTPException) as caught:
            serving._validate_active_json_constraint(sampling)
        assert caught.value.status_code == 400
    else:
        assert "json_schema" not in sampling
        serving._validate_active_json_constraint(sampling)


def test_preflight_bounded_cost_record(tmp_path):
    import statistics
    import time

    serving = object.__new__(module.OpenAIServingChat)
    serving.chat_encoding_spec = "kimi_k3"
    measurements = {}
    for count in (1, 32):
        sampling = {
            "json_schema": json.dumps(
                {
                    "type": "object",
                    "properties": {
                        f"field{i}": {"type": "integer"} for i in range(count)
                    },
                }
            )
        }
        times = []
        for _ in range(20):
            started = time.perf_counter()
            serving._validate_active_json_constraint(sampling)
            times.append((time.perf_counter() - started) * 1000)
        measurements[str(count)] = {
            "median_ms": statistics.median(times),
            "max_ms": max(times),
            "iterations": len(times),
        }
    (tmp_path / "preflight-perf.log").write_text(json.dumps(measurements))


@pytest.mark.parametrize("stream", [False, True])
def test_responses_actual_handler_rejects_schema_without_generation(stream):
    from responses_cpu_helpers import make_serving

    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
    from sglang.srt.entrypoints.openai.serving_chat import MessageProcessingResult

    serving = make_serving()
    serving.chat_encoding_spec = "kimi_k3"
    rendered = MessageProcessingResult(
        prompt="synthetic",
        prompt_ids=[1, 2, 3],
        image_data=None,
        audio_data=None,
        video_data=None,
        modalities=[],
        stop=[],
    )
    request = ResponsesRequest(
        model="x",
        input="synthetic",
        stream=stream,
        store=False,
        text={
            "format": {
                "type": "json_schema",
                "name": "synthetic",
                "strict": True,
                "schema": {"type": "number", "multipleOf": 2},
            }
        },
    )
    with patch.object(serving, "_apply_conversation_template", return_value=rendered):
        response = asyncio.run(serving.create_responses(request))
    assert response.status_code == 400
    serving.tokenizer_manager.generate_request.assert_not_called()


@pytest.mark.parametrize("event_count", [0, 1, 2])
@pytest.mark.parametrize("close_fault", [None, RuntimeError])
def test_responses_synthetic_event_close_releases_engine_immediately(
    event_count, close_fault
):
    from responses_cpu_helpers import engine_chunk, event_payloads, make_serving

    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
    from sglang.srt.entrypoints.openai.serving_chat import MessageProcessingResult

    serving = make_serving()
    rendered = MessageProcessingResult(
        prompt="synthetic",
        prompt_ids=[1, 2, 3],
        image_data=None,
        audio_data=None,
        video_data=None,
        modalities=[],
        stop=[],
    )
    closed = []

    async def generate(*args):
        try:
            yield engine_chunk("hello", finish=False)
            yield engine_chunk("hello world", finish=True)
        finally:
            closed.append(True)
            if close_fault:
                raise close_fault("Synthetic cleanup failure.")

    serving.tokenizer_manager.generate_request = generate

    async def run():
        result = await serving.create_responses(
            ResponsesRequest(model="x", input="synthetic", stream=True, store=False)
        )
        chunks = [await anext(result.body_iterator) for _ in range(event_count)]
        assert [p["type"] for p in event_payloads(chunks)] == [
            "response.created",
            "response.in_progress",
        ][:event_count]
        assert not closed
        await result.aclose()
        assert closed == [True]  # Before asyncio.run async-generator shutdown.
        await result.aclose()
        assert closed == [True]

    with patch.object(serving, "_apply_conversation_template", return_value=rendered):
        asyncio.run(run())


def test_prefetched_owner_close_without_consuming_first_item():
    serving = object.__new__(OpenAIServingResponses)

    class Engine:
        closed = 0

        async def __anext__(self):
            return {"meta_info": {}}

        async def aclose(self):
            self.closed += 1

    async def run():
        engine = Engine()
        owner = await serving._prefetch_response_generation(engine)
        await owner.aclose()
        assert engine.closed == 1
        await owner.aclose()
        assert engine.closed == 1

    asyncio.run(run())


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("code", KINDS)
def test_actual_nonstream_http_status_and_type_from_tokenizer(api, code):
    from responses_cpu_helpers import make_serving

    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
    from sglang.srt.entrypoints.openai.serving_chat import MessageProcessingResult

    serving = (
        make_serving()
        if api == "responses"
        else object.__new__(module.OpenAIServingChat)
    )

    async def generate(*args):
        state = SimpleNamespace(obj=SimpleNamespace(rid="synthetic"))
        manager = SimpleNamespace(
            rid_to_state={"synthetic": state},
            enable_lora=False,
            _release_lora_once=lambda state: None,
        )
        await TokenizerManager._handle_abort_finish_reason(
            manager, {"meta_info": {"finish_reason": abort(code)}}, state, False
        )
        yield None

    if api == "responses":
        serving.tokenizer_manager.generate_request = generate
        request = ResponsesRequest(
            model="x", input="synthetic", stream=False, store=False
        )
        rendered = MessageProcessingResult(
            prompt="synthetic",
            prompt_ids=[1, 2, 3],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )
        with patch.object(
            serving, "_apply_conversation_template", return_value=rendered
        ):
            response = asyncio.run(serving.create_responses(request))
        payload = json.loads(response.body)["error"]
    else:
        serving.chat_encoding_spec = None
        serving.tokenizer_manager = SimpleNamespace(
            generate_request=generate,
            request_logger=SimpleNamespace(log_requests=False),
        )
        serving._validate_request = lambda request: None
        serving._convert_to_internal_request = lambda request, raw: (
            SimpleNamespace(),
            request,
        )
        request = ChatCompletionRequest(
            model="synthetic", messages=[{"role": "user", "content": "synthetic"}]
        )
        response = asyncio.run(serving.handle_request(request, None))
        payload = json.loads(response.body)
    assert response.status_code == code
    assert payload["code"] == code and payload["type"] == KINDS[code]


@pytest.mark.parametrize("send_failure", [None, OSError, asyncio.CancelledError])
def test_responses_http_headers_exit_closes_prefetched_engine(send_failure):
    from responses_cpu_helpers import engine_chunk, make_serving
    from starlette.background import BackgroundTask
    from starlette.requests import ClientDisconnect

    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
    from sglang.srt.entrypoints.openai.serving_chat import MessageProcessingResult

    serving = make_serving()
    rendered = MessageProcessingResult(
        prompt="synthetic",
        prompt_ids=[1, 2, 3],
        image_data=None,
        audio_data=None,
        video_data=None,
        modalities=[],
        stop=[],
    )
    closed, backgrounds, messages = [], [], []

    async def generate(*args):
        try:
            yield engine_chunk("hello", finish=True)
        finally:
            closed.append(True)

    serving.tokenizer_manager.generate_request = generate

    async def run():
        result = await serving.create_responses(
            ResponsesRequest(model="x", input="synthetic", stream=True, store=False)
        )
        result.background = BackgroundTask(lambda: backgrounds.append(True))

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.start" and send_failure:
                raise send_failure("Synthetic send failure.")

        if send_failure:
            with pytest.raises(
                asyncio.CancelledError
                if send_failure is asyncio.CancelledError
                else ClientDisconnect
            ):
                await result(
                    {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
                )
            assert [m["type"] for m in messages] == ["http.response.start"]
        else:
            await result(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
            )
            assert backgrounds == [True]
            assert messages[0]["status"] == 200
            assert messages[-1]["more_body"] is False
        assert closed == [True]  # Assert before event-loop teardown.
        await result.aclose()
        assert closed == [True]

    with patch.object(serving, "_apply_conversation_template", return_value=rendered):
        asyncio.run(run())


def test_http_cleanup_failure_does_not_skip_explicit_owner():
    # The fork's main line owns stream cleanup in GenerationStreamingResponse.
    from sglang.srt.entrypoints.openai.serving_base import GenerationStreamingResponse

    closed = []

    async def body():
        try:
            yield "synthetic"
        finally:
            raise RuntimeError("Synthetic body cleanup failure.")

    class Owner:
        async def aclose(self):
            closed.append(True)

    async def run():
        response = GenerationStreamingResponse(body(), generation=Owner())
        await anext(response.body_iterator)
        with pytest.raises(RuntimeError, match="body cleanup failure"):
            await response.aclose()
        assert closed == [True]
        await response.aclose()
        assert closed == [True]

    asyncio.run(run())


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
