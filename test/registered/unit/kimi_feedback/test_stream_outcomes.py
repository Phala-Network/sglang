"""CPU-only synthetic generation: real chat stream outcome/cleanup path."""
import asyncio
import json
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from sglang.srt.entrypoints.openai import serving_chat as module
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

SECRET = "PRIVATE_GENERATOR_PAYLOAD"

class Generation:
    def __init__(self, finish=None, fault=None, close_fault=None, empty=False, indices=(0,)):
        self.finish, self.fault, self.close_fault = finish, fault, close_fault
        self.empty, self.indices, self.closed = empty, indices, 0
    def __aiter__(self):
        return self.run()
    async def run(self):
        if not self.empty:
            for index in self.indices:
                yield {"text": "synthetic", "index": index, "meta_info": {
                    "id": "synthetic", "prompt_tokens": 1, "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": (self.finish.get(index) if isinstance(self.finish, dict)
                        else {"type": self.finish} if self.finish else None)}}
        if self.fault:
            raise self.fault(SECRET)
    async def aclose(self):
        self.closed += 1
        if self.close_fault:
            raise self.close_fault(SECRET)


def run_case(generation, *, n=1, tool=False, close_early=False, encoding_spec=None):
    serving = object.__new__(module.OpenAIServingChat)
    serving.chat_encoding_spec = encoding_spec
    # Attributes __init__ sets on the fork's main line.
    serving.reasoning_parser = serving.tool_call_parser = None
    serving._encoding_adapter = None
    serving.tokenizer_manager = SimpleNamespace(generate_request=lambda *a: generation)
    serving._validate_kimi_k3_tool_call_completion = lambda *a: None
    async def content(**kwargs):
        if tool:
            kwargs["has_tool_calls"][kwargs["index"]] = True
        yield module.build_sse_content(chunk_id="synthetic", created=1,
            model="synthetic", index=kwargs["index"], content="synthetic")
    serving._generate_stream_content = content
    request = ChatCompletionRequest(model="synthetic", messages=[{"role":"user","content":"synthetic"}], stream=True, n=n)
    config = SimpleNamespace(stream_response_default_include_usage=False,
        return_input_ids=False, return_output_ids=False, enable_cache_report=False,
        incremental_streaming_output=False)
    async def collect():
        wire, failure = [], None
        stream = serving._generate_chat_stream(SimpleNamespace(), request, None)
        try:
            if close_early:
                wire.append(await anext(stream))
                await stream.aclose()
            else:
                async for chunk in stream:
                    wire.append(chunk)
        except BaseException as exc:
            failure = exc
        return "".join(wire), failure
    with patch.object(module, "get_serving", return_value=config):
        return asyncio.run(collect())


def events(wire):
    return [json.loads(line[6:]) for line in wire.splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"]

@pytest.mark.parametrize("case", ["runtime", "missing_finish", "empty", "close", "runtime_and_close", "missing_choice"])
def test_failure_is_explicit_without_success_marker(case):
    kwargs = {"runtime": {"fault": RuntimeError}, "missing_finish": {},
        "empty": {"empty": True}, "close": {"finish": "stop", "close_fault": RuntimeError},
        "runtime_and_close": {"fault": RuntimeError, "close_fault": RuntimeError},
        "missing_choice": {"finish": "stop"}}[case]
    generation = Generation(**kwargs)
    wire, failure = run_case(generation, n=2 if case == "missing_choice" else 1)
    assert failure is None
    assert SECRET not in wire
    assert wire.count("data: [DONE]") == 1
    assert sum("error" in event for event in events(wire)) == 1
    assert not [c for e in events(wire) for c in e.get("choices", []) if c.get("finish_reason")]
    assert generation.closed == 1

@pytest.mark.parametrize("finish,tool,expected", [("stop",False,"stop"),("length",False,"length"),("stop",True,"tool_calls")])
def test_normal_finish_once(finish, tool, expected):
    generation = Generation(finish=finish)
    wire, failure = run_case(generation, tool=tool)
    assert failure is None
    assert wire.count("data: [DONE]") == 1
    assert not any("error" in e for e in events(wire))
    assert [c["finish_reason"] for e in events(wire) for c in e.get("choices", []) if c.get("finish_reason")] == [expected]
    assert generation.closed == 1

@pytest.mark.parametrize("fault", [asyncio.CancelledError, GeneratorExit])
@pytest.mark.parametrize("close_fault", [None, RuntimeError, asyncio.CancelledError, GeneratorExit])
def test_cancellation_keeps_original_exception(fault, close_fault):
    generation = Generation(fault=fault, close_fault=close_fault)
    wire, failure = run_case(generation)
    assert isinstance(failure, fault)
    assert "[DONE]" not in wire
    assert not any("error" in e for e in events(wire))
    assert generation.closed == 1

@pytest.mark.parametrize("close_fault", [None, RuntimeError])
def test_consumer_close_cleans_up_without_yield(close_fault):
    generation = Generation(close_fault=close_fault)
    wire, failure = run_case(generation, close_early=True)
    assert failure is None
    assert "[DONE]" not in wire
    assert generation.closed == 1

@pytest.mark.parametrize("close_fault", [ValueError, RuntimeError])
def test_close_error_is_sanitized_before_success(close_fault):
    generation = Generation(finish="stop", close_fault=close_fault)
    wire, failure = run_case(generation)
    assert failure is None
    assert SECRET not in wire
    assert wire.count("data: [DONE]") == 1
    assert sum("error" in e for e in events(wire)) == 1
    assert not [c for e in events(wire) for c in e.get("choices", []) if c.get("finish_reason")]
    assert generation.closed == 1


@pytest.mark.parametrize("close_fault", [None, RuntimeError])
@pytest.mark.parametrize("indices", [(0, 1), (1, 0)])
def test_error_abort_ends_stream_without_late_success(indices, close_fault):
    generation = Generation(
        finish={indices[0]: {"type": "stop"}, indices[1]: {
            "type": "abort", "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
            "message": "Synthetic abort."}},
        indices=indices, close_fault=close_fault,
    )
    # The fork's main line scopes this contract to the K3 encoding.
    wire, failure = run_case(generation, n=2, encoding_spec="kimi_k3")
    assert failure is None
    parsed = events(wire)
    assert sum("error" in event for event in parsed) == 1
    assert "error" in parsed[-1], "successful metadata escaped after an error abort"
    assert not [c for e in parsed for c in e.get("choices", []) if c.get("finish_reason")]
    assert wire.count("data: [DONE]") == 1
    assert generation.closed == 1


def test_graceful_abort_keeps_existing_finish_contract():
    generation = Generation(finish="abort")
    wire, failure = run_case(generation)
    assert failure is None
    assert not any("error" in event for event in events(wire))
    assert [c["finish_reason"] for e in events(wire) for c in e.get("choices", [])
            if c.get("finish_reason")] == ["abort"]
    assert wire.count("data: [DONE]") == 1
    assert generation.closed == 1
