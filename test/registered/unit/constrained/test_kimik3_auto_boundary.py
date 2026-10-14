"""Generated strict-auto K3 grammar owns termination, including literal markers."""

import json
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import test_serving_chat as serving_tests
import torch
from jsonschema import Draft202012Validator
from test_kimik3_json_string import _text
from test_kimik3_reasoner_boundary import (
    _byte_tokens,
    _call,
    _decoded,
    _feed,
    _make,
    _mask,
    _only_allowed,
    _rejects,
    engine as _engine_fixture,
)

from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.constrained.reasoner_grammar_backend import (
    KIMIK3_REQUIRED_NATIVE_BOUNDARY,
    ReasonerGrammarBackend,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    MessageProcessingResult,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.parser.reasoning_parser import KimiK3Detector
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
engine = _engine_fixture


def _serving_parameters(schema, reasoning, parallel, explicit):
    case = serving_tests.ServingChatTestCase()
    case.setUp()
    try:
        chat = case.chat
        chat.chat_encoding_spec = "kimi_k3"
        chat.tool_call_parser = chat.reasoning_parser = "kimi_k3"
        chat._reasoning_detector = KimiK3Detector()
        chat.default_sampling_params = {}
        case.template_manager.chat_template_name = None
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "synthetic"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "fixture",
                        "strict": True,
                        "parameters": schema,
                    },
                }
            ],
            tool_choice="auto",
            parallel_tool_calls=parallel,
            max_tokens=4096,
            stop=explicit,
        )
        user_stops = [explicit] if isinstance(explicit, str) else explicit or []
        rendered_stops = list(user_stops)
        if MESSAGE_CLOSE not in rendered_stops:
            rendered_stops.append(MESSAGE_CLOSE)
        rendered = MessageProcessingResult(
            "synthetic", [1, 2, 3], None, None, None, [], rendered_stops
        )
        with (
            patch.object(chat, "_apply_jinja_template", return_value=rendered),
            patch.object(chat, "_get_reasoning_from_request", return_value=reasoning),
        ):
            internal, _ = chat._convert_to_internal_request(request)
        params = internal.sampling_params
        assert params["custom_params"][KIMIK3_REQUIRED_NATIVE_BOUNDARY] is True
        assert params["stop"] == user_stops
        return params, request.tools
    finally:
        case.doCleanups()


def _scheduler_request(engine, params, reasoning):
    tokenizer, backend = engine
    reset_context()
    publish(
        ServerArgs(model_path="dummy", reasoning_parser="kimi_k3"), role="scheduler"
    )
    sampling = SamplingParams(**params)
    sampling.normalize(tokenizer)
    req = Req(
        rid="strict-auto-marker",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sampling,
        eos_token_ids={tokenizer.model_eos},
        vocab_size=tokenizer.vocab_size,
    )
    req.tokenizer, req.require_reasoning = tokenizer, reasoning
    req.grammar = ReasonerGrammarBackend(
        backend, SimpleNamespace(detector=KimiK3Detector()), tokenizer
    )._init_value_dispatch(("structural_tag", params["structural_tag"]), reasoning)
    manager = object.__new__(GrammarManager)
    manager.scheduler = SimpleNamespace(
        tokenizer=tokenizer,
        model_config=SimpleNamespace(hf_eos_token_id={tokenizer.model_eos}),
    )
    manager._apply_request_reasoning_config(req)
    assert not req.finished()
    assert req.grammar._kimik3_token_bytes is not None
    return req


@pytest.mark.parametrize("marker", [TOOLS_CLOSE, MESSAGE_CLOSE])
@pytest.mark.parametrize("constraint", ["const", "enum"])
@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_strict_auto_scheduler_keeps_json_string_markers_until_eos(
    engine, marker, constraint, reasoning, parallel
):
    value = "prefix " + marker + " suffix"
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "string",
                constraint: value if constraint == "const" else [value],
            }
        },
        "required": ["payload"],
        "additionalProperties": False,
    }
    params, tools = _serving_parameters(schema, reasoning, parallel, None)
    try:
        req = _scheduler_request(engine, params, reasoning)
        text = ("reason" + THINK_CLOSE if reasoning else "") + _text(value)
        prefix = text[: text.index(marker) + len(marker)]
        ids = _feed(req.grammar, engine, prefix)
        req.output_ids = array("q", ids)
        req.update_finish_state(new_accepted_len=len(ids))
        assert not req.finished()
        assert not req.grammar.is_terminated()
        ids = _feed(req.grammar, engine, text[len(prefix) :])
        req.output_ids.extend(ids)
        req.update_finish_state(new_accepted_len=len(ids))
        assert not req.finished()
        _feed(req.grammar, engine, [engine[0].model_eos])
        req.output_ids.append(engine[0].model_eos)
        req.update_finish_state()
        assert req.finished() and req.finished_reason.matched == engine[0].model_eos
        calls = FunctionCallParser(
            tools, "kimi_k3", tool_choice="auto"
        ).parse_non_stream(text)[1]
        assert len(calls) == 1
        arguments = json.loads(calls[0].parameters)
        assert arguments == {"payload": value}
        Draft202012Validator(schema).validate(arguments)
    finally:
        reset_context()


@pytest.mark.parametrize("marker", [TOOLS_CLOSE, MESSAGE_CLOSE])
def test_strict_auto_scheduler_preserves_user_explicit_marker_stop(engine, marker):
    value = "prefix " + marker + " suffix"
    schema = {
        "type": "object",
        "properties": {"payload": {"type": "string", "const": value}},
    }
    params, _ = _serving_parameters(schema, False, True, marker)
    try:
        req = _scheduler_request(engine, params, False)
        text = _text(value)
        ids = _feed(req.grammar, engine, text[: text.index(marker) + len(marker)])
        req.output_ids = array("q", ids)
        req.update_finish_state(new_accepted_len=len(ids))
        assert req.finished() and req.finished_reason.matched == marker
        assert not req.grammar.is_terminated()
    finally:
        reset_context()


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_strict_auto_native_keeps_optional_answer_path(engine, reasoning, wrapped):
    obj = _make(engine, choice="auto", reasoning=reasoning, strict_thinking=True)
    answer = (
        RESPONSE_OPEN + "Done." + RESPONSE_CLOSE + MESSAGE_CLOSE if wrapped else "Done."
    )
    prefix = "reason" + THINK_CLOSE if reasoning else ""
    _feed(obj, engine, prefix + answer)
    assert obj._is_generation()
    _feed(obj, engine, [engine[0].model_eos])
    assert obj.is_terminated()


@pytest.mark.parametrize("parallel", [False, True])
def test_strict_auto_native_keeps_parallel_call_policy(engine, parallel):
    obj = _make(engine, choice="auto", parallel=parallel)
    _feed(obj, engine, TOOLS_OPEN + _call())
    if parallel:
        _feed(obj, engine, _call("other"))
    else:
        assert _rejects(obj.copy(), engine, _call("other"))
    _feed(obj, engine, TOOLS_CLOSE)
    _feed(obj, engine, [engine[0].eos_token_id])
    assert obj.is_terminated()


@pytest.mark.parametrize("spelling", ["canonical", "byte_alias"])
def test_strict_auto_response_copy_rollback_restores_boundary(engine, spelling):
    tokens = (
        engine[0].encode(RESPONSE_OPEN)
        if spelling == "canonical"
        else _byte_tokens(engine, RESPONSE_OPEN)
    )
    for count in sorted({0, 1, len(tokens) // 2, len(tokens) - 1}):
        obj = _make(engine, choice="auto", strict_thinking=spelling == "byte_alias")
        _feed(obj, engine, tokens[:count])
        clone = obj.copy()
        state = obj.tokens_in_think, obj.tokens_after_end, obj._kimik3_boundary_matches
        expected = _mask(obj, engine)
        assert torch.equal(_mask(clone, engine), expected)
        _feed(obj, engine, tokens[count:])
        assert obj._is_generation()
        obj.rollback(len(tokens) - count)
        assert (
            obj.tokens_in_think,
            obj.tokens_after_end,
            obj._kimik3_boundary_matches,
        ) == state
        assert torch.equal(_mask(obj, engine), expected)
        _feed(obj, engine, tokens[count:])
        _feed(clone, engine, tokens[count:])
        assert obj._is_generation() and clone._is_generation()
        assert torch.equal(_mask(obj, engine), _mask(clone, engine))


def test_strict_auto_response_budget_keeps_completable_boundary_and_policy(engine):
    strict = _make(engine, choice="auto", strict_thinking=True)
    # Preserve the existing strict-thinking policy for the canonical response
    # channel token: it requires an explicit think close while budget remains.
    assert _rejects(strict.copy(), engine, RESPONSE_OPEN)
    _feed(
        strict,
        engine,
        THINK_CLOSE + RESPONSE_OPEN + "Done." + RESPONSE_CLOSE + MESSAGE_CLOSE,
    )
    assert strict._is_generation()
    tokens = _byte_tokens(engine, RESPONSE_OPEN)
    for count in sorted({0, 1, len(tokens) // 2, len(tokens) - 1}):
        obj = _make(engine, choice="auto", strict_thinking=True)
        _feed(obj, engine, tokens[:count])
        clone = obj.copy()
        state = obj.tokens_in_think, obj.tokens_after_end, obj._kimik3_boundary_matches
        expected = _mask(obj, engine)
        obj.max_think_tokens = count
        for _ in range(30):
            if obj._is_generation():
                break
            allowed = _only_allowed(_mask(obj, engine), engine)
            assert len(allowed) == 1
            obj.accept_token(allowed[0])
        assert obj._is_generation()
        text = _decoded(engine, obj.grammar.accepted_tokens)
        if text.endswith(TOOLS_OPEN):
            _feed(obj, engine, _call() + TOOLS_CLOSE)
        else:
            assert text.endswith((THINK_CLOSE, RESPONSE_OPEN))
            _feed(
                obj,
                engine,
                (RESPONSE_OPEN if text.endswith(THINK_CLOSE) else "")
                + "Done."
                + RESPONSE_CLOSE
                + MESSAGE_CLOSE,
            )
        obj.rollback(len(obj.grammar.accepted_tokens) - count)
        obj.max_think_tokens = -1
        assert (
            obj.tokens_in_think,
            obj.tokens_after_end,
            obj._kimik3_boundary_matches,
        ) == state
        assert torch.equal(_mask(obj, engine), expected)
        _feed(obj, engine, tokens[count:])
        _feed(clone, engine, tokens[count:])
        assert obj._is_generation() and clone._is_generation()
        assert torch.equal(_mask(obj, engine), _mask(clone, engine))
