"""Native CPU masks, with optional pinned production tokenizer verification."""

import json
import os
import runpy
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import xgrammar as xgr
from jsonschema import Draft202012Validator
from transformers import AutoTokenizer

from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.constrained.reasoner_grammar_backend import (
    KIMIK3_REQUIRED_NATIVE_BOUNDARY,
    ReasonerGrammarBackend,
)
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    MESSAGE_CLOSE,
    THINK_CLOSE,
    THINK_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.srt.function_call.kimik3_structural_tag import get_kimik3_structural_tag
from sglang.srt.function_call.utils import strip_structural_tag_excludes
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.parser.reasoning_parser import KimiK3Detector
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_utils import traverse_tree
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Tokenizer:
    def __init__(self):
        self.vocab = [bytes([i]) for i in range(256)] + [
            token.encode()
            for token in (
                "<|open|>",
                "<|close|>",
                "<|sep|>",
                "[EOS]",
                "<|end_of_msg|>",
                "[EOT]",
                "think",
                "tools",
                "message",
                "response",
            )
        ]
        self.vocab_size = len(self.vocab)
        self.eos_token_id = 259
        self.model_eos = 260
        self.additional_stop_token_ids = None
        self.model = SimpleNamespace(decode_single_token_bytes=lambda i: self.vocab[i])

    def encode(self, text, add_special_tokens=False):
        data = text.encode()
        tokens = []
        while data:
            selected = next(
                (
                    i
                    for i in range(256, len(self.vocab))
                    if data.startswith(self.vocab[i])
                ),
                data[0],
            )
            tokens.append(selected)
            data = data[len(self.vocab[selected]) :]
        return tokens

    def init_xgrammar(self):
        return xgr.TokenizerInfo(self.vocab, stop_token_ids=[self.model_eos]), None

    def decode(self, token_ids, **kwargs):
        return b"".join(self.vocab[token] for token in token_ids).decode()


@pytest.fixture(scope="module", params=["synthetic", "production"])
def engine(request):
    if request.param == "production":
        path = os.environ.get("KIMIK3_TOKENIZER_DIR")
        if not path:
            pytest.skip("Set KIMIK3_TOKENIZER_DIR for pinned tokenizer CPU acceptance")
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer.additional_stop_token_ids = None
        tokenizer.model_eos = json.loads((Path(path) / "config.json").read_text())[
            "eos_token_id"
        ]
    else:
        tokenizer = _Tokenizer()
    backend = XGrammarGrammarBackend(
        tokenizer, tokenizer.vocab_size, model_eos_token_ids=[tokenizer.model_eos]
    )
    return tokenizer, backend


def _tag(strict=True, choice="required", parallel=True):
    tools = [
        Tool(
            type="function",
            function=Function(
                name=name,
                strict=strict,
                parameters={
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "minimum": 1, "maximum": 3},
                        "note": {"type": "string"},
                    },
                    "required": ["x"],
                    "additionalProperties": False,
                },
            ),
        )
        for name in ("fixture", "other")
    ]
    if choice == "named":
        choice = ToolChoice(
            type="function", function=ToolChoiceFuncName(name="fixture")
        )
    tag = get_kimik3_structural_tag(tools, choice, parallel_tool_calls=parallel)
    strip_structural_tag_excludes(tag, (THINK_OPEN, THINK_CLOSE))
    return tag


def _make(
    engine,
    strict=True,
    choice="required",
    reasoning=True,
    strict_thinking=False,
    budget=-1,
    native=True,
    parallel=True,
):
    tokenizer, backend = engine
    reasoner = ReasonerGrammarBackend(
        backend,
        SimpleNamespace(detector=KimiK3Detector()),
        tokenizer,
        enable_strict_thinking=strict_thinking,
    )
    key = ("structural_tag", _tag(strict, choice, parallel).model_dump_json())
    obj = reasoner._init_value_dispatch(key, reasoning)
    obj.max_think_tokens = budget
    if native:
        obj.enable_kimik3_required_native_boundary(
            tokenizer, [tokenizer.model_eos, tokenizer.eos_token_id]
        )
    return obj


def _mask(obj, engine):
    mask = obj.allocate_vocab_mask(engine[0].vocab_size, 1, "cpu")
    mask.fill_(-1)
    obj.fill_vocab_mask(mask, 0)
    return mask


def _allowed(mask, token):
    return bool(int(mask[0, token // 32]) & (1 << (token % 32)))


def _feed(obj, engine, value):
    tokens = (
        engine[0].encode(value, add_special_tokens=False)
        if isinstance(value, str)
        else value
    )
    for token in tokens:
        assert _allowed(_mask(obj, engine), token), (token, obj.tokens_after_end)
        obj.accept_token(token)
    return tokens


def _rejects(obj, engine, value):
    for token in (
        engine[0].encode(value, add_special_tokens=False)
        if isinstance(value, str)
        else value
    ):
        if not _allowed(_mask(obj, engine), token):
            return True
        obj.accept_token(token)
    return False


def _alias(engine, control, name):
    t = engine[0]
    return (
        t.encode(control)
        + [token for char in name for token in t.encode(char)]
        + t.encode("<|sep|>")
    )


def _decoded(engine, tokens):
    return b"".join(
        engine[0].model.decode_single_token_bytes(token) for token in tokens
    ).decode()


def _byte_tokens(engine, text):
    return [token for char in text for token in engine[0].encode(char)]


def _only_allowed(mask, engine):
    bit_positions = torch.arange(32)
    bits = (mask[0, :, None].bitwise_right_shift(bit_positions) & 1).flatten()
    return torch.where(bits[: engine[0].vocab_size])[0].tolist()


def _call(name="fixture", note=False):
    extra = (
        (
            '<|open|>argument key="note" type="string"<|sep|>message'
            "<|close|>argument<|sep|>"
        )
        if note
        else ""
    )
    return (
        f'<|open|>call tool="{name}" index="1"<|sep|>'
        '<|open|>argument key="x" type="number"<|sep|>1'
        "<|close|>argument<|sep|>" + extra + CALL_CLOSE
    )


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("choice", ["required", "named"])
@pytest.mark.parametrize("reasoning", [False, True])
def test_required_masks_stop_and_both_eos_from_first_token(
    engine, strict, choice, reasoning
):
    obj = _make(engine, strict, choice, reasoning)
    _feed(obj, engine, "reason")
    for marker in (TOOLS_CLOSE, MESSAGE_CLOSE):
        assert _rejects(obj.copy(), engine, marker)
    for eos in (engine[0].model_eos, engine[0].eos_token_id):
        assert not _allowed(_mask(obj, engine), eos)
    if reasoning:
        _feed(obj, engine, THINK_CLOSE)
        assert obj._is_generation()
    _feed(obj, engine, TOOLS_OPEN + _call() + TOOLS_CLOSE)
    assert not obj.is_terminated()
    assert set(_only_allowed(_mask(obj, engine), engine)) == {
        engine[0].model_eos,
        engine[0].eos_token_id,
    }
    for eos in (engine[0].model_eos, engine[0].eos_token_id):
        copy = obj.copy()
        _feed(copy, engine, [eos])
        assert copy.is_terminated()


def test_alias_markers_cannot_stop_and_tools_alias_ends_thinking(engine):
    obj = _make(engine, strict_thinking=True)
    for name in ("tools", "message"):
        assert _rejects(obj.copy(), engine, _alias(engine, "<|close|>", name))
    tokens = _alias(engine, "<|open|>", "tools")
    assert len(tokens) > len(engine[0].encode(TOOLS_OPEN))
    _feed(obj, engine, tokens)
    assert obj._is_generation()
    _feed(obj, engine, _call(note=True) + TOOLS_CLOSE)
    think = _make(engine, strict_thinking=True)
    _feed(think, engine, _alias(engine, "<|close|>", "think"))
    assert think._is_generation()


@pytest.mark.parametrize("marker", [THINK_CLOSE, TOOLS_OPEN])
def test_rollback_and_native_copy_restore_every_marker_prefix(engine, marker):
    tokens = engine[0].encode("reason" + marker)
    for count in range(len(tokens) + 1):
        obj = _make(engine)
        _feed(obj, engine, tokens[:count])
        copy = obj.copy()
        assert torch.equal(_mask(copy, engine), _mask(obj, engine))
        assert copy.grammar.accepted_tokens == obj.grammar.accepted_tokens
        assert copy.grammar.accepted_tokens is not obj.grammar.accepted_tokens
        if count:
            obj.rollback(count)
        obj.rollback(0)
        assert copy.grammar.accepted_tokens == tokens[:count]
        assert obj.current_token is None
        assert torch.equal(_mask(obj, engine), _mask(_make(engine), engine))
        _feed(obj, engine, tokens)
        _feed(
            obj,
            engine,
            (TOOLS_OPEN if marker == THINK_CLOSE else "") + _call() + TOOLS_CLOSE,
        )


@pytest.mark.parametrize("marker", [THINK_CLOSE, TOOLS_OPEN])
@pytest.mark.parametrize("spelling", ["canonical", "channel_alias", "byte_alias"])
def test_strict_budget_intersects_native_mask_at_every_boundary_prefix(
    engine, marker, spelling
):
    if spelling == "canonical":
        boundary = engine[0].encode(marker)
    elif spelling == "channel_alias":
        boundary = _alias(
            engine,
            "<|close|>" if marker == THINK_CLOSE else "<|open|>",
            "think" if marker == THINK_CLOSE else "tools",
        )
    else:
        boundary = _byte_tokens(engine, marker)
    for prefix_count in range(len(boundary)):
        obj = _make(engine, strict_thinking=True)
        prefix = boundary[:prefix_count]
        _feed(obj, engine, prefix)
        obj.max_think_tokens = len(prefix)
        for _ in range(30):
            if obj._is_generation():
                break
            mask = _mask(obj, engine)
            native = obj.grammar.allocate_vocab_mask(engine[0].vocab_size, 1, "cpu")
            native.fill_(-1)
            obj.grammar.fill_vocab_mask(native, 0)
            assert torch.equal(mask.bitwise_and(native), mask)
            allowed = _only_allowed(mask, engine)
            assert len(allowed) == 1, (
                marker,
                spelling,
                prefix_count,
                "budget must preserve a completable native boundary",
            )
            obj.accept_token(allowed[0])
        assert obj._is_generation()
        text = _decoded(engine, obj.grammar.accepted_tokens)
        assert text.endswith((THINK_CLOSE, TOOLS_OPEN))
        if text.endswith(THINK_CLOSE):
            _feed(obj, engine, TOOLS_OPEN)
        else:
            assert marker == TOOLS_OPEN and prefix_count > 0
            assert text == TOOLS_OPEN
        _feed(obj, engine, _call() + TOOLS_CLOSE)


def test_strict_off_budget_cannot_disable_native_safety(engine):
    for budget in (0, 1, -1):
        obj = _make(engine, budget=budget)
        assert not obj.enable_token_filter
        _feed(obj, engine, "ordinary thinking text")
        assert obj._is_thinking()
        assert _rejects(obj.copy(), engine, TOOLS_CLOSE)
        for eos in (engine[0].eos_token_id, engine[0].model_eos):
            assert not _allowed(_mask(obj, engine), eos)
        _feed(obj, engine, TOOLS_OPEN + _call() + TOOLS_CLOSE)


@pytest.mark.parametrize("marker", [THINK_CLOSE, TOOLS_OPEN])
def test_spec_dfs_restores_alias_boundary_and_rejected_siblings(engine, marker):
    obj = _make(engine, strict_thinking=True)
    tokens = _byte_tokens(engine, marker)
    _feed(obj, engine, tokens[:-1])
    prefix = obj.grammar.accepted_tokens[:]
    state = (obj.tokens_in_think, obj.tokens_after_end, obj._kimik3_boundary_matches)
    oracle = _mask(obj, engine)
    # EOS is rejected; complete the boundary, visit a legal child, then test
    # another legal sibling from the same original thinking prefix.
    child = engine[0].encode(TOOLS_OPEN if marker == THINK_CLOSE else _call())[0]
    drafts = torch.tensor(
        [prefix[-1], engine[0].model_eos, tokens[-1], child, tokens[-1]]
    )
    next_token = torch.tensor([1, -1, 3, -1, -1], dtype=torch.int32)
    next_sibling = torch.tensor([-1, 2, 4, -1, -1], dtype=torch.int32)
    masks = obj.allocate_vocab_mask(engine[0].vocab_size, len(drafts), "cpu")
    masks.zero_()
    traverse_tree(next_token, next_sibling, drafts, obj, masks, engine[0].vocab_size)
    assert not bool(torch.any(masks[1]))
    assert torch.equal(masks[2], masks[4])
    assert torch.equal(_mask(obj, engine), oracle)
    assert obj.grammar.accepted_tokens == prefix
    assert (
        obj.tokens_in_think,
        obj.tokens_after_end,
        obj._kimik3_boundary_matches,
    ) == state
    assert obj.current_token == prefix[-1]


def test_spec_dfs_eos_siblings_restore_complete_tools_prefix(engine):
    obj = _make(engine)
    _feed(obj, engine, TOOLS_OPEN + _call() + TOOLS_CLOSE)
    prefix = obj.grammar.accepted_tokens[:]
    expected = _mask(obj, engine)
    drafts = torch.tensor([prefix[-1], engine[0].model_eos, engine[0].eos_token_id])
    masks = obj.allocate_vocab_mask(engine[0].vocab_size, len(drafts), "cpu")
    masks.zero_()
    traverse_tree(
        torch.tensor([1, -1, -1], dtype=torch.int32),
        torch.tensor([-1, 2, -1], dtype=torch.int32),
        drafts,
        obj,
        masks,
        engine[0].vocab_size,
    )
    assert not bool(torch.any(masks[1:]))
    assert obj.grammar.accepted_tokens == prefix
    assert not obj.is_terminated()
    assert torch.equal(_mask(obj, engine), expected)
    assert obj.current_token == prefix[-1]


def test_native_jump_retokenization_and_invalid_rollback_preserve_state(engine):
    obj = _make(engine, strict_thinking=True)
    old = _feed(obj, engine, TOOLS_OPEN + _call() + TOOLS_CLOSE)
    # Identical IDs must not call the backend's rollback(0), which used to
    # erase accepted-token history via [:-0].
    expected = _mask(obj, engine)
    obj.jump_and_retokenize(old, old, -1)
    assert obj.grammar.accepted_tokens == old
    assert torch.equal(_mask(obj, engine), expected)
    for bad_count in (-1, len(old) + 1):
        with pytest.raises(ValueError):
            obj.rollback(bad_count)
        assert obj.grammar.accepted_tokens == old
        assert torch.equal(_mask(obj, engine), expected)
    new = _alias(engine, "<|close|>", "think") + engine[0].encode(
        TOOLS_OPEN + _call() + TOOLS_CLOSE
    )
    obj.jump_and_retokenize(old, new, -1)
    oracle = _make(engine, strict_thinking=True)
    _feed(oracle, engine, new)
    assert obj.grammar.accepted_tokens == new
    assert torch.equal(_mask(obj, engine), _mask(oracle, engine))
    obj.rollback(len(new))
    assert obj._is_thinking() and obj.current_token is None
    assert torch.equal(
        _mask(obj, engine), _mask(_make(engine, strict_thinking=True), engine)
    )


def test_native_accept_failure_does_not_commit_wrapper_or_inner_state(engine):
    obj = _make(engine)
    _feed(obj, engine, "reason")
    prefix = obj.grammar.accepted_tokens[:]
    expected = _mask(obj, engine)
    with pytest.raises(ValueError):
        obj.accept_token(engine[0].eos_token_id)
    assert obj.current_token == obj.grammar.current_token == prefix[-1]
    assert obj.grammar.accepted_tokens == prefix
    assert torch.equal(_mask(obj, engine), expected)


def test_xgrammar_zero_and_invalid_rollback_do_not_touch_native_state(
    engine, monkeypatch
):
    obj = _make(engine)
    _feed(obj, engine, "ordinary reasoning history")
    inner = obj.grammar
    history = inner.accepted_tokens
    expected = history[:]
    expected_mask = _mask(obj, engine)
    rollback = MagicMock(wraps=inner.matcher.rollback)
    with monkeypatch.context() as patch:
        patch.setattr(inner, "matcher", SimpleNamespace(rollback=rollback))
        inner.rollback(0)
        for invalid in (-1, len(history) + 1, True, 1.0, None):
            with pytest.raises(ValueError):
                inner.rollback(invalid)
        rollback.assert_not_called()
    assert inner.accepted_tokens is history
    assert history == expected
    assert torch.equal(_mask(obj, engine), expected_mask)


def test_xgrammar_native_rollback_failure_preserves_history(engine, monkeypatch):
    obj = _make(engine)
    _feed(obj, engine, "ordinary reasoning history")
    inner = obj.grammar
    history = inner.accepted_tokens
    expected = history[:]
    expected_mask = _mask(obj, engine)
    state = obj.tokens_in_think, obj.tokens_after_end, obj.current_token
    rollback = MagicMock(side_effect=RuntimeError("native rollback rejected"))
    with monkeypatch.context() as patch:
        patch.setattr(inner, "matcher", SimpleNamespace(rollback=rollback))
        with pytest.raises(RuntimeError, match="native rollback rejected"):
            obj.rollback(1)
        rollback.assert_called_once_with(1)
    assert inner.accepted_tokens is history
    assert history == expected
    assert (obj.tokens_in_think, obj.tokens_after_end, obj.current_token) == state
    assert torch.equal(_mask(obj, engine), expected_mask)


def test_xgrammar_tail_rollback_preserves_native_copy_isolation(engine):
    obj = _make(engine)
    tokens = _feed(obj, engine, "ordinary reasoning history")
    copied = obj.copy()
    history = obj.grammar.accepted_tokens
    obj.rollback(1)
    assert obj.grammar.accepted_tokens is history
    assert history == tokens[:-1]
    assert copied.grammar.accepted_tokens == tokens
    assert copied.grammar.accepted_tokens is not history
    obj.accept_token(tokens[-1])
    assert torch.equal(_mask(copied, engine), _mask(obj, engine))


def test_named_schema_parallel_and_auto_controls(engine):
    named = _make(engine, choice="named")
    _feed(named, engine, TOOLS_OPEN)
    assert _rejects(named.copy(), engine, _call("other"))
    assert _rejects(
        named.copy(), engine, _call().replace('type="number"', 'type="string"')
    )
    _feed(named, engine, _call() + _call() + TOOLS_CLOSE)
    single = _make(engine, parallel=False)
    _feed(single, engine, TOOLS_OPEN + _call())
    assert _rejects(single, engine, _call())
    auto = _make(engine, choice="auto", native=False)
    assert auto._kimik3_token_bytes is None
    _feed(auto, engine, TOOLS_CLOSE)
    assert auto.grammar.accepted_tokens == []


@pytest.mark.parametrize("marker", [TOOLS_CLOSE, CALL_CLOSE, MESSAGE_CLOSE])
def test_native_argument_literal_markers_are_data_until_complete_tools_section(
    engine, marker
):
    obj = _make(engine, strict_thinking=True)
    _feed(
        obj,
        engine,
        TOOLS_OPEN + _call(note=True).replace("<|sep|>message", "<|sep|>" + marker),
    )
    assert not obj.is_terminated()
    assert not _allowed(_mask(obj, engine), engine[0].model_eos)
    _feed(obj, engine, TOOLS_CLOSE)
    assert set(_only_allowed(_mask(obj, engine), engine)) == {
        engine[0].model_eos,
        engine[0].eos_token_id,
    }
    _feed(obj, engine, [engine[0].model_eos])
    assert obj.is_terminated()
    accepted = obj.grammar.accepted_tokens[:]
    state = obj.tokens_after_end, obj.current_token
    obj.accept_token(engine[0].eos_token_id)
    assert obj.grammar.accepted_tokens == accepted
    assert (obj.tokens_after_end, obj.current_token) == state


def test_scheduler_finishes_native_tools_on_eos_after_literal_marker_arguments(engine):
    reset_context()
    publish(
        ServerArgs(model_path="dummy", reasoning_parser="kimi_k3"), role="scheduler"
    )
    try:
        obj = _make(engine)
        tokenizer = engine[0]
        params = SamplingParams(max_new_tokens=4096, stop=[])
        params.normalize(tokenizer)
        req = Req(
            rid="native-k3",
            origin_input_text="",
            origin_input_ids=array("q", [0]),
            sampling_params=params,
            eos_token_ids={tokenizer.model_eos},
            vocab_size=tokenizer.vocab_size,
        )
        req.tokenizer, req.grammar = tokenizer, obj
        text = TOOLS_OPEN + _call(note=True).replace(
            "<|sep|>message", "<|sep|>" + TOOLS_CLOSE + MESSAGE_CLOSE
        )
        ids = _feed(obj, engine, text)
        req.output_ids = array("q", ids)
        req.update_finish_state(new_accepted_len=len(ids))
        assert not req.finished()
        ids = _feed(obj, engine, TOOLS_CLOSE)
        req.output_ids.extend(ids)
        req.update_finish_state(new_accepted_len=len(ids))
        assert not req.finished()
        _feed(obj, engine, [tokenizer.model_eos])
        req.output_ids.append(tokenizer.model_eos)
        req.update_finish_state()
        assert req.finished() and req.finished_reason.matched == tokenizer.model_eos
        assert req.finished_len == len(req.output_ids)
    finally:
        reset_context()


@pytest.mark.parametrize("choice", ["required", "named"])
@pytest.mark.parametrize("constraint", ["const", "enum"])
@pytest.mark.parametrize("stream", [False, True])
def test_shared_live_marker_schema_native_masks_and_parser_roundtrip(
    engine, choice, constraint, stream
):
    fixture = os.environ.get("KIMIK3_MARKER_CASES_FILE")
    if not fixture:
        pytest.skip("Set KIMIK3_MARKER_CASES_FILE for shared model-matrix fixture")
    shared = runpy.run_path(fixture)
    expected = shared["MARKER_ARGUMENTS"]
    schema = shared["marker_schema"](constraint)
    tools = [
        Tool(
            type="function",
            function=Function(
                name="final_result",
                strict=True,
                parameters=schema,
            ),
        )
    ]
    tool_choice = (
        ToolChoice(type="function", function=ToolChoiceFuncName(name="final_result"))
        if choice == "named"
        else "required"
    )
    tag = get_kimik3_structural_tag(tools, tool_choice, parallel_tool_calls=False)
    strip_structural_tag_excludes(tag, (THINK_OPEN, THINK_CLOSE))
    reasoner = ReasonerGrammarBackend(
        engine[1],
        SimpleNamespace(detector=KimiK3Detector()),
        engine[0],
    )
    obj = reasoner._init_value_dispatch(("structural_tag", tag.model_dump_json()), True)
    obj.enable_kimik3_required_native_boundary(
        engine[0], [engine[0].model_eos, engine[0].eos_token_id]
    )
    text = (
        TOOLS_OPEN + '<|open|>call tool="final_result" index="1"<|sep|>'
        '<|open|>argument key="label" type="json-string"<|sep|>'
        + json.dumps(expected["label"], ensure_ascii=False)
        + ARGUMENT_CLOSE
        + '<|open|>argument key="nested" type="object"<|sep|>'
        + json.dumps(expected["nested"], separators=(",", ":"))
        + ARGUMENT_CLOSE
        + CALL_CLOSE
        + TOOLS_CLOSE
    )
    unsafe_raw = text.replace(
        'type="json-string"<|sep|>' + json.dumps(expected["label"], ensure_ascii=False),
        'type="string"<|sep|>' + expected["label"],
        1,
    )
    assert unsafe_raw != text and _rejects(obj.copy(), engine, unsafe_raw)
    _feed(obj, engine, text)
    assert set(_only_allowed(_mask(obj, engine), engine)) == {
        engine[0].model_eos,
        engine[0].eos_token_id,
    }
    _feed(obj, engine, [engine[0].model_eos])
    assert obj.is_terminated()
    parser = FunctionCallParser(tools, "kimi_k3", tool_choice=tool_choice)
    if stream:
        calls = []
        for char in text:
            normal, items = parser.parse_stream_chunk(char)
            assert normal == ""
            calls.extend(items)
        normal, items = parser.parse_stream_end()
        assert normal == ""
        calls.extend(items)
    else:
        normal, calls = parser.parse_non_stream(text)
        assert normal == ""
    assert len(calls) == 1
    assert (
        calls[0].name == "final_result" and json.loads(calls[0].parameters) == expected
    )
    Draft202012Validator(schema).validate(json.loads(calls[0].parameters))
    if (
        choice == "named"
        and constraint == "enum"
        and hasattr(engine[0], "apply_chat_template")
    ):
        from sglang.srt.entrypoints.openai.encoding_k3 import apply_chat_template

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "opaque_fixture",
                        "type": "function",
                        "function": {
                            "name": calls[0].name,
                            "arguments": calls[0].parameters,
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "opaque_fixture", "content": "stored"},
            {"role": "user", "content": "continue"},
        ]
        rendered = apply_chat_template(
            engine[0], messages, tokenize=False, thinking=False
        )
        restored = FunctionCallParser(
            tools, "kimi_k3", tool_choice=tool_choice
        ).parse_non_stream(rendered[rendered.index(TOOLS_OPEN) :])[1]
        assert len(restored) == 1 and json.loads(restored[0].parameters) == expected


def test_manager_cache_keeps_native_state_eos_and_budget_per_request(engine):
    reset_context()
    publish(
        ServerArgs(model_path="dummy", reasoning_parser="kimi_k3"), role="scheduler"
    )
    try:
        tokenizer, backend = engine
        reasoner = ReasonerGrammarBackend(
            backend,
            SimpleNamespace(detector=KimiK3Detector()),
            tokenizer,
            enable_strict_thinking=True,
        )
        key = ("structural_tag", _tag().model_dump_json())
        mgr = object.__new__(GrammarManager)
        proxy = SimpleNamespace(
            model=tokenizer.model,
            encode=tokenizer.encode,
            eos_token_id=tokenizer.eos_token_id,
        )
        mgr.scheduler = SimpleNamespace(
            tokenizer=proxy,
            model_config=SimpleNamespace(
                hf_eos_token_id={tokenizer.model_eos},
                request_selectable_think_end_id_sequences=None,
            ),
        )
        mgr.grammar_backend = reasoner
        mgr._enable_strict_thinking = True
        mgr.grammar_queue = []
        mgr.grammar_sync_size = 1
        mgr.pp_rank, mgr.pp_size, mgr.pp_group = 0, 1, None
        mgr.SGLANG_GRAMMAR_POLL_INTERVAL = 0.05
        mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 2
        previous = None
        for index, (budget, reasoning) in enumerate(((0, True), (5, False), (9, True))):
            if index == 1:
                # Model config normally uses a set; a single-ID config must
                # behave identically. This request alone also has an extra EOS.
                mgr.scheduler.model_config.hf_eos_token_id = tokenizer.model_eos
                proxy.additional_stop_token_ids = tokenizer.encode("[EOT]")
            else:
                mgr.scheduler.model_config.hf_eos_token_id = [tokenizer.model_eos]
                proxy.additional_stop_token_ids = None
            req = MagicMock()
            req.sampling_params = SimpleNamespace(
                json_schema=None,
                regex=None,
                ebnf=None,
                structural_tag=key[1],
                custom_params={
                    KIMIK3_REQUIRED_NATIVE_BOUNDARY: True,
                    "thinking_budget": budget,
                },
            )
            req.require_reasoning = reasoning
            req.grammar_wait_ct = 0
            req.finished.return_value = False
            queued = mgr.process_req_with_grammar(req)
            assert queued is (index == 0)
            if queued:
                req.grammar.result(timeout=30)
                assert mgr.get_ready_grammar_requests() == [req]
                assert not mgr.grammar_queue
            req.set_finish_with_abort.assert_not_called()
            current = req.grammar
            assert current._kimik3_token_bytes is not None
            assert current.max_think_tokens == budget
            assert current._is_thinking() is reasoning
            expected_stops = {tokenizer.eos_token_id, tokenizer.model_eos} | set(
                proxy.additional_stop_token_ids or []
            )
            assert set(current.grammar.override_stop_tokens) == expected_stops
            assert not _allowed(_mask(current, engine), tokenizer.eos_token_id)
            assert reasoner.cache[key]._kimik3_token_bytes is None
            assert reasoner.cache[key].grammar.override_stop_tokens is None
            assert reasoner.cache[key].max_think_tokens != budget
            assert reasoner.cache[key].grammar.accepted_tokens == []
            if previous is not None:
                assert current is not previous
                assert current.grammar.matcher is not previous.grammar.matcher
                assert current.grammar.accepted_tokens == []
            # A completed request or partial tool opener cannot leak into the
            # next cache hit, even when its thinking mode and budget differ.
            if budget == 0:
                while current._is_thinking():
                    current.accept_token(
                        _only_allowed(_mask(current, engine), engine)[0]
                    )
            _feed(current, engine, TOOLS_OPEN + _call() + TOOLS_CLOSE)
            previous = current
    finally:
        reset_context()


def test_manager_native_boundary_fails_closed_on_unsupported_backend(engine):
    reset_context()
    publish(
        ServerArgs(model_path="dummy", reasoning_parser="kimi_k3"), role="scheduler"
    )
    try:
        mgr = object.__new__(GrammarManager)
        mgr.scheduler = SimpleNamespace(
            tokenizer=engine[0],
            model_config=SimpleNamespace(
                hf_eos_token_id={engine[0].model_eos},
            ),
        )
        req = MagicMock()
        req.grammar = object()
        req.sampling_params = SimpleNamespace(
            structural_tag="{}",
            custom_params={KIMIK3_REQUIRED_NATIVE_BOUNDARY: True},
        )
        mgr._apply_request_reasoning_config(req)
        assert (
            "Unable to configure K3 native boundary"
            in req.set_finish_with_abort.call_args.args[0]
        )
    finally:
        reset_context()
