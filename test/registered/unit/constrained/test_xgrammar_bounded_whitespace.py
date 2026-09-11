"""Bound grammar whitespace without breaking compact JSON or string contents."""

import json

import pytest
from xgrammar import TokenizerInfo, VocabType, allocate_token_bitmask

from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EOS = 256
COMPACT_DELIMITER = 257
VOCAB = [bytes([i]) for i in range(256)] + [b"<eos>", b'":"']


class ByteTokenizer:
    def init_xgrammar(self):
        return TokenizerInfo(VOCAB, VocabType.RAW, stop_token_ids=[EOS]), [EOS]


def backend(limit=None, any_whitespace=True):
    return XGrammarGrammarBackend(
        ByteTokenizer(), vocab_size=len(VOCAB), any_whitespace=any_whitespace,
        max_whitespace_cnt=limit,
    )


def accepts(schema, text, limit=64):
    grammar = backend(limit).dispatch_json(json.dumps(schema))
    return grammar.matcher.accept_string(text) and grammar.matcher.accept_token(EOS)


SCHEMA = {
    "type": "object", "properties": {"count": {"type": "integer"}, "ok": {"type": "boolean"}},
    "required": ["count", "ok"], "additionalProperties": False,
}


@pytest.mark.parametrize("schema", [{"type": "object"}, SCHEMA])
@pytest.mark.parametrize("whitespace", [" ", "\t", "\n"])
@pytest.mark.parametrize("size", [0, 1, 63, 64, 65, 256])
def test_whitespace_boundary(schema, whitespace, size):
    text = '{"count":' + whitespace * size + '1,"ok":true}'
    assert accepts(schema, text) == (size <= 64)


@pytest.mark.parametrize("limit", [1, 8, 64])
def test_compact_nested_arrays_and_unrestricted_string_whitespace(limit):
    value = {"s": " " * 300 + '\t\n中文"\\', "nested": [{"a": [1, None, True]}]}
    assert accepts({"type": "object"}, json.dumps(value, separators=(",", ":")), limit)


def test_default_is_unchanged_and_large_whitespace_remains_allowed():
    assert accepts(SCHEMA, '{"count":' + ' ' * 256 + '1,"ok":true}', None)


@pytest.mark.parametrize("schema", [{"type": "object"}, SCHEMA])
def test_carriage_return_matches_existing_xgrammar_boundary(schema):
    # XGrammar 0.2.1 already rejects a bare CR. Keep that limitation visible
    # instead of reporting it as a regression caused by the new bound.
    text = '{"count":\r1,"ok":true}'
    assert not accepts(schema, text, None)
    assert not accepts(schema, text, 64)


@pytest.mark.parametrize("value", [1, None, True, "s", [], {"a": [1, 2]}])
def test_builtin_json_preserves_unrestricted_root_types(value):
    grammar = backend(64).dispatch_json("$$ANY$$")
    assert grammar.matcher.accept_string(json.dumps(value, separators=(",", ":")))
    assert grammar.matcher.accept_token(EOS)


def test_builtin_json_bound_is_not_silently_ignored():
    text = '{"count":' + ' ' * 65 + '1}'
    assert backend(None).dispatch_json("$$ANY$$").matcher.accept_string(text)
    assert not backend(64).dispatch_json("$$ANY$$").matcher.accept_string(text)


@pytest.mark.parametrize("value", [
    {"count": "wrong", "ok": True}, {"count": 1},
    {"count": 1, "ok": True, "extra": False}, {"count": 1, "ok": "yes"},
])
def test_strict_schema_constraints_preserved(value):
    assert not accepts(SCHEMA, json.dumps(value))


def test_speculative_rollback_and_copy_isolation():
    grammar = backend(64).dispatch_json('{"type":"object"}')
    copied = grammar.copy()
    for token in b'{"location':
        grammar.accept_token(token)
    before = allocate_token_bitmask(1, len(VOCAB))
    grammar.fill_vocab_mask(before, 0)
    assert (int(before[0, COMPACT_DELIMITER // 32]) >> (COMPACT_DELIMITER % 32)) & 1
    grammar.accept_token(COMPACT_DELIMITER)
    grammar.accept_token(ord("B"))
    grammar.rollback(2)
    after = allocate_token_bitmask(1, len(VOCAB))
    grammar.fill_vocab_mask(after, 0)
    assert before.equal(after)
    grammar.accept_token(COMPACT_DELIMITER)
    for token in b'Boston"}':
        grammar.accept_token(token)
    grammar.accept_token(EOS)
    assert grammar.is_terminated()
    assert copied.matcher.accept_string('{"other":[1,2]}')
    assert copied.matcher.accept_token(EOS)


@pytest.mark.parametrize("limit,flexible", [(-1, True), (0, True), (1, False)])
def test_invalid_configuration_is_rejected(limit, flexible):
    with pytest.raises(ValueError):
        backend(limit, flexible)


def test_serving_configuration_exposes_opt_in_argument():
    from sglang.srt.server_args import ServerArgs

    assert ServerArgs.__dataclass_fields__["constrained_json_max_whitespace_cnt"].default is None
