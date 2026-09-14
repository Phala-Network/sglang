"""Decoded string minima and token rollback across finite-state block edges."""

import json

import pytest
import torch
import xgrammar as xg


@pytest.fixture(scope="module")
def compiler():
    vocab = [bytes([i]) for i in range(256)] + [b"aa", b'a"', b'aa"', b"<eos>"]
    info = xg.TokenizerInfo(vocab, xg.VocabType.RAW, stop_token_ids=[259])
    return xg.GrammarCompiler(info, max_threads=2)


@pytest.mark.parametrize("minimum", [1, 127, 128, 129, 200, 1000, 16385])
@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("ascii_only", [False, True])
def test_decoded_unicode_length_at_block_edges(compiler, minimum, delta, ascii_only):
    ctx = compiler.compile_json_schema(
        json.dumps({"type": "string", "minLength": minimum})
    )
    count = minimum + delta
    value = "a" * max(count - 1, 0) + ("😀" if count else "")
    matcher = xg.GrammarMatcher(ctx)
    assert matcher.accept_string(json.dumps(value, ensure_ascii=ascii_only)) == (
        delta >= 0
    )


@pytest.mark.parametrize("escape", [r"\n", r"\"", r"\\", r"\u0041", r"\uD83D\uDE00"])
def test_escape_is_one_character_at_128_boundary(compiler, escape):
    ctx = compiler.compile_json_schema('{"type":"string","minLength":129}')
    matcher = xg.GrammarMatcher(ctx)
    assert matcher.accept_string('"' + "a" * 127 + escape)
    assert not matcher.accept_string('"')
    assert matcher.accept_string('a"')


@pytest.mark.parametrize(
    "invalid", [r"\uD800", r"\uDC00", r"\uD83D\u0041", "\t", "\x00"]
)
def test_invalid_escape_or_raw_control_rejected(compiler, invalid):
    ctx = compiler.compile_json_schema('{"type":"string","minLength":129}')
    matcher = xg.GrammarMatcher(ctx)
    assert not matcher.accept_string('"' + "a" * 128 + invalid + '"')


def test_multichar_token_mask_and_rollback_across_block_boundary(compiler):
    ctx = compiler.compile_json_schema('{"type":"string","minLength":129}')
    matcher = xg.GrammarMatcher(ctx, max_rollback_tokens=5)
    assert matcher.accept_string('"' + "a" * 127)
    mask = xg.allocate_token_bitmask(1, 260)
    matcher.fill_next_token_bitmask(mask)
    before = mask.clone()

    def allowed(index):
        return bool(int(mask[0, index // 32]) & (1 << (index % 32)))

    assert allowed(256)  # Two characters cross the minimum.
    assert not allowed(257)  # One character plus quote is still too short.
    assert allowed(258)  # Two characters plus quote close valid JSON.
    assert matcher.accept_token(256)
    matcher.fill_next_token_bitmask(mask)
    reached = mask.clone()
    assert allowed(ord('"'))
    matcher.rollback(1)
    matcher.fill_next_token_bitmask(mask)
    assert torch.equal(mask, before)
    assert matcher.accept_token(256)
    matcher.fill_next_token_bitmask(mask)
    assert torch.equal(mask, reached)
    assert matcher.accept_token(ord('"'))
    assert matcher.accept_token(259)
    assert matcher.is_terminated()
    matcher.rollback(2)
    matcher.fill_next_token_bitmask(mask)
    assert torch.equal(mask, reached)
