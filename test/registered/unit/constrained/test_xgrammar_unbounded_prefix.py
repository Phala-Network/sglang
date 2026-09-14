"""Preserve counted-language and token rollback semantics for large minima."""

import json

import pytest
import torch
import xgrammar as xg


@pytest.fixture(scope="module")
def compiler():
    vocab = [bytes([i]) for i in range(256)] + [b"<eos>"]
    info = xg.TokenizerInfo(vocab, xg.VocabType.RAW, stop_token_ids=[256])
    return xg.GrammarCompiler(info, max_threads=2)


@pytest.mark.parametrize("minimum", [128, 129, 200, 1000])
@pytest.mark.parametrize("delta", [-1, 0, 1, 129])
def test_repetition_minimum_is_preserved(compiler, minimum, delta):
    grammar = f'root ::= "[" ("ab" | "cd"){{{minimum},}} "]"'
    matcher = xg.GrammarMatcher(compiler.compile_grammar(grammar))
    assert matcher.accept_string("[" + "ab" * (minimum + delta))
    assert matcher.accept_string("]") == (delta >= 0)
    if delta >= 0:
        assert matcher.accept_token(256)
        assert matcher.is_terminated()


@pytest.mark.parametrize("minimum", [129, 200])
def test_rollback_across_minimum_and_termination(compiler, minimum):
    matcher = xg.GrammarMatcher(
        compiler.compile_grammar(f'root ::= "a"{{{minimum},}} "!"')
    )
    mask = xg.allocate_token_bitmask(1, 257)
    for _ in range(minimum - 1):
        assert matcher.accept_token(ord("a"))
    matcher.fill_next_token_bitmask(mask)
    before = mask.clone()
    assert not (int(mask[0, ord("!") // 32]) & (1 << (ord("!") % 32)))
    assert matcher.accept_token(ord("a"))
    matcher.fill_next_token_bitmask(mask)
    reached = mask.clone()
    assert int(mask[0, ord("!") // 32]) & (1 << (ord("!") % 32))
    matcher.rollback(1)
    matcher.fill_next_token_bitmask(mask)
    assert torch.equal(mask, before)
    assert matcher.accept_token(ord("a"))
    matcher.fill_next_token_bitmask(mask)
    assert torch.equal(mask, reached)
    assert matcher.accept_token(ord("!"))
    assert matcher.accept_token(256)
    matcher.rollback(2)
    matcher.fill_next_token_bitmask(mask)
    assert torch.equal(mask, reached)


@pytest.mark.parametrize("tail", ["a", "\n", '"', "\\", "中", "😀"])
def test_json_length_counts_decoded_characters(compiler, tail):
    schema = {"type": "string", "minLength": 129}
    ctx = compiler.compile_json_schema(json.dumps(schema))
    for count in (128, 129, 130):
        matcher = xg.GrammarMatcher(ctx)
        text = "a" * (count - 1) + tail
        assert matcher.accept_string(json.dumps(text, ensure_ascii=True)) == (
            count >= 129
        )


@pytest.mark.parametrize("count", [128, 129, 130, 200, 201])
def test_bounded_range_unchanged(compiler, count):
    ctx = compiler.compile_grammar('root ::= "[" "a"{129,200} "]"')
    matcher = xg.GrammarMatcher(ctx)
    assert matcher.accept_string("[" + "a" * count + "]") == (129 <= count <= 200)


def test_large_minimum_keeps_compact_compilation(compiler):
    # A large client minimum must not turn into a million explicitly unrolled
    # rules. Check the accepted language on both sides of the exact boundary.
    ctx = compiler.compile_grammar('root ::= "a"{100000,} "!"')
    matcher = xg.GrammarMatcher(ctx)
    assert matcher.accept_string("a" * 99999)
    assert not matcher.accept_string("!")
    assert matcher.accept_string("a!")


@pytest.mark.parametrize("length", [128, 129, 258, 300])
def test_overlapping_repeated_alternatives(compiler, length):
    ctx = compiler.compile_grammar('root ::= ("a" | "aa"){129,} "!"')
    matcher = xg.GrammarMatcher(ctx)
    assert matcher.accept_string("a" * length + "!") == (length >= 129)


@pytest.mark.parametrize("length", [0, 1, 129, 200])
def test_nullable_repeated_rule(compiler, length):
    ctx = compiler.compile_grammar('root ::= ("a" | ""){129,} "!"')
    matcher = xg.GrammarMatcher(ctx)
    assert matcher.accept_string("a" * length + "!")
