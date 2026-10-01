"""CPU correctness checks for the llguidance Kimi response trailer adapter."""

import json
import unittest
from types import SimpleNamespace

import tiktoken
import torch
from llguidance import LLMatcher, LLTokenizer
from llguidance.torch import allocate_token_bitmask

from sglang.srt.constrained.base_grammar_backend import GrammarRow
from sglang.srt.constrained.llguidance_backend import (
    GuidanceBackend,
    GuidanceGrammar,
    _create_llguidance_tokenizer,
    _normalize_llguidance_schema_noops,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLLGuidanceResponseSuffix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = LLTokenizer("byte")
        serialized = LLMatcher.grammar_from_json_schema(
            '{"type":"integer","multipleOf":2}'
        )
        cls.template = GuidanceGrammar(cls.tokenizer, serialized)
        cls.suffix = (ord("<"), ord("E"), ord("O"), 261)

    @staticmethod
    def _allowed(mask, token):
        return bool((int(mask[0, token // 32]) >> (token % 32)) & 1)

    def _mask(self, grammar):
        mask = allocate_token_bitmask(1, self.tokenizer.vocab_size)
        grammar.fill_vocab_mask(mask, 0)
        return mask

    def _complete_json(self, grammar):
        grammar.accept_token(ord("2"))

    def test_suffix_requires_complete_json_and_ends_at_eos(self):
        grammar = self.template.copy().with_response_suffix(self.suffix)
        self.assertFalse(self._allowed(self._mask(grammar), self.suffix[0]))
        self._complete_json(grammar)
        self.assertTrue(self._allowed(self._mask(grammar), self.suffix[0]))
        grammar.accept_token(self.suffix[0])
        for token in self.suffix[1:]:
            self.assertTrue(self._allowed(self._mask(grammar), token))
            grammar.accept_token(token)
        self.assertTrue(grammar.is_terminated())

    def test_rollback_restores_json_and_eos_state(self):
        grammar = self.template.copy().with_response_suffix(self.suffix)
        self._complete_json(grammar)
        for token in self.suffix:
            grammar.accept_token(token)
        grammar.rollback(len(self.suffix))
        self.assertFalse(grammar.is_terminated())
        self.assertEqual(grammar.suffix_position, 0)
        self.assertTrue(self._allowed(self._mask(grammar), self.suffix[0]))

        direct = self.template.copy()
        self._complete_json(direct)
        direct.accept_token(261)
        self.assertTrue(direct.is_terminated())
        direct.rollback(1)
        self.assertFalse(direct.is_terminated())

    def test_string_content_does_not_enter_suffix_state(self):
        serialized = LLMatcher.grammar_from_json_schema('{"type":"string"}')
        grammar = GuidanceGrammar(self.tokenizer, serialized).with_response_suffix(
            self.suffix
        )
        for token in map(ord, '"<'):
            grammar.accept_token(token)
        self.assertEqual(grammar.suffix_position, 0)
        grammar.accept_token(ord('"'))
        self.assertTrue(self._allowed(self._mask(grammar), self.suffix[0]))

    def test_suffix_mask_uses_serial_fallback(self):
        grammar = self.template.copy().with_response_suffix(self.suffix)
        self._complete_json(grammar)
        serial = self._mask(grammar)
        batched = allocate_token_bitmask(1, self.tokenizer.vocab_size)
        GuidanceGrammar.fill_vocab_mask_batched(
            [GrammarRow(row=0, grammar=grammar)], batched
        )
        self.assertTrue(torch.equal(serial, batched))


class TestTiktokenLLGuidanceConversion(unittest.TestCase):
    def setUp(self):
        self.encoding = tiktoken.Encoding(
            name="llguidance-cpu-fixture",
            pat_str=r"(?s:.)",
            mergeable_ranks={bytes([i]): i for i in range(256)},
            special_tokens={"<eos>": 256, "<other_eos>": 257},
        )
        self.tokenizer = SimpleNamespace(model=self.encoding, eos_token_id=256)

    def test_native_encoding_preserves_utf8_and_default_eos(self):
        tokenizer = _create_llguidance_tokenizer(self.tokenizer, 258, None)
        for text in ("plain", "caf\u00e9", "\u4f60\u597d", '{"x":2}'):
            ids = self.encoding.encode(text)
            self.assertEqual(tokenizer.tokenize_str(text), ids)
            self.assertEqual(tokenizer.decode_bytes(ids), text.encode("utf-8"))
        self.assertEqual(tokenizer.eos_tokens, [256])

    def test_explicit_eos_and_padded_vocabulary_are_preserved(self):
        tokenizer = _create_llguidance_tokenizer(self.tokenizer, 264, [256, 257])
        self.assertEqual(tokenizer.vocab_size, 264)
        self.assertEqual(set(tokenizer.eos_tokens), {256, 257})
        grammar = GuidanceGrammar(
            tokenizer, LLMatcher.grammar_from_json_schema('{"type":"integer"}')
        )
        mask = allocate_token_bitmask(1, tokenizer.vocab_size)
        grammar.fill_vocab_mask(mask, 0)
        for token in range(258, 264):
            self.assertFalse((int(mask[0, token // 32]) >> (token % 32)) & 1)

    def test_json_schema_noops_are_preserved_without_weakening_constraints(self):
        backend = GuidanceBackend(self.tokenizer, n_vocab=258, eos_token_ids=[256])
        schemas = [
            {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
                "additionalProperties": False,
                "propertyNames": True,
            },
            {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
                "additionalProperties": False,
                "uniqueItems": True,
            },
            {
                "type": "array",
                "items": {"type": "integer"},
                "minContains": 1,
                "maxContains": 2,
                "uniqueItems": False,
            },
        ]
        for schema in schemas:
            normalized = _normalize_llguidance_schema_noops(schema)
            self.assertIsInstance(
                backend.dispatch_json(json.dumps(schema)), GuidanceGrammar
            )
            self.assertNotEqual(normalized, schema)

        # An effective array uniqueness constraint remains strict and is not
        # silently erased merely because the no-op cases above are accepted.
        effective = {
            "type": "array",
            "items": {"type": "integer"},
            "uniqueItems": True,
        }
        self.assertIn("uniqueItems", _normalize_llguidance_schema_noops(effective))
        self.assertNotIsInstance(
            backend.dispatch_json(json.dumps(effective)), GuidanceGrammar
        )


class TestLLGuidanceStructuralTags(unittest.TestCase):
    def test_fast_tokenizer_mixed_literal_and_special_triggers(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        tokenizer_impl = Tokenizer(
            models.BPE(
                vocab={c: i for i, c in enumerate(pre_tokenizers.ByteLevel.alphabet())},
                merges=[],
            )
        )
        tokenizer_impl.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer_impl.decoder = decoders.ByteLevel()
        tokenizer_impl.add_special_tokens(["<eos>", "<call>"])
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_impl, eos_token="<eos>"
        )
        backend = GuidanceBackend(tokenizer, n_vocab=len(tokenizer))
        for triggers in (("<call>", "<other>"), ("<call>alpha", "<call>beta")):
            grammar_spec = {
                "type": "structural_tag",
                "triggers": list(triggers),
                "structures": [
                    {"begin": tag, "end": "</end>", "schema": {"type": "integer"}}
                    for tag in triggers
                ],
            }
            grammar = backend.dispatch_structural_tag(json.dumps(grammar_spec))
            self.assertIsInstance(grammar, GuidanceGrammar)
            for tag in triggers:
                constrained = backend.dispatch_structural_tag(json.dumps(grammar_spec))
                for token in tokenizer.encode(tag, add_special_tokens=False):
                    constrained.accept_token(token)
                mask = allocate_token_bitmask(1, len(tokenizer))
                constrained.fill_vocab_mask(mask, 0)
                for bad_token in (
                    tokenizer.encode("x", add_special_tokens=False)[0],
                    tokenizer.encode("</end>", add_special_tokens=False)[0],
                    tokenizer.eos_token_id,
                ):
                    self.assertFalse(
                        (int(mask[0, bad_token // 32]) >> (bad_token % 32)) & 1,
                        (triggers, tag, bad_token),
                    )
                text = tag + "2</end>"
                for token in tokenizer.encode(text, add_special_tokens=False):
                    mask = allocate_token_bitmask(1, len(tokenizer))
                    grammar.fill_vocab_mask(mask, 0)
                    self.assertTrue(
                        (int(mask[0, token // 32]) >> (token % 32)) & 1,
                        (triggers, text, token),
                    )
                    grammar.accept_token(token)


if __name__ == "__main__":
    unittest.main(verbosity=2)
