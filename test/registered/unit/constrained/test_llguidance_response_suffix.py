"""CPU correctness checks for the llguidance Kimi response trailer adapter."""

import unittest

import torch
from llguidance import LLMatcher, LLTokenizer
from llguidance.torch import allocate_token_bitmask

from sglang.srt.constrained.base_grammar_backend import GrammarRow
from sglang.srt.constrained.llguidance_backend import GuidanceGrammar
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
