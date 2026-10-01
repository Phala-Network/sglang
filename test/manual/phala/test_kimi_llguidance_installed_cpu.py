"""Focused CPU test against installed llguidance and the real Kimi tokenizer.

KIMI_TOKENIZER_PATH points to the immutable tokenizer/config fixture; no model
weights, network, CUDA initialization, or serving-process mutation is needed.
"""

import json
import os
import unittest
from pathlib import Path

import torch
from llguidance.torch import allocate_token_bitmask

from sglang.srt.constrained.llguidance_backend import (
    GuidanceBackend,
    GuidanceResponseSuffixGrammar,
)
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarBackend

ROLLBACK_HISTORY_TEST_TOKENS = 200
from sglang.srt.parser.reasoning_parser import KimiK3Detector, ReasoningParser
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer


class KimiResponseSuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ["KIMI_TOKENIZER_PATH"])
        cls.tokenizer = get_tokenizer(str(path), trust_remote_code=True)
        config = json.loads((path / "config.json").read_text())
        cls.backend = GuidanceBackend(
            cls.tokenizer,
            n_vocab=len(cls.tokenizer),
            eos_token_ids=[config["eos_token_id"]],
        )
        cls.suffix = cls.tokenizer.encode(
            KimiK3Detector.grammar_response_suffix, add_special_tokens=False
        )
        cls.reasoner = ReasonerGrammarBackend(
            cls.backend,
            ReasoningParser(model_type="kimi_k3", tokenizer=cls.tokenizer),
            cls.tokenizer,
        )

    def grammar(self, schema=None):
        return self.backend.dispatch_json(
            json.dumps(schema or {"type": "integer", "multipleOf": 2})
        ).with_response_suffix(self.suffix)

    def accept(self, grammar, text):
        for token in self.tokenizer.encode(text, add_special_tokens=False):
            grammar.accept_token(token)

    def mask(self, grammar):
        mask = allocate_token_bitmask(1, self.backend.llguidance_tokenizer.vocab_size)
        grammar.fill_vocab_mask(mask, 0)
        return mask

    def allowed(self, mask, token):
        return bool((int(mask[0, token // 32]) >> (token % 32)) & 1)

    def test_valid_completion_allows_exact_suffix_and_direct_eos(self):
        for schema in ({"type": "integer"}, {"type": "integer", "multipleOf": 2}):
            for text in ("2", "22", "222"):
                with self.subTest(schema=schema, text=text):
                    grammar = self.grammar(schema)
                    self.accept(grammar, text)
                    mask = self.mask(grammar)
                    self.assertTrue(self.allowed(mask, self.suffix[0]))
                    self.assertTrue(self.allowed(mask, self.suffix[-1]))
                    for i, token in enumerate(self.suffix):
                        self.assertTrue(self.allowed(self.mask(grammar), token))
                        grammar.accept_token(token)
                        self.assertEqual(
                            grammar.is_terminated(), i == len(self.suffix) - 1
                        )
                    direct = self.grammar(schema)
                    self.accept(direct, text)
                    direct.accept_token(self.suffix[-1])
                    self.assertTrue(direct.is_terminated())

    def test_incomplete_or_odd_json_cannot_escape(self):
        for schema, text in (
            ({"type": "integer", "multipleOf": 2}, ""),
            ({"type": "integer", "multipleOf": 2}, "3"),
            (
                {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
                '{"x":',
            ),
        ):
            with self.subTest(text=text):
                grammar = self.grammar(schema)
                self.accept(grammar, text)
                self.assertFalse(self.allowed(self.mask(grammar), self.suffix[0]))
                with self.assertRaises(ValueError):
                    grammar.accept_token(self.suffix[0])
                self.assertEqual(grammar.suffix_position, 0)
                self.assertFalse(grammar.is_terminated())
        literal = self.grammar({"type": "string"})
        self.accept(literal, '"unfinished')
        # Kimi's close marker is also valid literal JSON string data. It must
        # stay inside native grammar, not start a protocol trailer or abort.
        literal.accept_token(self.suffix[0])
        self.assertEqual(literal.suffix_position, 0)
        self.assertFalse(literal.is_terminated())
        self.accept(literal, '"')
        for token in self.suffix:
            literal.accept_token(token)
        self.assertTrue(literal.is_terminated())

    def test_trailer_rejects_wrong_token_and_early_eos(self):
        grammar = self.grammar()
        self.accept(grammar, "2")
        grammar.accept_token(self.suffix[0])
        for bad in (self.suffix[-1], self.tokenizer.encode("2")[0]):
            self.assertFalse(self.allowed(self.mask(grammar), bad))
            with self.assertRaises(ValueError):
                grammar.accept_token(bad)
            self.assertEqual(grammar.suffix_position, 1)
        for token in self.suffix[1:]:
            grammar.accept_token(token)
        self.assertTrue(grammar.is_terminated())

    def test_speculative_rollback_crosses_trailer_and_json(self):
        grammar = self.grammar()
        self.accept(grammar, "2")
        for token in self.suffix:
            grammar.accept_token(token)
        grammar.rollback(3)
        self.assertEqual(grammar.suffix_position, len(self.suffix) - 3)
        self.assertFalse(grammar.is_terminated())
        for token in self.suffix[-3:]:
            grammar.accept_token(token)
        self.assertTrue(grammar.is_terminated())
        grammar.rollback(len(self.suffix) + 1)
        self.assertEqual(grammar.suffix_position, 0)
        self.assertFalse(self.allowed(self.mask(grammar), self.suffix[0]))
        self.accept(grammar, "2")
        for token in self.suffix:
            grammar.accept_token(token)
        self.assertTrue(grammar.is_terminated())

    def test_copy_preserves_state_and_jump_forward_is_disabled(self):
        grammar = self.grammar()
        self.accept(grammar, "2")
        grammar.accept_token(self.suffix[0])
        clone = grammar.copy()  # llguidance deep-copies current matcher state.
        self.assertIsInstance(clone, GuidanceResponseSuffixGrammar)
        self.assertEqual(clone.suffix_position, 1)
        self.assertTrue(torch.equal(self.mask(clone), self.mask(grammar)))
        for token in self.suffix[1:]:
            clone.accept_token(token)
        self.assertTrue(clone.is_terminated())
        self.assertFalse(grammar.is_terminated())
        self.assertEqual(grammar.suffix_position, 1)
        self.assertIsNone(grammar.try_jump_forward(self.tokenizer))
        self.assertIsNone(clone.try_jump_forward(self.tokenizer))

    def test_suffix_entry_preserves_full_native_rollback_history(self):
        grammar = self.grammar()
        digit = self.tokenizer.encode("2", add_special_tokens=False)[0]
        for _ in range(ROLLBACK_HISTORY_TEST_TOKENS):
            grammar.accept_token(digit)
        grammar.accept_token(self.suffix[0])
        # Keep the adapter's historical 200-token boundary covered, without
        # assuming the current native library still has a finite history limit.
        grammar.rollback(ROLLBACK_HISTORY_TEST_TOKENS + 1)
        self.assertEqual(grammar.suffix_position, 0)
        self.assertFalse(self.allowed(self.mask(grammar), self.suffix[0]))

    def test_reasoner_rollback_crosses_think_json_and_trailer(self):
        grammar = self.reasoner._init_value_dispatch(
            ("json", '{"type":"integer","multipleOf":2}'), True
        )
        marker = self.tokenizer.encode(
            "<|close|>think<|sep|><|open|>response<|sep|>",
            add_special_tokens=False,
        )
        grammar.set_request_think_end_ids(marker)
        self.accept(grammar, "thought")
        for token in marker:
            grammar.accept_token(token)
        self.assertTrue(grammar._is_generation())
        self.accept(grammar, "2")
        grammar.accept_token(self.suffix[0])
        grammar.rollback(3)  # trailer opener, JSON digit, final think marker
        self.assertFalse(grammar._is_generation())
        self.assertEqual(grammar.grammar.suffix_position, 0)
        grammar.accept_token(marker[-1])
        self.accept(grammar, "2")
        for token in self.suffix:
            grammar.accept_token(token)
        self.assertTrue(grammar.is_terminated())

    def test_only_kimi_json_dispatch_gets_wrapper(self):
        wrapped = self.reasoner._init_value_dispatch(
            ("json", '{"type":"integer"}'), False
        )
        self.assertIsInstance(wrapped.grammar, GuidanceResponseSuffixGrammar)
        regex = self.reasoner._init_value_dispatch(("regex", "[0-9]+"), False)
        self.assertNotIsInstance(regex.grammar, GuidanceResponseSuffixGrammar)
        structural = self.reasoner._init_value_dispatch(
            (
                "structural_tag",
                json.dumps(
                    {
                        "type": "structural_tag",
                        "triggers": ["<call>"],
                        "structures": [
                            {
                                "begin": "<call>",
                                "end": "</call>",
                                "schema": {"type": "integer"},
                            }
                        ],
                    }
                ),
            ),
            False,
        )
        self.assertNotIsInstance(structural.grammar, GuidanceResponseSuffixGrammar)
        other = ReasonerGrammarBackend(
            self.backend,
            ReasoningParser(model_type="qwen3", tokenizer=self.tokenizer),
            self.tokenizer,
        )._init_value_dispatch(("json", '{"type":"integer"}'), False)
        self.assertNotIsInstance(other.grammar, GuidanceResponseSuffixGrammar)
        plain = self.backend.dispatch_json('{"type":"integer"}')
        self.assertNotIsInstance(plain, GuidanceResponseSuffixGrammar)
        self.accept(plain, "2")
        self.assertFalse(self.allowed(self.mask(plain), self.suffix[0]))

    def test_response_parser_removes_the_exact_trailer(self):
        detector = KimiK3Detector(stream_reasoning=False, force_reasoning=False)
        # The scheduler excludes matched EOS from decoded response text.
        trailer = self.tokenizer.decode(self.suffix[:-1])
        result = detector.detect_and_parse("2" + trailer)
        self.assertEqual(result.normal_text, "2")
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main(verbosity=2)
