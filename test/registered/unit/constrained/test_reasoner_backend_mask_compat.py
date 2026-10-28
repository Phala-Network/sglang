"""CPU regressions for reasoning wrappers around grammar-owned vocab masks."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from llguidance import LLTokenizer
from transformers import AutoTokenizer
from xgrammar import TokenizerInfo

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    InvalidGrammarObject,
)
from sglang.srt.constrained.llguidance_backend import GuidanceBackend
from sglang.srt.constrained.reasoner_grammar_backend import (
    ReasonerGrammarBackend,
    ReasonerGrammarObject,
)
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _allowed(mask, token):
    return bool(int(mask[0, token // 32].item()) & (1 << (token % 32)))


def _parser():
    return SimpleNamespace(
        detector=SimpleNamespace(
            think_end_token="</think>",
            think_excluded_tokens=["<tool_call>", "</tool_call>"],
        )
    )


def _tokenizer():
    info = TokenizerInfo(
        [chr(i) for i in range(128)] + ["</think>", "<eos>"],
        stop_token_ids=[129],
    )
    return SimpleNamespace(
        encode=lambda text, add_special_tokens=False: {
            "</think>": [128],
            "<tool_call>": [3],
            "</tool_call>": [5],
        }.get(text, list(text.encode())),
        init_xgrammar=lambda: (info, None),
    )


class TestReasonerBackendMaskCompatibility(unittest.TestCase):
    def setUp(self):
        self.backends = []

    def tearDown(self):
        for backend in self.backends:
            backend.executor.shutdown(wait=True)

    def _reasoner(self, name="llguidance", strict=False):
        tokenizer = _tokenizer()
        if name == "llguidance":
            backend = object.__new__(GuidanceBackend)
            BaseGrammarBackend.__init__(backend)
            backend.llguidance_tokenizer = LLTokenizer("byte")
            backend.any_whitespace = True
            backend.whitespace_pattern = None
        else:
            backend = XGrammarGrammarBackend(tokenizer, 130)
        self.backends.append(backend)
        reasoner = ReasonerGrammarBackend(
            backend, _parser(), tokenizer, enable_strict_thinking=strict
        )
        self.backends.append(reasoner)
        return reasoner

    def _mask(self, grammar, vocab_size):
        mask = grammar.allocate_vocab_mask(vocab_size, 1, "cpu")
        grammar.fill_vocab_mask(mask, 0)
        return grammar.move_vocab_mask(mask, "cpu")

    def test_llguidance_wraps_nonempty_grammar_and_applies_its_mask(self):
        reasoner = self._reasoner()
        grammar = reasoner._init_value_dispatch(("regex", "[0-9]{1,8}"), False)
        self.assertIsInstance(grammar, ReasonerGrammarObject)
        vocab_size = reasoner.grammar_backend.llguidance_tokenizer.vocab_size
        mask = self._mask(grammar, vocab_size)
        self.assertEqual(mask.dtype, torch.int32)
        logits = torch.zeros((1, vocab_size))
        grammar.apply_vocab_mask(logits, mask)
        self.assertEqual(
            torch.isfinite(logits[0]).nonzero().flatten().tolist(),
            list(range(ord("0"), ord("9") + 1)),
        )

    def test_llguidance_defers_constraints_until_reasoning_end(self):
        reasoner = self._reasoner()
        grammar = reasoner._init_value_dispatch(("regex", "ab|cd"), True)
        mask = self._mask(grammar, 257)
        self.assertTrue((mask == -1).all())
        grammar.accept_token(ord("z"))
        grammar.accept_token(128)
        mask = self._mask(grammar, 257)
        self.assertTrue(_allowed(mask, ord("a")))
        self.assertTrue(_allowed(mask, ord("c")))
        self.assertFalse(_allowed(mask, ord("z")))

    def test_cached_llguidance_copies_delegate_to_independent_matchers(self):
        reasoner = self._reasoner()
        key = ("regex", "ab|cd")
        future, hit = reasoner.get_cached_or_future_value(key, False)
        self.assertFalse(hit)
        template = future.result(timeout=30)
        reasoner.set_cache(key, template)
        first, hit = reasoner.get_cached_or_future_value(key, False)
        self.assertTrue(hit)
        second, hit = reasoner.get_cached_or_future_value(key, False)
        self.assertTrue(hit)
        thinking, _ = reasoner.get_cached_or_future_value(key, True)
        self.assertIsNot(first.grammar, second.grammar)
        first.accept_token(ord("a"))
        first_mask = self._mask(first, 257)
        second_mask = self._mask(second, 257)
        template_mask = self._mask(template, 257)
        self.assertTrue(_allowed(first_mask, ord("b")))
        self.assertFalse(_allowed(first_mask, ord("c")))
        self.assertTrue(_allowed(second_mask, ord("a")))
        self.assertTrue(torch.equal(second_mask, template_mask))
        self.assertTrue((self._mask(thinking, 257) == -1).all())
        first.rollback(1)
        self.assertTrue(torch.equal(self._mask(first, 257), second_mask))
        first.accept_token(ord("c"))
        first.accept_token(ord("d"))
        eos = reasoner.grammar_backend.llguidance_tokenizer.eos_tokens[0]
        self.assertTrue(_allowed(self._mask(first, 257), eos))
        first.accept_token(eos)
        self.assertTrue(first.is_terminated())
        self.assertFalse(second.is_terminated())

    def test_xgrammar_nonempty_masks_and_request_copy_are_unchanged(self):
        reasoner = self._reasoner("xgrammar")
        grammar = reasoner._init_value_dispatch(("regex", "ab|cd"), False)
        copied = grammar.copy()
        grammar.accept_token(ord("a"))
        mask = self._mask(copied, 130)
        logits = torch.zeros((1, 130))
        copied.apply_vocab_mask(logits, mask)
        self.assertTrue(torch.isfinite(logits[0, ord("a")]))
        self.assertTrue(torch.isneginf(logits[0, ord("b")]))
        self.assertTrue(_allowed(self._mask(grammar, 130), ord("b")))

    def test_xgrammar_strict_only_retains_backend_hooks_and_budget(self):
        with patch.dict(os.environ, {"SGLANG_MAX_THINK_TOKENS": "2"}):
            reasoner = self._reasoner("xgrammar", strict=True)
        grammar = reasoner.init_strict_reasoning_grammar(True)
        self.assertIsNone(grammar.grammar)
        mask = self._mask(grammar, 130)
        self.assertFalse(_allowed(mask, 3))
        self.assertFalse(_allowed(mask, 5))
        self.assertTrue(_allowed(mask, 128))
        grammar.accept_token(ord("a"))
        grammar.accept_token(ord("b"))
        copied = grammar.copy()
        forced_mask = self._mask(copied, 130)
        self.assertEqual(
            [i for i in range(130) if _allowed(forced_mask, i)],
            [128],
        )
        # b070's strict-only backend apply is GPU-only; preserve that boundary.
        with self.assertRaisesRegex(RuntimeError, "Unsupported device: cpu"):
            copied.apply_vocab_mask(torch.zeros((1, 130)), mask)
        copied.accept_token(128)
        self.assertTrue(copied._is_generation())
        self.assertTrue(grammar._is_thinking())

    def test_llguidance_strict_filter_still_rejects_unsupported_backend(self):
        with self.assertRaisesRegex(ValueError, "does not support token filtering"):
            self._reasoner(strict=True)

    def test_no_schema_with_strict_disabled_still_has_no_grammar(self):
        self.assertIsNone(self._reasoner().init_strict_reasoning_grammar(True))

    def test_invalid_grammar_and_whole_assistant_ebnf_are_not_wrapped(self):
        reasoner = self._reasoner()
        invalid = reasoner._init_value_dispatch(("json", "not json"), False)
        self.assertIsInstance(invalid, InvalidGrammarObject)
        full = reasoner._init_value_dispatch(
            ("full_assistant_ebnf", 'root ::= "OK"'), True
        )
        self.assertNotIsInstance(full, ReasonerGrammarObject)
        self.assertNotIsInstance(full, InvalidGrammarObject)


class TestGemmaReasonerMaskCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snapshot = os.environ.get("GRAMMAR_COMPAT_GEMMA_SNAPSHOT")
        if not snapshot:
            raise unittest.SkipTest("requires the pinned local Gemma tokenizer")
        cls.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        cls.backend = GuidanceBackend(
            cls.tokenizer,
            n_vocab=len(cls.tokenizer),
            eos_token_ids=cls.tokenizer.eos_token_id,
        )
        cls.reasoner = ReasonerGrammarBackend(
            cls.backend,
            ReasoningParser("gemma4", stream_reasoning=False, tokenizer=cls.tokenizer),
            cls.tokenizer,
        )

    @classmethod
    def tearDownClass(cls):
        cls.reasoner.executor.shutdown(wait=True)
        cls.backend.executor.shutdown(wait=True)

    def _accept_sample(self, schema, sample):
        grammar = self.reasoner._init_value_dispatch(
            ("json", json.dumps(schema)), False
        )
        self.assertIsInstance(grammar, ReasonerGrammarObject)
        vocab_size = self.backend.llguidance_tokenizer.vocab_size
        mask = grammar.allocate_vocab_mask(vocab_size, 1, "cpu")
        for token in self.tokenizer.encode(sample, add_special_tokens=False):
            grammar.fill_vocab_mask(mask, 0)
            logits = torch.zeros((1, vocab_size))
            grammar.apply_vocab_mask(logits, grammar.move_vocab_mask(mask, "cpu"))
            self.assertTrue(torch.isfinite(logits[0, token]), f"rejected token {token}")
            grammar.accept_token(token)
        grammar.fill_vocab_mask(mask, 0)
        self.assertTrue(_allowed(mask, self.tokenizer.eos_token_id))
        grammar.accept_token(self.tokenizer.eos_token_id)
        self.assertTrue(grammar.is_terminated())

    def test_pinned_gemma_strict_city_object_grammar(self):
        schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
                "population": {"type": "integer"},
                "notable": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["city", "country", "population", "notable"],
            "additionalProperties": False,
        }
        self._accept_sample(
            schema,
            '{"city":"Nagoya","country":"Japan","population":2330000,'
            '"notable":["Nagoya Castle","Toyota"]}',
        )

    def test_pinned_gemma_named_tool_array_grammar(self):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": ["get_weather"]},
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
                "required": ["name", "parameters"],
                "additionalProperties": False,
            },
            "minItems": 1,
        }
        self._accept_sample(
            schema, '[{"name":"get_weather","parameters":{"city":"Paris"}}]'
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
