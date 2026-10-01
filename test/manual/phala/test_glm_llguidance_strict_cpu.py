"""Real GLM tokenizer + startup grammar factory; CPU, not GPU startup evidence.

GLM_TOKENIZER_PATH must contain the pinned deployment's tokenizer/config assets.
No tokenizer substitutes, model weights, network, or CUDA initialization are used.
"""

import json
import os
import unittest
from pathlib import Path

import torch

from sglang.srt.constrained.base_grammar_backend import create_grammar_backend
from sglang.srt.constrained.llguidance_backend import GuidanceBackend, GuidanceGrammar
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarBackend
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.runtime_context import get_context
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer


class TestGLMLLGuidanceStrictCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ["GLM_TOKENIZER_PATH"])
        cls.tokenizer = get_tokenizer(str(path), trust_remote_code=False)
        config = json.loads((path / "config.json").read_text())
        generation = json.loads((path / "generation_config.json").read_text())
        cls.eos = set(generation["eos_token_id"])
        cls.vocab_size = config["vocab_size"]
        cls.parser = ReasoningParser(
            model_type="glm45", stream_reasoning=False, tokenizer=cls.tokenizer
        )
        cls.end = cls.tokenizer.encode(
            cls.parser.detector.think_end_token, add_special_tokens=False
        )
        override = get_context().override_server_args(
            grammar_backend="llguidance",
            reasoning_parser="glm45",
            tool_call_parser="glm47",
            enable_strict_thinking=True,
            constrained_json_whitespace_pattern=None,
            constrained_json_disable_any_whitespace=False,
            constrained_json_max_whitespace_cnt=None,
        )
        args = override.install()
        try:
            # Exact factory called by Scheduler.init_grammar_manager; neither
            # backend nor reasoner constructor is mocked.
            cls.backend = create_grammar_backend(
                args, cls.tokenizer, cls.vocab_size, cls.eos, cls.end
            )
        finally:
            override.restore()
        cls.word = cls.tokenizer.encode("reasoning", add_special_tokens=False)[0]
        assert (
            cls.word not in cls.backend.think_excluded_token_ids
            and cls.word not in cls.end
        )
        assert not torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls):
        cls.backend.executor.shutdown(wait=True)
        cls.backend.grammar_backend.executor.shutdown(wait=True)

    @staticmethod
    def allowed(mask, token, row=0):
        return bool((int(mask[row, token // 32]) >> (token % 32)) & 1)

    def mask(self, obj, batch=1, row=0):
        mask = obj.allocate_vocab_mask(self.vocab_size, batch, "cpu")
        mask.fill_(-1)
        obj.fill_vocab_mask(mask, row)
        return mask

    def json_object(self, reasoning=True):
        obj = self.backend._init_value_dispatch(("json", '{"const":2}'), reasoning)
        self.assertIsInstance(obj.grammar, GuidanceGrammar)
        return obj

    def feed(self, obj, ids):
        for token in ids:
            self.assertTrue(self.allowed(self.mask(obj), token), str(token))
            obj.accept_token(token)

    def test_actual_factory_supports_strict_glm_reasoner(self):
        self.assertIsInstance(self.backend, ReasonerGrammarBackend)
        self.assertIsInstance(self.backend.grammar_backend, GuidanceBackend)
        self.assertTrue(self.backend.enable_strict_thinking)
        self.assertTrue(self.backend.enable_token_filter)
        self.assertTrue(self.backend.grammar_backend.is_support_token_filter)
        self.assertEqual(self.backend.think_end_ids, self.end)
        excluded = [
            i
            for marker in self.parser.detector.think_excluded_tokens
            for i in self.tokenizer.encode(marker, add_special_tokens=False)
        ]
        self.assertEqual(self.backend.think_excluded_token_ids, excluded)
        self.assertEqual(len(self.parser.detector.think_excluded_tokens), 5)

    def test_strict_only_excludes_real_markers_and_applies_logits(self):
        obj = self.backend.init_strict_reasoning_grammar(True)
        self.assertIsNone(obj.grammar)
        mask = self.mask(obj, batch=2, row=1)
        for token in self.backend.think_excluded_token_ids:
            self.assertFalse(self.allowed(mask, token, 1))
            self.assertTrue(self.allowed(mask, token, 0))
        self.assertTrue(self.allowed(mask, self.end[0], 1))
        self.assertTrue(self.allowed(mask, self.word, 1))
        moved = obj.move_vocab_mask(mask, "cpu")
        logits = torch.zeros((2, self.vocab_size), dtype=torch.float32)
        obj.apply_vocab_mask(logits, moved)
        self.assertTrue(
            torch.isneginf(logits[1, self.backend.think_excluded_token_ids]).all()
        )
        self.assertTrue(torch.isfinite(logits[0]).all())

    def test_min_and_max_think_budgets(self):
        obj = self.backend.init_strict_reasoning_grammar(True)
        obj.min_think_tokens = 1
        obj.max_think_tokens = 2
        self.assertFalse(self.allowed(self.mask(obj), self.end[0]))
        self.feed(obj, [self.word])
        self.assertTrue(self.allowed(self.mask(obj), self.end[0]))
        self.feed(obj, [self.word])
        mask = self.mask(obj)
        allowed = [i for i in range(self.vocab_size) if self.allowed(mask, i)]
        self.assertEqual(allowed, [self.end[0]])

    def test_reasoning_isolated_until_end_then_strict_json(self):
        obj = self.json_object()
        pristine = obj.grammar.copy()
        self.feed(obj, [self.word, self.word])
        self.assertEqual(obj.tokens_after_end, -1)
        a = pristine.allocate_vocab_mask(self.vocab_size, 1, "cpu")
        b = obj.grammar.allocate_vocab_mask(self.vocab_size, 1, "cpu")
        pristine.fill_vocab_mask(a, 0)
        obj.grammar.fill_vocab_mask(b, 0)
        self.assertTrue(torch.equal(a, b))
        self.feed(obj, self.end)
        self.assertEqual(obj.tokens_after_end, 0)
        self.assertFalse(self.allowed(self.mask(obj), self.word))
        self.feed(obj, self.tokenizer.encode("2", add_special_tokens=False))
        self.feed(obj, [next(iter(self.eos))])
        self.assertTrue(obj.is_terminated())

    def test_copy_and_rollback_across_reasoning_end(self):
        obj = self.json_object()
        self.feed(obj, [self.word])
        clone = obj.copy()
        self.feed(obj, self.end)
        generation_ids = self.tokenizer.encode("2", add_special_tokens=False)
        self.feed(obj, generation_ids)
        obj.rollback(len(generation_ids) + len(self.end))
        self.assertEqual(obj.tokens_after_end, -1)
        self.assertEqual(obj.tokens_in_think, clone.tokens_in_think)
        self.assertTrue(torch.equal(self.mask(obj), self.mask(clone)))
        self.feed(clone, [self.word])
        self.assertNotEqual(obj.tokens_in_think, clone.tokens_in_think)
        self.feed(obj, self.end + generation_ids)
        self.assertFalse(
            self.allowed(self.mask(clone), self.backend.think_excluded_token_ids[0])
        )

    def test_strict_only_generation_and_rollback_have_fresh_masks(self):
        obj = self.backend.init_strict_reasoning_grammar(True)
        self.feed(obj, [self.word])
        clone = obj.copy()
        self.feed(obj, self.end)
        self.assertTrue(obj.vocab_mask_is_unconstrained)
        mask = self.mask(obj)
        for token in self.backend.think_excluded_token_ids:
            self.assertTrue(self.allowed(mask, token))
        obj.rollback(len(self.end))
        self.assertFalse(obj.vocab_mask_is_unconstrained)
        self.assertTrue(torch.equal(self.mask(obj), self.mask(clone)))

    def test_no_reasoning_requests_start_in_json_schema(self):
        obj = self.json_object(reasoning=False)
        self.assertEqual(obj.tokens_in_think, -1)
        self.assertFalse(self.allowed(self.mask(obj), self.word))
        self.feed(obj, self.tokenizer.encode("2", add_special_tokens=False))


if __name__ == "__main__":
    unittest.main()
