"""Exercise llguidance masks through the callbacks used by reasoning wrappers."""
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.constrained.llguidance_backend import GuidanceBackend
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject


class GuidanceReasonerCallbacks(unittest.TestCase):
    def test_backend_mask_uses_effective_tokenizer_vocabulary(self):
        backend = object.__new__(GuidanceBackend)
        backend.llguidance_tokenizer = SimpleNamespace(vocab_size=257)
        mask = backend.allocate_vocab_mask(300, 3, "cpu")
        self.assertEqual(mask.dtype, torch.int32)
        self.assertEqual(tuple(mask.shape), (3, 9))
        self.assertTrue(torch.equal(mask, backend.move_vocab_mask(mask, "cpu")))

    def test_reasoning_only_wrapper_filters_the_requested_tokens(self):
        backend = object.__new__(GuidanceBackend)
        backend.llguidance_tokenizer = SimpleNamespace(vocab_size=3)
        wrapped = ReasonerGrammarObject(
            grammar=None, think_end_ids=[2],
            allocate_vocab_mask_fn=backend.allocate_vocab_mask,
            move_vocab_mask_fn=backend.move_vocab_mask,
            apply_vocab_mask_fn=backend.apply_vocab_mask,
        )
        mask = wrapped.allocate_vocab_mask(3, 1, "cpu")
        mask.zero_()
        mask[0, 0] = 5
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        wrapped.apply_vocab_mask(logits, wrapped.move_vocab_mask(mask, "cpu"))
        self.assertEqual(logits[0, 0].item(), 1.0)
        self.assertTrue(torch.isneginf(logits[0, 1]).item())
        self.assertEqual(logits[0, 2].item(), 3.0)


if __name__ == "__main__":
    unittest.main()
