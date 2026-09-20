"""Identity-mask removal must preserve constraints, rollback and barriers."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_sampling_batch_info as sampling_tests
import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.constrained.torch_ops.token_filter_torch_ops import (
    set_token_filter_torch,
)
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
    generate_token_bitmask,
)


def allocate(vocab_size, batch_size, device):
    return torch.full((batch_size, (vocab_size + 31) // 32), -1, dtype=torch.int32)


def reasoner(*, thinking=False, inner=None):
    grammar = ReasonerGrammarObject(
        inner,
        [9, 10],
        think_excluded_token_ids=[7],
        enable_token_filter=True,
        token_filter_fn=set_token_filter_torch,
        allocate_vocab_mask_fn=Mock(side_effect=allocate),
        move_vocab_mask_fn=Mock(side_effect=lambda mask, device: mask),
    )
    grammar.maybe_init_reasoning(thinking)
    return grammar


def tree():
    return GrammarTree.from_host(
        torch.tensor([[1, 2, 3, -1]], dtype=torch.int32),
        torch.tensor([[-1, -1, -1, -1]], dtype=torch.int32),
        torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
    )


class Constraint(BaseGrammarObject):
    def __init__(self):
        super().__init__()
        self.allocations = 0
        self.accepted = []

    def allocate_vocab_mask(self, vocab_size, batch_size, device):
        self.allocations += 1
        return allocate(vocab_size, batch_size, device)

    def fill_vocab_mask(self, mask, idx):
        mask[idx, 0] &= ~(1 << 5)

    def move_vocab_mask(self, mask, device):
        return mask

    def accept_token(self, token):
        self.accepted.append(token)

    def rollback(self, k):
        del self.accepted[-k:]


class IdentityGrammarMaskTest(unittest.TestCase):
    def test_no_inner_grammar_generation_skips_allocation_and_clears_old_mask(self):
        grammar = reasoner()
        info = sampling_tests._make_info(batch_size=1, grammars=[grammar])
        info.grammar_mask = object()
        info.update_regex_vocab_mask()
        self.assertIsNone(info.grammar_mask)
        grammar.allocate_vocab_mask_fn.assert_not_called()
        grammar.move_vocab_mask_fn.assert_not_called()

    def test_thinking_and_inner_constraints_do_not_get_the_identity_fast_path(self):
        for grammar in (reasoner(thinking=True), reasoner(inner=Constraint())):
            info = sampling_tests._make_info(batch_size=1, grammars=[grammar])
            info.update_regex_vocab_mask()
            self.assertIsNotNone(info.grammar_mask)
            self.assertFalse(grammar.vocab_mask_is_unconstrained)

    def test_rollback_across_thinking_end_revokes_identity_capability(self):
        grammar = reasoner(thinking=True)
        grammar.accept_token(9)
        grammar.accept_token(10)
        self.assertTrue(grammar.vocab_mask_is_unconstrained)
        grammar.rollback(1)
        self.assertFalse(grammar.vocab_mask_is_unconstrained)
        info = sampling_tests._make_info(batch_size=1, grammars=[grammar])
        info.update_regex_vocab_mask()
        self.assertIsNotNone(info.grammar_mask)

    def test_mixed_batch_keeps_exact_row_positions_and_real_constraint(self):
        noop, constrained = reasoner(), Constraint()
        info = sampling_tests._make_info(
            batch_size=3, grammars=[noop, None, constrained]
        )
        info.update_regex_vocab_mask()
        mask = info.grammar_mask.vocab_mask
        self.assertEqual(mask.shape, (3, 1))
        self.assertEqual(mask[:2, 0].tolist(), [-1, -1])
        self.assertEqual(int(mask[2, 0]) & (1 << 5), 0)
        noop.allocate_vocab_mask_fn.assert_not_called()
        self.assertEqual(constrained.allocations, 1)

    def test_all_none_members_clear_previous_mask(self):
        info = sampling_tests._make_info(batch_size=2, grammars=[None, None])
        info.grammar_mask = object()
        info.update_regex_vocab_mask()
        self.assertIsNone(info.grammar_mask)

    def test_speculative_identity_retains_original_dfs_state_bookkeeping(self):
        baseline, candidate = reasoner(), reasoner()
        original, _ = generate_token_bitmask(
            [SimpleNamespace(grammar=baseline)], *tree().resolve(), 32
        )
        self.assertTrue(torch.all(original == -1))
        info = SimpleNamespace(vocab_size=32, grammar_mask=object())
        with patch.object(
            torch.Tensor,
            "to",
            side_effect=AssertionError("Identity mask should not transfer"),
        ):
            result = build_grammar_vocab_mask(
                reqs=[SimpleNamespace(grammar=candidate)],
                tree=tree(),
                sampling_info=info,
                device="cpu",
                barrier=None,
            )
        self.assertIsNone(result)
        self.assertIsNone(info.grammar_mask)
        for field in (
            "current_token",
            "tokens_after_end",
            "tokens_in_think",
            "_thinking_match_history",
        ):
            self.assertEqual(getattr(candidate, field), getattr(baseline, field))

    def test_barrier_runs_before_deciding_if_the_mask_is_unconstrained(self):
        grammar = reasoner(thinking=True)
        barrier = Mock(side_effect=lambda: grammar.maybe_init_reasoning(False))
        info = SimpleNamespace(vocab_size=32, grammar_mask=object())
        with patch.object(
            torch.Tensor, "to", side_effect=AssertionError("No identity copy")
        ):
            result = build_grammar_vocab_mask(
                reqs=[SimpleNamespace(grammar=grammar)],
                tree=tree(),
                sampling_info=info,
                device="cpu",
                barrier=barrier,
            )
        barrier.assert_called_once()
        self.assertIsNone(result)
        self.assertIsNone(info.grammar_mask)

    def test_speculative_thinking_constraints_still_transfer_and_apply(self):
        grammar = reasoner(thinking=True)
        info = SimpleNamespace(vocab_size=32, grammar_mask=object())
        result = build_grammar_vocab_mask(
            reqs=[SimpleNamespace(grammar=grammar)],
            tree=tree(),
            sampling_info=info,
            device="cpu",
            barrier=None,
        )
        self.assertIsNotNone(result)
        self.assertEqual(int(result.vocab_mask[0, 0]) & (1 << 7), 0)
        self.assertIsNone(info.grammar_mask)

    def test_speculative_no_grammar_members_clear_old_prefill_mask(self):
        info = SimpleNamespace(vocab_size=32, grammar_mask=object())
        result = build_grammar_vocab_mask(
            reqs=[SimpleNamespace(grammar=None)],
            tree=tree(),
            sampling_info=info,
            device="cpu",
            barrier=None,
        )
        self.assertIsNone(result)
        self.assertIsNone(info.grammar_mask)


if __name__ == "__main__":
    unittest.main()
