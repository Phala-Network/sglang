"""CPU tests of the installed Xing4 JSON reasoning/EOS wrapper."""
import types
import unittest

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject as Base
from sglang.srt.constrained.reasoner_grammar_backend import (
    ReasonerGrammarBackend as Backend,
    ReasonerGrammarObject,
)
from sglang.srt.parser.reasoning_parser import Xing4_0Detector as Xing4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1, "base-a-test-cpu")


class Inner(Base):
    """A small deterministic body grammar: token 4, then EOS 2."""
    def __init__(self):
        super().__init__()
        self.tokens = []

    def fill_vocab_mask(self, mask, idx):
        mask[idx][:] = [False] * len(mask[idx])
        mask[idx][4 if not self.tokens else 2] = True

    def accept_token(self, token):
        assert token == (4 if not self.tokens else 2)
        self.tokens.append(token)

    def rollback(self, count):
        del self.tokens[-count:]

    def copy(self):
        result = Inner()
        result.tokens = list(self.tokens)
        return result

    def is_terminated(self):
        return self.tokens == [4, 2]


class FakeBackend:
    is_support_token_filter = True
    allocate_vocab_mask = move_vocab_mask = apply_vocab_mask = None

    def _init_value_dispatch(self, key, reasoning):
        return Inner()

    @staticmethod
    def set_token_filter(mask, ids, idx, is_allowed=True):
        mask[idx][:] = [not is_allowed] * len(mask[idx])
        for token in ids:
            mask[idx][token] = is_allowed


def backend(xing=True, strict=False, stops=None, supported=True):
    impl = FakeBackend()
    impl.is_support_token_filter = supported
    detector = Xing4() if xing else types.SimpleNamespace(think_end_token="</think>", think_excluded_tokens=[])
    tokenizer = types.SimpleNamespace(eos_token_id=2, additional_stop_token_ids=[3] if stops is None else stops, encode=lambda *a, **k: [10])
    return Backend(impl, types.SimpleNamespace(detector=detector), tokenizer, strict)


def mask(obj, vocab_size=16):
    result = [[True] * vocab_size]
    obj.fill_vocab_mask(result, 0)
    return result[0]


class TestXing4JSONReasoningStop(unittest.TestCase):
    def test_eos_blocked_until_json_body_complete(self):
        obj = backend()._init_value_dispatch(("json", '{"type":"object"}'), True)
        self.assertFalse(mask(obj)[2])
        self.assertFalse(mask(obj)[3])
        self.assertTrue(mask(obj)[10])
        obj.accept_token(7)
        self.assertEqual(obj.grammar.tokens, [])
        obj.accept_token(10)
        self.assertFalse(mask(obj)[2])
        self.assertTrue(mask(obj)[4])
        obj.accept_token(4)
        self.assertTrue(mask(obj)[2])
        obj.accept_token(2)
        self.assertTrue(obj.is_terminated())

    def test_scope_and_nonreasoning(self):
        for xing, key in ((False, "json"), (True, "regex"), (True, "ebnf")):
            with self.subTest(xing=xing, key=key):
                obj = backend(xing)._init_value_dispatch((key, "x"), True)
                self.assertTrue(mask(obj)[2])
        self.assertIsNone(backend().init_strict_reasoning_grammar(True))
        obj = backend()._init_value_dispatch(("json", "{}"), False)
        self.assertEqual(obj.think_stop_token_ids, [2, 3])
        self.assertTrue(mask(obj)[4])
        cached = obj.copy()
        cached.maybe_init_reasoning(True)
        self.assertFalse(mask(cached)[2])
        self.assertFalse(mask(cached)[3])
        cached.maybe_init_reasoning(False)
        self.assertTrue(mask(cached)[4])
        unsupported = backend(supported=False)._init_value_dispatch(("json", "{}"), False)
        with self.assertRaises(ValueError):
            unsupported.copy().maybe_init_reasoning(True)

    def test_copy_and_rollback(self):
        obj = backend()._init_value_dispatch(("json", "{}"), True)
        obj.accept_token(10)
        clone = obj.copy()
        clone.accept_token(4)
        self.assertTrue(mask(clone)[2])
        clone.rollback(2)
        self.assertFalse(mask(clone)[2])
        self.assertTrue(mask(clone)[10])
        self.assertEqual(clone.grammar.tokens, [])
        self.assertTrue(mask(obj)[4])

    def test_strict_min_max_preserved(self):
        obj = backend(strict=True)._init_value_dispatch(("json", "{}"), True)
        obj.enable_token_filter = True
        obj.min_think_tokens = 1
        self.assertFalse(mask(obj)[10])
        self.assertFalse(mask(obj)[2])
        obj.accept_token(7)
        obj.max_think_tokens = 1
        self.assertEqual([i for i, allowed in enumerate(mask(obj)) if allowed], [10])

    def test_fail_closed_unsupported_or_overlapping_stop(self):
        for kwargs in ({"supported": False}, {"stops": [10]}, {"stops": [-1]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                backend(**kwargs)._init_value_dispatch(("json", "{}"), True)
        obj = backend()._init_value_dispatch(("json", "{}"), True)
        with self.assertRaises(ValueError):
            obj.set_request_think_end_ids([2])

    def test_real_xgrammar_json_control(self):
        import xgrammar as xgr
        info = xgr.TokenizerInfo(
            [bytes([i]) for i in range(256)] + [b"<eos>", b"</think>"],
            vocab_type=xgr.VocabType.RAW,
            stop_token_ids=[256],
        )
        compiler = xgr.GrammarCompiler(info)
        matcher = xgr.GrammarMatcher(
            compiler.compile_json_schema('{"type":"object"}')
        )

        class RealInner(Inner):
            def fill_vocab_mask(self, target, idx):
                bitmask = xgr.allocate_token_bitmask(1, 258)
                matcher.fill_next_token_bitmask(bitmask, 0)
                target[idx][:] = [
                    bool(int(bitmask[0, token // 32]) & (1 << (token % 32)))
                    for token in range(258)
                ]

            def accept_token(self, token):
                assert matcher.accept_token(token)

            def is_terminated(self):
                return matcher.is_terminated()

        obj = ReasonerGrammarObject(
            RealInner(), [257], think_stop_token_ids=[256],
            token_filter_fn=FakeBackend.set_token_filter,
        )
        obj.maybe_init_reasoning(True)
        self.assertFalse(mask(obj, 258)[256])
        self.assertTrue(mask(obj, 258)[257])
        obj.accept_token(257)
        self.assertFalse(mask(obj, 258)[256])
        for token in b'{"ok":true}':
            self.assertTrue(mask(obj, 258)[token])
            obj.accept_token(token)
        self.assertTrue(mask(obj, 258)[256])
        obj.accept_token(256)
        self.assertTrue(obj.is_terminated())


if __name__ == "__main__":
    unittest.main()
