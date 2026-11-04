import json
import os
import random
import signal
import unittest
from collections import namedtuple
from contextlib import contextmanager
from itertools import product
from unittest import mock

from transformers import AutoTokenizer

from sglang.srt.environ import envs
from sglang.srt.utils.patch_tokenizer import (
    _EncodePieceFastPathPatcher,
    _PadPaddingSideShim,
    _ParallelEncodePatcher,
    _SpecialTokensCachePatcher,
    decode_without_hf_kwargs,
    patch_tokenizer,
    unpatch_tokenizer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu", nightly=True)
register_cpu_ci(est_time=16, suite="stage-b-test-cpu-intel")


class TestPatchTokenizerEndToEndTest(unittest.TestCase):
    def test_patched_produces_same_results_as_raw(self):
        tokenizer = _load_tokenizer()
        test_texts = self._generate_test_texts(tokenizer)
        raw_results = self._run_tokenizer_ops(tokenizer, test_texts)

        _SpecialTokensCachePatcher.patch(tokenizer)
        patched_results = self._run_tokenizer_ops(tokenizer, test_texts)
        unpatch_tokenizer(tokenizer)

        self.assertEqual(raw_results, patched_results)

    @classmethod
    def _generate_test_texts(cls, tokenizer):
        special_tokens = tokenizer.all_special_tokens
        return [
            "Hello, world!",
            "This is a longer sentence with multiple words.",
            "Numbers 12345 and symbols !@#$%",
            "    leading and trailing spaces    ",
            "\n\nMultiple\n\nNewlines\n\n",
            *[f"Text with {tok} inside" for tok in special_tokens],
            " ".join(special_tokens),
            *[
                cls._random_text_from_tokens(tokenizer, num_tokens=100)
                for _ in range(5)
            ],
            *[
                cls._random_text_from_tokens(tokenizer, num_tokens=1000)
                for _ in range(3)
            ],
        ]

    @classmethod
    def _random_text_from_tokens(cls, tokenizer, num_tokens):
        token_ids = [
            random.randint(0, tokenizer.vocab_size - 1) for _ in range(num_tokens)
        ]
        return tokenizer.decode(token_ids)

    @classmethod
    def _run_tokenizer_ops(cls, tokenizer, texts):
        encode_results = [tokenizer.encode(t) for t in texts]
        batch_encode_results = tokenizer(texts)["input_ids"]
        return {
            "encode": encode_results,
            "batch_encode": batch_encode_results,
            "decode": [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in encode_results
            ],
            "batch_decode": tokenizer.batch_decode(
                encode_results, skip_special_tokens=True
            ),
            "special_tokens": tokenizer.all_special_tokens,
            "special_ids": tokenizer.all_special_ids,
        }


class TestPatchTokenizerUnitTest(unittest.TestCase):
    def test_patch_unpatch_restores_original(self):
        tokenizer = _load_tokenizer()
        cls = type(tokenizer)

        original_ids = _get_class_attr_ids(cls)

        _SpecialTokensCachePatcher.patch(tokenizer)
        self.assertTrue(getattr(cls, "_sglang_special_tokens_patched", False))

        patched_ids = _get_class_attr_ids(cls)
        changed_attrs = [
            name
            for name in original_ids
            if name in patched_ids and patched_ids[name] != original_ids[name]
        ]
        self.assertGreater(len(changed_attrs), 0, "Patch should change some attributes")

        unpatch_tokenizer(tokenizer)
        self.assertFalse(getattr(cls, "_sglang_special_tokens_patched", False))

        restored_ids = _get_class_attr_ids(cls)
        for name in original_ids:
            if name.startswith("_sglang") or name.startswith("_original"):
                continue
            self.assertEqual(
                restored_ids.get(name),
                original_ids[name],
                f"Attribute {name} should be restored to original",
            )

    def test_patch_caches_special_tokens(self):
        with _patched_tokenizer() as tokenizer:
            tokens1 = tokenizer.all_special_tokens
            ids1 = tokenizer.all_special_ids
            tokens2 = tokenizer.all_special_tokens
            ids2 = tokenizer.all_special_ids

            self.assertIs(tokens1, tokens2)
            self.assertIs(ids1, ids2)

    def test_patch_blocks_add_special_tokens(self):
        with _patched_tokenizer() as tokenizer:
            with self.assertRaises(AssertionError) as ctx:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.assertIn(
                "Cannot modify special tokens after patch", str(ctx.exception)
            )

    def test_patch_blocks_add_tokens_with_special_flag(self):
        with _patched_tokenizer() as tokenizer:
            with self.assertRaises(AssertionError) as ctx:
                tokenizer.add_tokens(["<new>"], special_tokens=True)
            self.assertIn("Cannot add special tokens after patch", str(ctx.exception))

            tokenizer.add_tokens(["<regular>"], special_tokens=False)

    def test_unpatch_clears_cache(self):
        with _patched_tokenizer() as tokenizer:
            _ = tokenizer.all_special_tokens
            _ = tokenizer.all_special_ids
            self.assertTrue(hasattr(tokenizer, "_sglang_cached_special_tokens"))
            self.assertTrue(hasattr(tokenizer, "_sglang_cached_special_ids"))

        self.assertFalse(hasattr(tokenizer, "_sglang_cached_special_tokens"))
        self.assertFalse(hasattr(tokenizer, "_sglang_cached_special_ids"))

    def test_double_patch_is_idempotent(self):
        tokenizer = _load_tokenizer()
        _SpecialTokensCachePatcher.patch(tokenizer)
        _SpecialTokensCachePatcher.patch(tokenizer)

        self.assertTrue(
            getattr(type(tokenizer), "_sglang_special_tokens_patched", False)
        )

        unpatch_tokenizer(tokenizer)

    def test_decode_without_hf_kwargs_uses_native_decode(self):
        tokenizer = _FakeDecodeTokenizer()

        self.assertEqual(
            decode_without_hf_kwargs(tokenizer, [1, 99, 2], True),
            "ab",
        )
        self.assertEqual(
            decode_without_hf_kwargs(tokenizer, [1, 99, 2], False),
            "a<special>b",
        )
        self.assertEqual(tokenizer.decode_calls, [[1, 2], [1, 99, 2]])


class TestPadPaddingSideShimUnitTest(unittest.TestCase):
    def test_patch_unpatch_restores_original(self):
        tokenizer = _FakeStaleTokenizer()
        cls = type(tokenizer)
        original_pad = cls._pad

        _PadPaddingSideShim.patch(tokenizer)
        self.addCleanup(unpatch_tokenizer, tokenizer)
        self.assertTrue(getattr(cls, "_sglang_pad_padding_side_patched", False))
        self.assertIsNot(cls._pad, original_pad)

        unpatch_tokenizer(tokenizer)
        self.assertFalse(getattr(cls, "_sglang_pad_padding_side_patched", False))
        self.assertIs(cls._pad, original_pad)

    def test_double_patch_is_idempotent(self):
        tokenizer = _FakeStaleTokenizer()
        cls = type(tokenizer)

        _PadPaddingSideShim.patch(tokenizer)
        self.addCleanup(unpatch_tokenizer, tokenizer)
        patched_once = cls._pad
        _PadPaddingSideShim.patch(tokenizer)

        self.assertIs(cls._pad, patched_once)
        self.assertTrue(getattr(cls, "_sglang_pad_padding_side_patched", False))

    def test_patch_applies_to_all_instances_of_the_class(self):
        tokenizer_a = _FakeStaleTokenizer(padding_side="left")
        tokenizer_b = _FakeStaleTokenizer(padding_side="right")
        self.assertIs(type(tokenizer_a), type(tokenizer_b))

        _PadPaddingSideShim.patch(tokenizer_a)
        try:
            self.assertTrue(
                getattr(type(tokenizer_b), "_sglang_pad_padding_side_patched", False)
            )

            tokenizer_a._pad({"input_ids": [1]})
            tokenizer_b._pad({"input_ids": [2]})

            self.assertEqual(tokenizer_a.pad_calls[-1][1], "left")
            self.assertEqual(tokenizer_b.pad_calls[-1][1], "right")
        finally:
            unpatch_tokenizer(tokenizer_a)

    def test_per_call_padding_side_override_applied_and_restored(self):
        tokenizer = _FakeStaleTokenizer(padding_side="right")
        _PadPaddingSideShim.patch(tokenizer)
        try:
            tokenizer._pad({"input_ids": [1]}, padding_side="left")
            self.assertEqual(tokenizer.pad_calls[-1][1], "left")
            self.assertEqual(tokenizer.padding_side, "right")
        finally:
            unpatch_tokenizer(tokenizer)

    def test_padding_side_override_matching_default_is_noop(self):
        tokenizer = _FakeStaleTokenizer(padding_side="right")
        _PadPaddingSideShim.patch(tokenizer)
        try:
            tokenizer._pad({"input_ids": [1]}, padding_side="right")
            self.assertEqual(tokenizer.pad_calls[-1][1], "right")
            self.assertEqual(tokenizer.padding_side, "right")
        finally:
            unpatch_tokenizer(tokenizer)

    def test_none_padding_side_passes_through(self):
        tokenizer = _FakeStaleTokenizer(padding_side="right")
        _PadPaddingSideShim.patch(tokenizer)
        try:
            tokenizer._pad({"input_ids": [1]})
            self.assertEqual(tokenizer.pad_calls[-1][1], "right")
        finally:
            unpatch_tokenizer(tokenizer)

    def test_padding_side_restored_after_pad_raises(self):
        tokenizer = _FakeRaisingTokenizer(padding_side="right")
        _PadPaddingSideShim.patch(tokenizer)
        try:
            with self.assertRaises(RuntimeError):
                tokenizer._pad(padding_side="left")
            self.assertEqual(tokenizer.padding_side, "right")
        finally:
            unpatch_tokenizer(tokenizer)


class _FakeStaleTokenizer:
    """Stand-in for a custom remote-code tokenizer whose _pad() predates
    transformers forwarding padding_side -- no padding_side param, no **kwargs."""

    def __init__(self, padding_side="right"):
        self.padding_side = padding_side
        self.pad_calls = []

    def _pad(self, encoded_inputs, max_length=None):
        self.pad_calls.append((encoded_inputs, self.padding_side))
        return encoded_inputs


class _FakeRaisingTokenizer:
    def __init__(self, padding_side="right"):
        self.padding_side = padding_side

    def _pad(self, *args, **kwargs):
        raise RuntimeError("boom")


class TestEncodePieceFastPathPatcher(CustomTestCase):
    """The fast path must be a pure shortcut: every segment shape it handles
    (or declines) has to encode to the same ids as the original method."""

    def test_encode_piece_matches_original_on_every_segment_shape(self):
        tokenizer = _load_k3_tokenizer()
        specials = list(tokenizer.special_tokens)
        rng = random.Random(0)
        random_texts = [
            _random_text_from_tokens(tokenizer, num_tokens=n, rng=rng)
            for n in (1, 7, 100, 1000)
        ]
        segments = [
            # exactly one special token -> table lookup
            *[(tok, True) for tok in specials],
            # special token with a suffix / two specials -> original path
            *[(tok + "x", True) for tok in specials[:8]],
            (specials[0] + specials[1], True),
            # plain text -> encode_ordinary
            *[(text, False) for text in random_texts],
            ("", False),
            ("", True),
            (" " * 40, False),
            # special literal inside plain text -> original path
            *[(f"user wrote {tok} literally", False) for tok in specials[:8]],
            # lone surrogate (json.loads('"\\ud83d"') in a tool result) ->
            # tiktoken's UnicodeEncodeError fix-up must still apply
            ("tool output \ud83d broken", False),
            ("tool output \ud83d broken", True),
            # longer than MAX_NO_WHITESPACES_CHARS -> original splitter path
            ("b" * 30_000 + " tail", False),
            ("b" * 30_000 + " tail", True),
        ]
        # Exercise both sides of the exact splitter shortcut boundary. Include
        # whitespace runs as well as the no-whitespace limit the guard mirrors.
        for size in (24_999, 25_000, 25_001):
            for text in ("a" * size, " " * size, "\n" * size):
                segments.extend((text, allow) for allow in (False, True))
        for text in (
            "中文工具结果 café e\u0301 👩\u200d💻 العربية",
            "\ud800",
            "\udfff",
            "paired \ud83d\ude00 and lone \ud83d",
            '\x00\t\r\n\\"',
            *[f"{tok}prefix{tok}" for tok in specials[:8]],
        ):
            segments.extend((text, allow) for allow in (False, True))
        original = type(tokenizer)._encode_text_piece
        _EncodePieceFastPathPatcher.patch(tokenizer)
        try:
            for text, allow_special in segments:
                self.assertEqual(
                    original(tokenizer, text, allow_special),
                    tokenizer._encode_text_piece(text, allow_special),
                    (text[:40], allow_special),
                )
        finally:
            _EncodePieceFastPathPatcher.unpatch(tokenizer)

    def test_chat_template_ids_unchanged_for_tool_call_conversation(self):
        # Kimi-K3's encoding_k3 renders one segment per control token or tag
        # name and four text segments per tool-call attribute; this is the
        # shape the fast path exists for.
        tokenizer = _load_k3_tokenizer()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            }
        ]
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "deploy it"},
        ]
        for i in range(300):
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": "Bash",
                                "arguments": json.dumps(
                                    {
                                        "command": f'printf "中文 café 👩‍💻"; ls /tmp/{i}'
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "content": (
                        f"total 0\n<|im_end|> literal in tool output {i}\n"
                        "中文 café e\u0301 👩\u200d💻 lone surrogate \ud83d\n"
                        + " ".join(tokenizer.special_tokens)
                    ),
                }
            )
        kwargs = dict(tools=tools, tokenize=True, add_generation_prompt=True)
        expected = tokenizer.apply_chat_template(messages, **kwargs)

        _EncodePieceFastPathPatcher.patch(tokenizer)
        try:
            self.assertEqual(
                expected, tokenizer.apply_chat_template(messages, **kwargs)
            )
        finally:
            _EncodePieceFastPathPatcher.unpatch(tokenizer)

    def test_unpatch_restores_encode_piece(self):
        tokenizer = _load_k3_tokenizer()
        cls = type(tokenizer)
        original = cls._encode_text_piece

        _EncodePieceFastPathPatcher.patch(tokenizer)
        self.assertIsNot(cls._encode_text_piece, original)
        _EncodePieceFastPathPatcher.patch(tokenizer)  # idempotent
        _EncodePieceFastPathPatcher.unpatch(tokenizer)

        self.assertIs(cls._encode_text_piece, original)
        self.assertFalse(hasattr(cls, "_original_encode_text_piece"))
        self.assertFalse(hasattr(tokenizer, "_sglang_special_literal_regex"))

    def test_k2_tokenizer_without_encode_text_piece_is_skipped(self):
        # The K2 family shares the TikTokenTokenizer class/module name but
        # inlines the segment loop into encode(); patch_tokenizer must still
        # apply the special-tokens cache and skip the encode-piece fast path.
        tokenizer = _load_tokenizer()
        cls = type(tokenizer)
        self.assertFalse(hasattr(cls, "_encode_text_piece"))
        self.assertFalse(_EncodePieceFastPathPatcher.applies_to(tokenizer))

        expected = tokenizer.encode("hello <|im_end|> world")
        with envs.SGLANG_PATCH_TOKENIZER.override(True):
            patched = patch_tokenizer(tokenizer)
        try:
            self.assertIs(patched, tokenizer)
            self.assertTrue(getattr(cls, "_sglang_special_tokens_patched", False))
            self.assertFalse(getattr(cls, "_sglang_encode_piece_patched", False))
            self.assertEqual(patched.encode("hello <|im_end|> world"), expected)
        finally:
            unpatch_tokenizer(tokenizer)


class TestParallelEncodePatcher(CustomTestCase):
    """The pool must return the checkpoint's serial ids on every input."""

    def test_ids_match_serial_at_every_eligible_cut(self):
        tokenizer = _load_k3_tokenizer()
        cls = type(tokenizer)
        self.assertFalse(hasattr(cls, _ParallelEncodePatcher._PATCHED_FLAG))
        specials = list(tokenizer.special_tokens)[:4]
        filler = "alpha beta\ngamma, delta! " * 20_000
        texts = [
            *specials,
            *[
                f"x{a}{mid}{b}y z"
                for a in " .'\r1"
                for mid in ("\n", "\n\n", "\r\n", " \n", "\n ", "\n \n", "\n\t\n")
                for b in (
                    "a",
                    "A",
                    "1",
                    ".",
                    "'s",
                    "<",
                    "\xa0",
                    "\x85",
                    "中",
                    *specials,
                )
            ],
            *[s[:i] + "\n" + s[i:] for s in specials for i in range(1, len(s))],
            "a\ud83d\nb\ude00c",
            "👩\u200d💻\nx\u0301中\nA文",
            *[
                c * n + "\nend"
                for c in " \n\xa0\x1ca"
                for n in (12_499, 24_998, 25_000, 25_001, 60_000)
            ],
            "x " * 6_000 + "a" * 30_000 + " y" * 6_000,
            filler,
            filler[: 400_000 - 3] + specials[0] + filler[:100],
            "a" * 1_000_000,
        ]
        serial = {
            (text, allow): cls._encode_text_piece(tokenizer, text, allow)
            for text in texts
            for allow in (True, False)
        }
        with envs.SGLANG_PATCH_TOKENIZER.override(True):
            patch_tokenizer(tokenizer)
        try:
            self.assertTrue(getattr(cls, _ParallelEncodePatcher._PATCHED_FLAG))
            with (
                envs.SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS.override(0),
                mock.patch.object(_ParallelEncodePatcher, "_POOL_MIN_CHARS", 1),
            ):
                for task_chars in (1, 32_768):
                    with mock.patch.object(
                        _ParallelEncodePatcher, "_TASK_CHARS", task_chars
                    ):
                        for (text, allow), ids in serial.items():
                            self.assertEqual(
                                ids,
                                tokenizer._encode_text_piece(text, allow),
                                (len(text), allow, task_chars),
                            )
        finally:
            unpatch_tokenizer(tokenizer)
        self.assertFalse(hasattr(cls, "_parallel_original_methods"))

    def test_run_check_never_passes_text_the_splitter_would_cut(self):
        split = type(_load_k3_tokenizer())._split_whitespaces_or_nonwhitespaces
        for limit in range(2, 7):
            with mock.patch.object(_ParallelEncodePatcher, "_MAX_RUN_CHARS", limit):
                for n in range(1, 13):
                    for bits in product("a ", repeat=n):
                        text = "".join(bits)
                        if not _ParallelEncodePatcher._runs_may_exceed_limit(text):
                            self.assertEqual(list(split(text, limit)), [text])

    def test_chat_template_ids_unchanged(self):
        tokenizer = _load_k3_tokenizer()
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "<|im_end|> line\n" * 20_000},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "中文 😀\nnext"},
        ]
        kwargs = dict(tokenize=True, add_generation_prompt=True)
        expected = tokenizer.apply_chat_template(messages, **kwargs)
        segments = list(_chat_segments(tokenizer, messages))
        _ParallelEncodePatcher.patch(tokenizer)
        try:
            with mock.patch.object(
                _ParallelEncodePatcher,
                "_encode_piece",
                wraps=_ParallelEncodePatcher._encode_piece,
            ) as encode_piece:
                self.assertEqual(
                    expected, tokenizer.apply_chat_template(messages, **kwargs)
                )
                encode_piece.assert_called()
                encode_piece.reset_mock()
                self.assertEqual(
                    expected, tokenizer._encode_chat_segments(iter(segments))
                )
                encode_piece.assert_called()
                encode_piece.reset_mock()
                with envs.SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS.override(10**9):
                    self.assertEqual(
                        expected, tokenizer.apply_chat_template(messages, **kwargs)
                    )
                encode_piece.assert_not_called()
        finally:
            _ParallelEncodePatcher.unpatch(tokenizer)

    def test_pool_takes_only_long_text(self):
        tokenizer = _load_k3_tokenizer()
        Piece = namedtuple("Piece", "text allow_special")
        _ParallelEncodePatcher.patch(tokenizer)
        try:
            with mock.patch.object(
                _ParallelEncodePatcher,
                "_encode_piece",
                wraps=_ParallelEncodePatcher._encode_piece,
            ) as encode_piece:
                for called, encode in (
                    (False, lambda: tokenizer.encode("a b\n" * 8191)),
                    (True, lambda: tokenizer.encode("a b\n" * 8192)),
                    (
                        False,
                        lambda: tokenizer._encode_chat_segments(
                            [Piece("hello ", False)] * 6_000
                        ),
                    ),
                ):
                    encode()
                    self.assertEqual(encode_piece.called, called)
                    encode_piece.reset_mock()
                with envs.SGLANG_PARALLEL_PROMPT_ENCODE.override(False):
                    tokenizer.encode("a b\n" * 8192)
                encode_piece.assert_not_called()
                self.assertEqual(tokenizer._encode_chat_segments([]), [])
                self.assertEqual(tokenizer.encode(""), [])
        finally:
            _ParallelEncodePatcher.unpatch(tokenizer)

    def test_forked_child_encodes(self):
        tokenizer = _load_k3_tokenizer()
        text = "line\n" * 20_000
        expected = tokenizer.encode(text)
        _ParallelEncodePatcher.patch(tokenizer)
        try:
            self.assertEqual(tokenizer.encode(text), expected)
            pid = os.fork()
            if pid == 0:
                status = 1
                try:
                    signal.alarm(60)
                    status = 0 if tokenizer.encode(text) == expected else 1
                finally:
                    os._exit(status)
            self.assertEqual(os.waitpid(pid, 0)[1], 0)
        finally:
            _ParallelEncodePatcher.unpatch(tokenizer)

    def test_another_checkpoint_or_the_gate_keeps_the_serial_path(self):
        tokenizer = _load_k3_tokenizer()
        self.assertTrue(_ParallelEncodePatcher.applies_to(tokenizer))
        with mock.patch.object(_ParallelEncodePatcher, "_SOURCE_SHA256", "other"):
            self.assertFalse(_ParallelEncodePatcher.applies_to(tokenizer))
        with mock.patch.object(tokenizer, "special_tokens", {"a\nb": 1}):
            self.assertFalse(_ParallelEncodePatcher.applies_to(tokenizer))
        with envs.SGLANG_PARALLEL_PROMPT_ENCODE.override(False):
            self.assertFalse(_ParallelEncodePatcher._worthwhile(10**9))


def _chat_segments(tokenizer, messages):
    template = type(tokenizer).apply_chat_template
    build = template.__globals__["build_chat_segments"]
    return build(messages, add_generation_prompt=True, thinking_effort="max")


class TestProcessorTokenizerIsPatched(CustomTestCase):
    def test_tokenizer_manager_patches_the_processor_tokenizer(self):
        from sglang.srt.managers import tokenizer_manager

        tokenizer = _load_k3_tokenizer()
        manager = mock.Mock()
        with (
            mock.patch.object(tokenizer_manager, "get_disagg"),
            mock.patch.object(tokenizer_manager, "get_serving") as get_serving,
            mock.patch.object(tokenizer_manager, "import_processors"),
            mock.patch.object(tokenizer_manager, "get_mm_processor"),
            mock.patch.object(tokenizer_manager, "determine_tensor_transport_mode"),
            mock.patch.object(
                tokenizer_manager,
                "get_processor_wrapper",
                return_value=mock.Mock(spec=["tokenizer"], tokenizer=tokenizer),
            ),
        ):
            manager.model_config.is_multimodal = True
            get_serving.return_value.skip_tokenizer_init = False
            tokenizer_manager.get_disagg.return_value.language_model_only = False
            tokenizer_manager.TokenizerManager.init_tokenizer_and_processor(manager)
        try:
            self.assertIs(manager.tokenizer, tokenizer)
            self.assertTrue(type(tokenizer)._sglang_encode_piece_patched)
        finally:
            unpatch_tokenizer(tokenizer)


def _random_text_from_tokens(tokenizer, num_tokens, rng):
    token_ids = [rng.randint(0, tokenizer.vocab_size - 1) for _ in range(num_tokens)]
    return tokenizer.decode(token_ids)


def _get_class_attr_ids(cls):
    return {
        n: id(v.fget if isinstance(v, property) else v) for n, v in vars(cls).items()
    }


def _load_tokenizer():
    # The slowness is mainly observed in Kimi
    return AutoTokenizer.from_pretrained(
        "nvidia/Kimi-K2-Thinking-NVFP4", trust_remote_code=True
    )


def _load_k3_tokenizer():
    # Only Kimi-K3's tokenization_kimi.py has _encode_text_piece / encoding_k3.
    local_path = os.environ.get("SGLANG_TEST_K3_TOKENIZER_PATH")
    return AutoTokenizer.from_pretrained(
        local_path or "moonshotai/Kimi-K3",
        revision="f831ab66814297da540d832a5235f8e904f29d06",
        trust_remote_code=True,
        local_files_only=bool(local_path),
    )


class TestEncodePieceFastPathBoundaries(unittest.TestCase):
    """Dependency-free branch guards; real tiktoken equivalence lives above."""

    def test_shortcut_and_fallback_branches_preserve_ids(self):
        class Model:
            def encode_ordinary(self, text):
                # Mirrors the public tiktoken surrogate fix-up boundary, not BPE.
                normalized = text.encode("utf-16", "surrogatepass").decode(
                    "utf-16", "replace"
                )
                return list(normalized.encode("utf-8"))

        class Tokenizer:
            special_tokens = {"<|im_end|>": 0, "[.*]": 257}

            def __init__(self):
                self.model = Model()
                self.fallback_calls = []

            def _encode_text_piece(self, text, allow_special_tokens=True):
                self.fallback_calls.append((text, allow_special_tokens))
                if allow_special_tokens and text in self.special_tokens:
                    return [self.special_tokens[text]]
                return self.model.encode_ordinary(text)

        tokenizer = Tokenizer()
        original = Tokenizer._encode_text_piece
        cases = [
            ("<|im_end|>", True, False),  # zero-valued special ID is valid
            ("[.*]", True, False),
            ("x[.*]y", False, True),  # regex literals must be escaped
            ("ordinary [xx] text", False, False),
            ("<|im_end|><|im_end|>", True, True),
            ("prefix<|im_end|>suffix", False, True),
            ("", False, False),
            ("", True, True),
            ("中文 👩\u200d💻 \ud800", False, False),
        ]
        cases += [
            ("a" * size, False, size > 25_000) for size in (24_999, 25_000, 25_001)
        ]
        _EncodePieceFastPathPatcher.patch(tokenizer)
        try:
            for text, allow, fallback in cases:
                expected = original(tokenizer, text, allow)
                tokenizer.fallback_calls.clear()
                self.assertEqual(expected, tokenizer._encode_text_piece(text, allow))
                self.assertEqual(bool(tokenizer.fallback_calls), fallback)
        finally:
            _EncodePieceFastPathPatcher.unpatch(tokenizer)
        self.assertIs(Tokenizer._encode_text_piece, original)

    def test_k2_shape_is_untouched(self):
        class WithoutEncodePiece:
            pass

        tokenizer = WithoutEncodePiece()
        self.assertFalse(_EncodePieceFastPathPatcher.applies_to(tokenizer))
        self.assertIs(_EncodePieceFastPathPatcher.patch(tokenizer), tokenizer)
        self.assertFalse(hasattr(WithoutEncodePiece, "_sglang_encode_piece_patched"))


@contextmanager
def _patched_tokenizer():
    tokenizer = _load_tokenizer()
    _SpecialTokensCachePatcher.patch(tokenizer)
    try:
        yield tokenizer
    finally:
        unpatch_tokenizer(tokenizer)


class _FakeDecodeTokenizer:
    all_special_ids_set = {99}

    def __init__(self):
        self.decode_calls = []

    def decode(self, token_ids):
        token_ids = list(token_ids)
        self.decode_calls.append(token_ids)
        token_text = {1: "a", 2: "b", 99: "<special>"}
        return "".join(token_text[token_id] for token_id in token_ids)


if __name__ == "__main__":
    unittest.main()
