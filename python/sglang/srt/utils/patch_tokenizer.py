import hashlib
import inspect
import logging
import os
import re
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from math import inf
from typing import List

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def patch_tokenizer(tokenizer):
    if not envs.SGLANG_PATCH_TOKENIZER.get():
        return tokenizer

    if _is_kimi_tiktoken_tokenizer(tokenizer):
        logger.info(
            f"Applying special tokens cache patch for Kimi tokenizer: {type(tokenizer)}"
        )
        _SpecialTokensCachePatcher.patch(tokenizer)
        if _EncodePieceFastPathPatcher.applies_to(tokenizer):
            logger.info(
                f"Applying encode-piece fast path patch for Kimi tokenizer: {type(tokenizer)}"
            )
            _EncodePieceFastPathPatcher.patch(tokenizer)
            if _ParallelEncodePatcher.applies_to(tokenizer):
                logger.info(
                    f"Applying parallel encode patch for Kimi tokenizer: {type(tokenizer)}"
                )
                _ParallelEncodePatcher.patch(tokenizer)
            else:
                logger.warning(
                    "Parallel encode stays off for this Kimi tokenizer: its source "
                    "is not pinned, so long prompts encode serially: %s",
                    type(tokenizer),
                )

    if _needs_pad_padding_side_shim(tokenizer):
        logger.info(
            f"Applying _pad(padding_side=...) compat shim for {type(tokenizer)}"
        )
        return _PadPaddingSideShim.patch(tokenizer)

    return tokenizer


def _needs_pad_padding_side_shim(tokenizer) -> bool:
    pad_fn = getattr(type(tokenizer), "_pad", None)
    if pad_fn is None:
        return False
    try:
        sig = inspect.signature(pad_fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "padding_side" in params:
        return False
    return not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def unpatch_tokenizer(tokenizer):
    tokenizer = _PadPaddingSideShim.unpatch(tokenizer)
    _ParallelEncodePatcher.unpatch(tokenizer)
    _EncodePieceFastPathPatcher.unpatch(tokenizer)
    return _SpecialTokensCachePatcher.unpatch(tokenizer)


def _is_kimi_tiktoken_tokenizer(tokenizer):
    cls = type(tokenizer)
    class_name = cls.__name__
    module_name = cls.__module__ or ""
    return class_name == "TikTokenTokenizer" and "tokenization_kimi" in module_name


def decode_without_hf_kwargs(tokenizer, token_ids, skip_special_tokens):
    if skip_special_tokens:
        special_ids = getattr(tokenizer, "all_special_ids_set", None)
        if special_ids is None:
            special_ids = getattr(tokenizer, "all_special_ids", None)
        if special_ids is not None:
            special_ids_set = set(special_ids)
            token_ids = [tid for tid in token_ids if tid not in special_ids_set]
    return tokenizer.decode(token_ids)


class _PadPaddingSideShim:
    _PATCHED_FLAG = "_sglang_pad_padding_side_patched"

    @classmethod
    def patch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        original_pad = tokenizer_cls._pad

        def patched_pad(self, *args, **kwargs):
            padding_side = kwargs.pop("padding_side", None)
            if padding_side is None or padding_side == self.padding_side:
                return original_pad(self, *args, **kwargs)
            previous_padding_side = self.padding_side
            self.padding_side = padding_side
            try:
                return original_pad(self, *args, **kwargs)
            finally:
                self.padding_side = previous_padding_side

        tokenizer_cls._original_pad = original_pad
        tokenizer_cls._pad = patched_pad
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)

        return tokenizer

    @classmethod
    def unpatch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if not getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls._pad = tokenizer_cls._original_pad
        del tokenizer_cls._original_pad
        delattr(tokenizer_cls, cls._PATCHED_FLAG)

        return tokenizer


class _SpecialTokensCachePatcher:
    _PATCHED_FLAG = "_sglang_special_tokens_patched"
    _CACHED_TOKENS_ATTR = "_sglang_cached_special_tokens"
    _CACHED_IDS_ATTR = "_sglang_cached_special_ids"

    @classmethod
    def patch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls._original_all_special_tokens = (
            tokenizer_cls.all_special_tokens.fget
        )
        tokenizer_cls._original_all_special_ids = tokenizer_cls.all_special_ids.fget
        tokenizer_cls._original_add_special_tokens = tokenizer_cls.add_special_tokens
        tokenizer_cls._original_add_tokens = tokenizer_cls.add_tokens

        patched_all_special_tokens = _make_cached_property(
            cls._CACHED_TOKENS_ATTR, tokenizer_cls._original_all_special_tokens
        )
        patched_all_special_ids = _make_cached_property(
            cls._CACHED_IDS_ATTR, tokenizer_cls._original_all_special_ids
        )

        def patched_add_special_tokens(self, *args, **kwargs):
            assert False, (
                "Cannot modify special tokens after patch. Call unpatch_tokenizer first."
            )

        def patched_add_tokens(self, new_tokens, special_tokens=False):
            assert not special_tokens, (
                "Cannot add special tokens after patch. Call unpatch_tokenizer first."
            )
            return tokenizer_cls._original_add_tokens(
                self, new_tokens, special_tokens=False
            )

        tokenizer_cls.all_special_tokens = patched_all_special_tokens
        tokenizer_cls.all_special_ids = patched_all_special_ids
        tokenizer_cls.add_special_tokens = patched_add_special_tokens
        tokenizer_cls.add_tokens = patched_add_tokens
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)

        return tokenizer

    @classmethod
    def unpatch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if not getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls.all_special_tokens = property(
            tokenizer_cls._original_all_special_tokens
        )
        tokenizer_cls.all_special_ids = property(
            tokenizer_cls._original_all_special_ids
        )
        tokenizer_cls.add_special_tokens = tokenizer_cls._original_add_special_tokens
        tokenizer_cls.add_tokens = tokenizer_cls._original_add_tokens

        del tokenizer_cls._original_all_special_tokens
        del tokenizer_cls._original_all_special_ids
        del tokenizer_cls._original_add_special_tokens
        del tokenizer_cls._original_add_tokens
        delattr(tokenizer_cls, cls._PATCHED_FLAG)

        for attr in [cls._CACHED_TOKENS_ATTR, cls._CACHED_IDS_ATTR]:
            if hasattr(tokenizer, attr):
                delattr(tokenizer, attr)

        logger.info(f"Unpatched special tokens cache for {tokenizer_cls.__name__}")
        return tokenizer


class _EncodePieceFastPathPatcher:
    """Short-circuit ``TikTokenTokenizer._encode_text_piece`` for the two segment
    shapes that dominate Kimi-K3 chat encoding.

    ``encoding_k3.build_chat_segments`` renders a conversation into tens of
    thousands of tiny segments -- one per control token or tag name, and four
    text segments per tool-call attribute (`` key``, ``="``, value, ``"``) --
    and ``_encode_text_piece`` is called once per segment.  Every call first
    runs a pure-Python per-character splitter over the segment (control
    segments included); it yields the input unchanged below
    ``MAX_NO_WHITESPACES_CHARS`` but still costs O(len) Python per call.  Then:

    * control segments call ``tiktoken.Encoding.encode(allowed_special="all")``,
      which passes the cached 256-entry special-token set through the
      Python/Rust boundary on every call -- a fixed ~15-30us per call that
      dominates for tiny segments -- for what is a dictionary lookup;
    * text segments call ``encode(disallowed_special=())``, which for text
      with no special-token literal is exactly ``encode_ordinary``.

    The patched method keeps the original as the fallback, so token ids are
    unchanged: a special-token literal inside a text segment, a control segment
    that is not exactly one special token, and long text all take the original
    path.

    Only the Kimi-K3 tokenizer has ``_encode_text_piece``; the K2 family
    inlines the same loop into ``encode``, so ``applies_to`` must be checked
    before ``patch``.
    """

    _PATCHED_FLAG = "_sglang_encode_piece_patched"
    # Mirrors MAX_NO_WHITESPACES_CHARS in tokenization_kimi.py: below this length
    # the original splitter yields the input unchanged, so skipping it is exact.
    _MAX_UNSPLIT_TEXT_CHARS = 25_000

    @classmethod
    def applies_to(cls, tokenizer) -> bool:
        return callable(getattr(type(tokenizer), "_encode_text_piece", None))

    @classmethod
    def patch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer
        if not cls.applies_to(tokenizer):
            logger.info(
                f"Skipping encode-piece fast path: {tokenizer_cls.__name__} has no _encode_text_piece"
            )
            return tokenizer

        original_encode_text_piece = tokenizer_cls._encode_text_piece
        max_unsplit_text_chars = cls._MAX_UNSPLIT_TEXT_CHARS

        def patched_encode_text_piece(
            self, text: str, allow_special_tokens: bool = True
        ) -> List[int]:
            if allow_special_tokens:
                special_id = self.special_tokens.get(text)
                if special_id is not None:
                    return [special_id]
                return original_encode_text_piece(self, text, allow_special_tokens)
            if len(text) <= max_unsplit_text_chars and not _special_literal_regex(
                self
            ).search(text):
                # disallowed_special=() encodes special literals as plain text,
                # so with none present encode() == encode_ordinary().  Go through
                # the public method, not _core_bpe, to keep tiktoken's
                # UnicodeEncodeError fix-up for lone surrogates in the text.
                return self.model.encode_ordinary(text)
            return original_encode_text_piece(self, text, allow_special_tokens)

        tokenizer_cls._original_encode_text_piece = original_encode_text_piece
        tokenizer_cls._encode_text_piece = patched_encode_text_piece
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)
        return tokenizer

    @classmethod
    def unpatch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if not getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls._encode_text_piece = tokenizer_cls._original_encode_text_piece
        del tokenizer_cls._original_encode_text_piece
        delattr(tokenizer_cls, cls._PATCHED_FLAG)
        if hasattr(tokenizer, _SPECIAL_LITERAL_REGEX_ATTR):
            delattr(tokenizer, _SPECIAL_LITERAL_REGEX_ATTR)

        logger.info(f"Unpatched encode-piece fast path for {tokenizer_cls.__name__}")
        return tokenizer


_TextPiece = namedtuple("_TextPiece", "text allow_special")


@cache
def _encode_pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=8, thread_name_prefix="k3-encode")


# A pool's threads do not survive a fork.
os.register_at_fork(after_in_child=_encode_pool.cache_clear)


class _ParallelEncodePatcher:
    """Encode long Kimi-K3 prompts on a bounded thread pool; tiktoken releases the GIL.

    The ids equal the checkpoint's serial encoding because it already encodes each
    chat segment, each 400K-char slice and each splitter output on its own, and the
    only further cut is after a newline followed by printable ASCII. For the
    checkpoint's ``pat_str`` a match holding a newline stops at ``[\\r\\n]*``,
    ``[\\r\\n]+``, so no match crosses that point, and the pattern has no
    lookbehind, so the text after it matches the same on its own. The one lookahead,
    ``\\s+(?!\\S)``, is never tried on a whitespace run holding a newline: ``\\s*[\\r\\n]+``
    comes first. No special token contains a newline, so none spans the cut either.

    ``applies_to`` pins the tokenizer class by its source, so any other checkpoint
    keeps the serial path.
    """

    _PATCHED_FLAG = "_sglang_parallel_encode_patched"
    # K3, then K3.1, whose class differs only in formatting (same AST and tiktoken.model).
    _SOURCE_SHA256 = (
        "9c2c234cc5936cb5501097a23ffeab377650ed56d6d289502211514dccee808b",
        "c8b53154e0d930d6a084db2c92a92ad3d294b9ee287674a2b20565eacad6c03e",
    )
    # Mirror TIKTOKEN_MAX_ENCODE_CHARS and MAX_NO_WHITESPACES_CHARS in tokenization_kimi.py.
    _MAX_ENCODE_CHARS = 400_000
    _MAX_RUN_CHARS = 25_000
    # Tiny segments spend their time in Python, so the pool would only add contention.
    _POOL_MIN_CHARS = 4_096
    _TASK_CHARS = 32_768
    _CUT = re.compile(r"\n(?=[!-~])")
    _SPACE = re.compile(r"\s")
    _NON_SPACE = re.compile(r"\S")

    @classmethod
    def applies_to(cls, tokenizer) -> bool:
        try:
            source = inspect.getsource(type(tokenizer))
        except (OSError, TypeError):
            return False
        source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        return source_sha256 in cls._SOURCE_SHA256 and not any(
            "\n" in token for token in tokenizer.special_tokens
        )

    @classmethod
    def _runs_may_exceed_limit(cls, text: str) -> bool:
        # A window holding both kinds of character stops a run, so with every full
        # window mixed a run is shorter than two windows and the splitter yields text whole.
        window = cls._MAX_RUN_CHARS // 2
        return any(
            not (
                cls._SPACE.search(text, start, start + window)
                and cls._NON_SPACE.search(text, start, start + window)
            )
            for start in range(0, len(text) - window + 1, window)
        )

    @classmethod
    def _pieces(cls, tokenizer, text: str):
        for begin in range(0, len(text), cls._MAX_ENCODE_CHARS):
            chunk = text[begin : begin + cls._MAX_ENCODE_CHARS]
            if cls._runs_may_exceed_limit(chunk):
                yield from tokenizer._split_whitespaces_or_nonwhitespaces(
                    chunk, cls._MAX_RUN_CHARS
                )
                continue
            start = 0
            while len(chunk) - start > cls._TASK_CHARS:
                cut = cls._CUT.search(chunk, start + cls._TASK_CHARS)
                if cut is None:
                    break
                yield chunk[start : cut.end()]
                start = cut.end()
            yield chunk[start:]

    @staticmethod
    def _encode_piece(tokenizer, text: str, allow_special: bool) -> List[int]:
        if allow_special:
            return tokenizer.model.encode(text, allowed_special="all")
        return tokenizer.model.encode(text, disallowed_special=())

    @classmethod
    def _encode(cls, tokenizer, encode_inline, segments, pool_min_chars) -> List[int]:
        pool = _encode_pool()
        parts, inline = [], []
        for segment in segments:
            text, allow_special = segment.text, segment.allow_special
            if len(text) < pool_min_chars:
                inline.extend(encode_inline(tokenizer, text, allow_special))
                continue
            parts.append(inline)
            inline = []
            parts.extend(
                pool.submit(cls._encode_piece, tokenizer, piece, allow_special)
                for piece in cls._pieces(tokenizer, text)
            )
        parts.append(inline)
        ids = []
        for part in parts:
            ids.extend(part if isinstance(part, list) else part.result())
        return ids

    @staticmethod
    def _worthwhile(chars: int) -> bool:
        return (
            envs.SGLANG_PARALLEL_PROMPT_ENCODE.get()
            and chars >= envs.SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS.get()
        )

    @classmethod
    def patch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        original_encode_text_piece = tokenizer_cls._encode_text_piece
        _encode_pool()

        def patched_encode_chat_segments(self, segments) -> List[int]:
            segments = list(segments)
            chars = sum(len(segment.text) for segment in segments)
            pool_min_chars = cls._POOL_MIN_CHARS if cls._worthwhile(chars) else inf
            return cls._encode(
                self, original_encode_text_piece, segments, pool_min_chars
            )

        def patched_encode_text_piece(
            self, text: str, allow_special_tokens: bool = True
        ) -> List[int]:
            if len(text) < cls._POOL_MIN_CHARS or not cls._worthwhile(len(text)):
                return original_encode_text_piece(self, text, allow_special_tokens)
            return cls._encode(
                self,
                original_encode_text_piece,
                [_TextPiece(text, allow_special_tokens)],
                cls._POOL_MIN_CHARS,
            )

        tokenizer_cls._parallel_original_methods = (
            tokenizer_cls._encode_chat_segments,
            original_encode_text_piece,
        )
        tokenizer_cls._encode_chat_segments = patched_encode_chat_segments
        tokenizer_cls._encode_text_piece = patched_encode_text_piece
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)
        return tokenizer

    @classmethod
    def unpatch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if not getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        (
            tokenizer_cls._encode_chat_segments,
            tokenizer_cls._encode_text_piece,
        ) = tokenizer_cls._parallel_original_methods
        del tokenizer_cls._parallel_original_methods
        delattr(tokenizer_cls, cls._PATCHED_FLAG)
        return tokenizer


_SPECIAL_LITERAL_REGEX_ATTR = "_sglang_special_literal_regex"


def _special_literal_regex(tokenizer):
    regex = getattr(tokenizer, _SPECIAL_LITERAL_REGEX_ATTR, None)
    if regex is None:
        regex = re.compile(
            "|".join(
                re.escape(token)
                for token in sorted(tokenizer.special_tokens, key=len, reverse=True)
            )
        )
        setattr(tokenizer, _SPECIAL_LITERAL_REGEX_ATTR, regex)
    return regex


def _make_cached_property(cache_attr, original_fn):
    @property
    def cached_prop(self):
        if getattr(self, cache_attr, None) is None:
            setattr(self, cache_attr, original_fn(self))
        return getattr(self, cache_attr)

    return cached_prop
