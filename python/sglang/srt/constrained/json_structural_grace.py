"""Default-off, structural-only JSON completion after the original token budget.

Admission reserves B + 32. Only a whole suffix proven on the original matcher
may cross B; a failed proof retains the native length finish at B. The suffix
contains only a value's closing quote, container closers and the real EOS.
"""

import json
from collections import OrderedDict

_VOCABS = OrderedDict()

def _vocab(ctx):
    key = id(ctx)
    if key not in _VOCABS:
        info = ctx.tokenizer_info
        _VOCABS[key] = (
            ctx,
            (
                tuple(info.decoded_vocab),
                frozenset(info.special_token_ids) | frozenset(info.stop_token_ids),
                tuple(info.stop_token_ids),
            ),
        )
        if len(_VOCABS) > 4:
            _VOCABS.popitem(last=False)
    _VOCABS.move_to_end(key)
    return _VOCABS[key][1]

def _accept(matcher, token, vocab_size):
    from xgrammar import allocate_token_bitmask

    if not 0 <= token < vocab_size or matcher.is_terminated():
        return False
    mask = allocate_token_bitmask(1, vocab_size)
    apply = matcher.fill_next_token_bitmask(mask)
    if apply and not ((int(mask[0, token // 32]) >> (token % 32)) & 1):
        return False
    return matcher.accept_token(token)

def _legal_plan(matcher, ids, vocab_size):
    trial = matcher.fork()
    return (
        all(_accept(trial, t, vocab_size) for t in ids)
        and trial.is_completed()
        and trial.is_terminated()
    )

GRACE_TOKENS = 32
_BASE_FIELD = "json_structural_grace_base_max_new_tokens"


def reserve_json_structural_grace(obj, input_len, context_len, server_args, model_config):
    """Reserve once before token validation; unsupported requests keep their cap."""
    # Batch items may share the caller's dictionary; each reservation is local.
    p = obj.sampling_params = dict(obj.sampling_params)
    previous = getattr(obj, "_json_structural_grace_reserved_budget", None)
    p.pop(_BASE_FIELD, None)
    if previous is not None and p.get("max_new_tokens") == previous + GRACE_TOKENS:
        p["max_new_tokens"] = previous
    obj._json_structural_grace_reserved_budget = None
    if (
        getattr(server_args, "grammar_backend", None) != "xgrammar"
        or getattr(server_args, "disaggregation_mode", "null") != "null"
        or getattr(model_config, "request_static_yarn_config", None) is not None
        or obj.require_reasoning
        or obj.session_id is not None
        or obj.session_params
        or any(getattr(obj, k, None) for k in (
            "image_data", "video_data", "audio_data", "input_embeds",
            "positional_embed_overrides", "custom_logit_processor",
            "return_logprob", "return_sampling_mask", "return_hidden_states",
            "return_routed_experts", "return_indexer_topk", "token_ids_logprob",
        ))
        or not isinstance(p.get("json_schema"), str)
        or p["json_schema"] == "$$ANY$$"
        or p.get("n", 1) != 1
        or p.get("beam_width") not in (None, 1)
        or any(p.get(k) for k in (
            "regex", "ebnf", "structural_tag", "ignore_eos", "min_new_tokens",
            "stop", "stop_strs", "stop_regex", "stop_regex_strs", "stop_token_ids",
            "logit_bias", "frequency_penalty", "presence_penalty",
        ))
    ):
        return
    budget = p.get("max_new_tokens")
    if type(budget) is int and budget > 0 and input_len + budget + GRACE_TOKENS <= context_len:
        obj._json_structural_grace_reserved_budget = budget
        p[_BASE_FIELD] = budget
        p["max_new_tokens"] = budget + GRACE_TOKENS


def _structural_suffix(data):
    """Never supply content, an escape, a key, a colon or a missing array item."""
    text = data.decode("utf-8")
    stack = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return None
    if escaped or len(stack) > 64:
        return None
    suffix = ('"' if in_string else "") + "".join(reversed(stack))
    # This also rejects partial escapes/Unicode, unfinished keys, trailing
    # commas, missing values and incomplete numbers. Schema checks follow.
    json.loads(text + suffix)
    return suffix.encode()


def _suffix_paths(ctx, data):
    vocab, excluded, _ = _vocab(ctx)
    options = [(t, raw) for t, raw in enumerate(vocab)
               if t not in excluded and raw and raw in data]
    paths = []

    def visit(offset, path):
        if len(paths) >= 128:
            raise ValueError("structural tokenization bound")
        if offset == len(data):
            paths.append(path)
        elif len(path) < GRACE_TOKENS - 1:
            for token, raw in options:
                if data.startswith(raw, offset):
                    visit(offset + len(raw), (*path, token))

    visit(0, ())
    return sorted(paths, key=lambda p: (len(p), p))


class JsonStructuralGrace:
    """Request-local proof; the anchor includes token identity across DFS rollback."""

    def __init__(self, budget):
        self.budget = budget
        self.anchor = None
        self.plan = None

    def plan_for(self, grammar, history=None):
        history = grammar.accepted_tokens if history is None else history
        if len(history) < self.budget:
            return None
        anchor = tuple(history[:self.budget])
        if self.anchor != anchor:
            self.anchor, self.plan = anchor, None
            try:
                vocab, excluded, stops = _vocab(grammar.ctx)
                if len(stops) != 1 or any(t in excluded for t in anchor):
                    return None
                current = grammar.accepted_tokens
                if tuple(current[:min(len(current), self.budget)]) != anchor[:len(current)]:
                    return None
                trial = grammar.matcher.fork()
                if len(current) > self.budget:
                    trial.rollback(len(current) - self.budget)
                elif not all(_accept(trial, t, grammar.vocab_size)
                             for t in anchor[len(current):]):
                    return None
                data = b"".join(vocab[t] for t in anchor)
                if len(data) > 1024 * 1024:
                    return None
                suffix = _structural_suffix(data)
                if suffix is None:
                    return None
                for path in _suffix_paths(grammar.ctx, suffix):
                    plan = (*path, stops[0])
                    if len(plan) <= GRACE_TOKENS and _legal_plan(trial, plan, grammar.vocab_size):
                        self.plan = plan
                        break
            except (IndexError, TypeError, ValueError, RuntimeError, UnicodeError):
                # A conservative failed proof preserves the original length cap.
                return None
        tail = tuple(history[self.budget:])
        if self.plan is None or tail != self.plan[:len(tail)]:
            return None
        return self.plan

    def fill_vocab_mask(self, grammar, mask, idx):
        if len(grammar.accepted_tokens) < self.budget:
            return
        plan = self.plan_for(grammar)
        offset = len(grammar.accepted_tokens) - self.budget
        if plan is None or offset >= len(plan):
            # Never create an empty row: the original B finish discards scratch.
            return
        token = plan[offset]
        word, bit = token // 32, token % 32
        allowed = int(mask[idx, word]) & (1 << bit)
        if allowed:
            mask[idx].zero_()
            mask[idx, word] = allowed if allowed < 1 << 31 else allowed - (1 << 32)

    def at_limit(self, grammar):
        count = len(grammar.accepted_tokens)
        return count >= self.budget and (
            self.plan_for(grammar) is None or count >= self.budget + GRACE_TOKENS
        )


def _inner(grammar):
    from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
    from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar

    if type(grammar) is ReasonerGrammarObject:
        if grammar.tokens_in_think != -1 or grammar._thinking_match_history:
            return None
        inner = grammar.grammar
        if type(inner) is not XGrammarGrammar or grammar.tokens_after_end != len(inner.accepted_tokens):
            return None
        grammar = inner
    return grammar if type(grammar) is XGrammarGrammar else None


def bind_json_structural_grace(req):
    budget = req.sampling_params.json_structural_grace_base_max_new_tokens
    if budget is None:
        return
    grammar = _inner(req.grammar)
    if (grammar is not None and not req.require_reasoning
        and grammar.key_string == req.sampling_params.json_schema
        and grammar.grammar_stats is not None
        and grammar.grammar_stats.dispatch_type == "json"):
        grammar.json_structural_grace = JsonStructuralGrace(budget)


def request_grace_limit(req):
    p = req.sampling_params
    budget = p.json_structural_grace_base_max_new_tokens
    if budget is None:
        return p.max_new_tokens
    if len(req.output_ids) >= budget:
        grammar = _inner(req.grammar)
        grace = getattr(grammar, "json_structural_grace", None)
        if grace is not None and grace.plan_for(grammar, req.output_ids) is not None:
            return min(p.max_new_tokens, budget + GRACE_TOKENS)
    return min(p.max_new_tokens, budget)


def grammar_grace_at_limit(req):
    if req.sampling_params.json_structural_grace_base_max_new_tokens is None:
        return False
    grammar = _inner(req.grammar)
    grace = getattr(grammar, "json_structural_grace", None)
    return grace is not None and grace.at_limit(grammar)
