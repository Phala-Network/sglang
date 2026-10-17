"""Opt-in structural JSON close inside the existing request hard cap.

Only the greedy EAGLE front chain is supported. Decisions use the original
matcher and its accepted history, including the overlap grammar barrier's
advance. No closure mode or output history is stored by this policy.
"""

import json
import logging
from collections import OrderedDict

from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang.srt.environ import envs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

_RESERVE = 16
_MAX_DEPTH = 8
_MAX_PATHS = 128
_MAX_PREFIX_BYTES = 1024 * 1024


def _state(data):
    """Return (kind, closing punctuation), never certify a partial primitive."""
    text = data.decode("utf-8")
    decoder = json.JSONDecoder()
    stack, root, last = [], "value", None
    i, n = 0, len(text)

    def completed(kind):
        nonlocal root, last
        if stack:
            stack[-1][1] = "comma_or_end"
        else:
            root = "done"
        last = kind

    while True:
        while i < n and text[i] in " \t\r\n":
            i += 1
        state = stack[-1][1] if stack else root
        if i == n:
            if last in ("string", "object", "array") and state in (
                "comma_or_end",
                "done",
            ):
                return (
                    last,
                    "".join(
                        "}" if f[0] == "object" else "]" for f in reversed(stack)
                    ).encode(),
                )
            return None, None
        ch = text[i]
        if state == "done":
            return None, None
        if state in ("key_or_end", "key_required"):
            if state == "key_or_end" and ch == "}":
                stack.pop()
                i += 1
                completed("object")
                continue
            try:
                key, end = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                return None, None
            if not isinstance(key, str):
                return None, None
            i, stack[-1][1], last = end, "colon", None
            continue
        if state == "colon":
            if ch != ":":
                return None, None
            i, stack[-1][1], last = i + 1, "value", None
            continue
        if state == "comma_or_end":
            closer = "}" if stack[-1][0] == "object" else "]"
            if ch == closer:
                kind = stack.pop()[0]
                i += 1
                completed(kind)
                continue
            if ch != ",":
                return None, None
            stack[-1][1] = (
                "key_required" if stack[-1][0] == "object" else "value_required"
            )
            i, last = i + 1, None
            continue
        if state == "value_or_end" and ch == "]":
            stack.pop()
            i += 1
            completed("array")
            continue
        if ch in "{[":
            if len(stack) >= _MAX_DEPTH:
                return None, None
            stack.append(
                ["object", "key_or_end"] if ch == "{" else ["array", "value_or_end"]
            )
            i, last = i + 1, None
            continue
        try:
            value, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            return ("incomplete_string", None) if ch == '"' else (None, None)
        if not isinstance(value, str) and (end == n or text[end] not in " \t\r\n,]}"):
            return None, None
        i = end
        completed("string" if isinstance(value, str) else "primitive")


# CompiledGrammar is unhashable in xgrammar0.2.7. Retain the context in
# each bounded identity-cache entry so an id cannot be reused while cached.
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


def _paths(ctx, data):
    """Bounded exact-byte punctuation paths; overflow is unsupported."""
    if len(data) > _MAX_DEPTH or any(c not in b"]}" for c in data):
        return ()
    if not data:
        return ((),)
    vocab, excluded, _ = _vocab(ctx)
    options = [
        (token, raw)
        for token, raw in enumerate(vocab)
        if token not in excluded and raw and raw in data
    ]
    paths = []

    def visit(offset, path):
        if len(paths) >= _MAX_PATHS:
            raise ValueError("structural tokenization bound")
        if offset == len(data):
            paths.append(path)
            return
        for token, raw in options:
            if data.startswith(raw, offset):
                visit(offset + len(raw), (*path, token))

    try:
        visit(0, ())
    except ValueError:
        return ()
    return tuple(sorted(paths, key=lambda p: (len(p), p)))


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


def _select(grammar, matcher, prefix, count, hard_cap, proposal):
    if (
        count < hard_cap - _RESERVE
        or count >= hard_cap
        or len(prefix) > _MAX_PREFIX_BYTES
    ):
        return proposal
    vocab, excluded, stops = _vocab(grammar.ctx)
    if len(stops) != 1 or not _accept(matcher.fork(), proposal, grammar.vocab_size):
        return proposal
    stop = stops[0]
    kind, suffix = _state(prefix)
    plans = []
    if suffix is not None:
        plans = [(*path, stop) for path in _paths(grammar.ctx, suffix)]
    elif kind == "incomplete_string":
        raw = vocab[proposal]
        # Support only a natural value-ending quote followed by one comma.
        # Any business tail must fit, with its quote, in the first replacement
        # token. Never commit a partial tail and remember a future close plan.
        if not raw.endswith(b'",') or len(raw) > 64:
            return proposal
        piece = raw[:-1]
        end_kind, suffix = _state(prefix + piece)
        if end_kind != "string" or suffix is None:
            return proposal
        data = piece + suffix
        choices = [
            (t, b)
            for t, b in enumerate(vocab)
            if t not in excluded and len(b) >= len(piece) and b and data.startswith(b)
        ]
        if len(choices) > _MAX_PATHS:
            return proposal
        for token, first in choices:
            if _state(prefix + first)[1] is None:
                continue
            for path in _paths(grammar.ctx, data[len(first) :]):
                plans.append((token, *path, stop))
                if len(plans) > _MAX_PATHS:
                    return proposal
    for plan in sorted(plans, key=lambda p: (len(p), p)):
        # Keep one token of margin; do not alter native finish precedence.
        if count + len(plan) < hard_cap and _legal_plan(
            matcher, plan, grammar.vocab_size
        ):
            return plan[0]
    return proposal


def _request_grammar(req):
    p = req.sampling_params
    if (
        req.require_reasoning
        or req.is_retracted
        or req.finished()
        or req.to_finish is not None
        or req.return_logprob
        or req.return_sampling_mask
        or req.return_hidden_states
        or req.return_routed_experts
        or req.return_indexer_topk
        or req.custom_logit_processor
        or p.json_schema is None
        or p.json_schema == "$$ANY$$"
        or p.regex
        or p.ebnf
        or p.structural_tag
        or p.top_k != 1
        or p.n != 1
        or p.beam_width not in (None, 1)
        or p.ignore_eos
        or p.min_new_tokens
        or p.stop_strs
        or p.stop_regex_strs
        or p.stop_token_ids
        or p.frequency_penalty
        or p.presence_penalty
        or p.logit_bias
        or type(p.max_new_tokens) is not int
        or p.max_new_tokens <= _RESERVE
    ):
        return None
    grammar = req.grammar
    if type(grammar) is ReasonerGrammarObject:
        # Require reasoning disabled from the start, not merely ended now.
        if grammar.tokens_in_think != -1 or grammar._thinking_match_history:
            return None
        inner = grammar.grammar
        if type(inner) is not XGrammarGrammar or grammar.tokens_after_end != len(
            inner.accepted_tokens
        ):
            return None
        grammar = inner
    if (
        type(grammar) is not XGrammarGrammar
        or grammar.finished
        or grammar.is_terminated()
        or grammar.key_string != p.json_schema
        or grammar.grammar_stats is None
        or grammar.grammar_stats.dispatch_type != "json"
    ):
        return None
    return grammar


def maybe_apply_json_budget_close(
    verify_input, batch, target_predict, *, tp_size, pp_size
):
    """Replace only supported target proposals before the existing greedy verify."""
    if (
        not envs.SGLANG_ENABLE_JSON_BUDGET_CLOSE.get()
        or tp_size != 1
        or pp_size != 1
        or batch.spec_algorithm
        not in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.EAGLE3)
        or not batch.has_grammar
        or not batch.sampling_info.is_all_greedy
        or batch.sampling_info.has_custom_logit_processor
        or batch.return_logprob
        or batch.return_hidden_states
        or verify_input.tree_topk != 1
        or not 1 <= verify_input.draft_token_num <= 16
        or verify_input.max_tree_depth != verify_input.draft_token_num
    ):
        return target_predict
    width = verify_input.draft_token_num
    changes = []
    for row, req in enumerate(batch.reqs):
        row_changes = []
        try:
            grammar = _request_grammar(req)
            if grammar is None:
                continue
            # output_ids may lag the overlap barrier. Only FSM history counts.
            accepted = grammar.accepted_tokens
            hard_cap = req.sampling_params.max_new_tokens
            if not accepted or len(accepted) + width - 1 < hard_cap - _RESERVE:
                continue
            history = list(accepted)
            index = verify_input.retrieve_index[row].tolist()
            child = verify_input.retrieve_next_token[row].tolist()
            sibling = verify_input.retrieve_next_sibling[row].tolist()
            if (
                index != list(range(row * width, (row + 1) * width))
                or child != [*range(1, width), -1]
                or sibling != [-1] * width
            ):
                continue
            candidates = verify_input.draft_token.reshape(-1, width)[row].tolist()
            if candidates[0] != history[-1]:
                continue
            predictions = target_predict[row].tolist()
            vocab = _vocab(grammar.ctx)[0]
            prefix = b"".join(vocab[t] for t in history)
            matcher = grammar.matcher.fork()
            count = len(history)
            for node, proposal in enumerate(predictions):
                selected = _select(grammar, matcher, prefix, count, hard_cap, proposal)
                if selected != proposal:
                    row_changes.append((row, node, selected, count, hard_cap))
                if node + 1 == width or selected != candidates[node + 1]:
                    break  # existing kernel returns selected as the bonus
                if not _accept(matcher, selected, grammar.vocab_size):
                    break
                prefix += vocab[selected]
                count += 1
                if matcher.is_terminated():
                    break
            changes.extend(row_changes)
        except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
            RuntimeError,
            UnicodeError,
        ):
            # Unsupported states preserve this request's original predictions.
            continue
    if not changes:
        return target_predict
    result = target_predict.clone()
    for row, node, token, count, hard_cap in changes:
        result[row, node] = token
        logger.info(
            "JSON budget close preverify: events=1 row=%d node=%d prefix_tokens=%d H=%d repetition_penalty=%g",
            row,
            node,
            count,
            hard_cap,
            batch.reqs[row].sampling_params.repetition_penalty,
        )
    return result
