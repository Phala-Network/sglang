"""Source-integrated adapter, converted from dsv41_protocol_compat.py.
No import hooks or external runtime code paths.
"""

import functools
import sys
import traceback
from typing import Dict, Optional

_PROTOCOL = "sglang.srt.entrypoints.openai.protocol"


_USAGE_PROCESSOR = "sglang.srt.entrypoints.openai.usage_processor"


_SERVING_CHAT = "sglang.srt.entrypoints.openai.serving_chat"


_MARK = "_dsv41_protocol_compat"


_warned = False


_DEFAULT_ALIASES = {"default"}


_TRUE_STRINGS = {"1", "true", "yes", "y", "on"}


def _int_budget(value):
    """Return ``value`` as a 1-100 reasoning budget, or None if it is not one.

    Only a JSON integer is read as a budget: a float is sglang's documented
    fine-grained effort in [0.0, 0.99] and must keep that meaning, and ``0``
    stays a float effort too (``0.0`` already validates today).  An integer
    outside the encoder's range is a 400 with a message that says so, instead of
    upstream's "input should be less than or equal to 0.99".
    """
    if isinstance(value, bool) or not isinstance(value, int) or value == 0:
        return None
    if not 1 <= value <= 100:
        raise ValueError(
            f"invalid reasoning_effort {value}: an integer budget must be in [1, 100]"
        )
    return value


def _is_false(value):
    """True only when ``value`` is an explicit false, not when it is absent."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in _TRUE_STRINGS
    return not value


def _force_continuous_usage(out):
    """(6) Force per-chunk cumulative usage onto every streaming request.

    ``out`` is already a shallow copy of the request dict (the caller's
    ``dict(values)``); this mutates and returns it plus whether it changed
    anything, so the caller's existing ``changed`` bookkeeping stays correct
    and a client that changes nothing about usage gets the exact original
    dict back (no spurious copies).

    Only acts when ``stream`` is truthy -- a non-streaming request is
    untouched. ``include_usage``/``continuous_usage_stats`` are set ``True``
    whenever either is missing or falsy; an already-``True`` value is left
    alone (nothing to downgrade -- the only value this ever writes is
    ``True``). ``stream_options`` is created from scratch if absent, or if it
    was sent as something other than a JSON object (pydantic would reject
    that anyway; replacing it here just means the 400 for a malformed
    ``stream_options`` looks like the 400 for a missing one).
    """
    if not out.get("stream"):
        return out, False
    stream_options = out.get("stream_options")
    stream_options = dict(stream_options) if isinstance(stream_options, dict) else {}
    if (
        stream_options.get("include_usage") is True
        and stream_options.get("continuous_usage_stats") is True
    ):
        return out, False
    stream_options["include_usage"] = True
    stream_options["continuous_usage_stats"] = True
    out["stream_options"] = stream_options
    return out, True


def _normalize_reasoning_inputs(values):
    """Fail-open entry point: our own ValueErrors are 400s, anything else is a bug
    and must leave the request exactly as upstream would have seen it."""
    try:
        return _normalize(values)
    except ValueError:
        raise
    except Exception:  # noqa: BLE001
        global _warned
        if not _warned:
            _warned = True
            print(
                "[dsv41-patches] reasoning-input normalization raised; "
                "passing the request through unchanged",
                file=sys.stderr,
            )
            traceback.print_exc()
        return values


def _normalize(values):
    if not isinstance(values, dict):
        return values

    reasoning = values.get("reasoning")
    reasoning = dict(reasoning) if isinstance(reasoning, dict) else None
    out = dict(values)
    changed = "dsv41_reasoning_off_requested" in out
    out.pop("dsv41_reasoning_off_requested", None)

    # (6) Every streaming request gets both usage flags forced on, regardless
    # of what the client sent -- see the module docstring, point 6.
    out, usage_changed = _force_continuous_usage(out)
    changed = changed or usage_changed

    # (2) "default" means "whatever the server defaults to" -> drop the field.
    for holder, keys in (
        (out, ("reasoning_effort",)),
        (reasoning, ("effort", "reasoning_effort")),
    ):
        if holder is None:
            continue
        for key in keys:
            value = holder.get(key)
            if isinstance(value, str) and value.strip().lower() in _DEFAULT_ALIASES:
                holder.pop(key)
                changed = True

    # Preserve upstream's nested effort precedence during request validation.
    # The serving layer applies this off-switch only for the dsv41 encoder.
    if reasoning is not None and ("enabled" in reasoning or "enable" in reasoning):
        enabled = reasoning.get("enabled")
        if enabled is None:
            enabled = reasoning.get("enable")
        if _is_false(enabled) and out.get("reasoning_effort") is None:
            out["dsv41_reasoning_off_requested"] = True
            changed = True

    # The typed request field carries integer budgets to the custom encoder and
    # lets serving reject them for other models. Preserve nested precedence.
    _int_budget(out.get("reasoning_effort"))
    if reasoning is not None:
        nested = reasoning.get("effort")
        if nested is None:
            nested = reasoning.get("reasoning_effort")
        nested_budget = _int_budget(nested)
        if nested_budget is not None:
            out["reasoning_effort"] = nested_budget
            reasoning.pop("effort", None)
            reasoning.pop("reasoning_effort", None)
            changed = True

    # (3) Record reasoning.exclude for the response side.
    if (
        reasoning is not None
        and "exclude" in reasoning
        and "reasoning_exclude" not in out
    ):
        exclude = reasoning.get("exclude")
        if isinstance(exclude, str):
            exclude = exclude.strip().lower() in _TRUE_STRINGS
        out["reasoning_exclude"] = bool(exclude)
        changed = True

    if not changed:
        return values
    if reasoning is not None:
        out["reasoning"] = reasoning
    return out


def _add_field(model_class, name, annotation, default) -> None:
    """Add one field to an already-built pydantic model. Caller rebuilds."""
    from pydantic.fields import FieldInfo

    if name in model_class.__pydantic_fields__:
        return
    model_class.__annotations__[name] = annotation
    model_class.__pydantic_fields__[name] = FieldInfo(
        annotation=annotation, default=default
    )


def _patch_reasoning_inputs(module) -> None:
    request_class = module.ChatCompletionRequest
    decorator = request_class.__pydantic_decorators__.model_validators[
        "normalize_reasoning_inputs"
    ]
    original = decorator.func
    if getattr(original, _MARK, False):
        return

    if getattr(original, "__self__", None) is not None:
        # pydantic 2.13 stores the classmethod already bound to the class.
        @functools.wraps(original)
        def normalize_reasoning_inputs(values, *args, **kwargs):
            return original(_normalize_reasoning_inputs(values), *args, **kwargs)

    else:

        @functools.wraps(original)
        def normalize_reasoning_inputs(cls, values, *args, **kwargs):
            return original(cls, _normalize_reasoning_inputs(values), *args, **kwargs)

    setattr(normalize_reasoning_inputs, _MARK, True)
    decorator.func = normalize_reasoning_inputs
    _add_field(request_class, "reasoning_exclude", bool, False)
    request_class.model_rebuild(force=True)


def _patch_usage_details(module) -> None:
    _add_field(
        module.UsageInfo,
        "completion_tokens_details",
        Optional[Dict[str, int]],
        None,
    )
    module.UsageInfo.model_rebuild(force=True)
    # Parents inlined UsageInfo's old schema when they were built; only the chat
    # surfaces are rebuilt, so no other endpoint's usage shape moves.
    for name in ("ChatCompletionResponse", "ChatCompletionStreamResponse"):
        getattr(module, name).model_rebuild(force=True)


def _patch_usage_processor(module) -> None:
    usage_class = module.UsageProcessor
    original = usage_class.calculate_token_usage
    if getattr(original, _MARK, False):
        return

    @functools.wraps(original)
    def calculate_token_usage(*args, **kwargs):
        usage = original(*args, **kwargs)
        reasoning_tokens = getattr(usage, "reasoning_tokens", None)
        if (
            reasoning_tokens is not None
            and "completion_tokens_details" in type(usage).model_fields
        ):
            usage.completion_tokens_details = module.CompletionTokensDetails(
                reasoning_tokens=int(reasoning_tokens)
            )
        return usage

    # The single funnel for chat (streaming and not) and /v1/completions usage;
    # both callers reach it by class attribute, so the wrapper is picked up.
    # Embeddings/score build UsageInfo directly and stay on the old shape.
    setattr(calculate_token_usage, _MARK, True)
    usage_class.calculate_token_usage = staticmethod(calculate_token_usage)


def _patch_reasoning_exclude(module) -> None:
    serving_class = module.OpenAIServingChat
    build_original = serving_class._build_chat_response
    stream_original = serving_class._process_reasoning_stream
    if getattr(build_original, _MARK, False):
        return

    @functools.wraps(build_original)
    def _build_chat_response(self, request, *args, **kwargs):
        response = build_original(self, request, *args, **kwargs)
        if getattr(request, "reasoning_exclude", False):
            for choice in getattr(response, "choices", None) or []:
                message = getattr(choice, "message", None)
                if getattr(message, "reasoning_content", None):
                    message.reasoning_content = None
        return response

    @functools.wraps(stream_original)
    def _process_reasoning_stream(
        self, index, delta, reasoning_parser_dict, content, request, *args, **kwargs
    ):
        reasoning_text, normal_text = stream_original(
            self, index, delta, reasoning_parser_dict, content, request, *args, **kwargs
        )
        if getattr(request, "reasoning_exclude", False):
            return None, normal_text
        return reasoning_text, normal_text

    setattr(_build_chat_response, _MARK, True)
    setattr(_process_reasoning_stream, _MARK, True)
    serving_class._build_chat_response = _build_chat_response
    serving_class._process_reasoning_stream = _process_reasoning_stream
