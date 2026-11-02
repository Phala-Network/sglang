"""Request-local adapter for K3's segmented checkpoint encoder.

Only marker-bearing historical strings change representation. The checkpoint
encoder, its module globals, and the caller's messages remain untouched.
"""

import json
from functools import lru_cache
from types import FunctionType

from sglang.srt.function_call.kimik3_format import ARGUMENT_CLOSE


def _with_globals(function, namespace):
    adapted = FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    adapted.__kwdefaults__ = function.__kwdefaults__
    return adapted


@lru_cache(maxsize=4)
def _history_template(template):
    builder = template.__globals__.get("build_chat_segments")
    if (
        template.__name__ != "apply_chat_template"
        or not isinstance(builder, FunctionType)
        or builder.__name__ != "build_chat_segments"
        or not builder.__module__.endswith(".encoding_k3")
    ):
        raise ValueError("K3 history encoding requires its segmented tokenizer")
    normalizer = builder.__globals__.get("normalize_conversation")
    if (
        not isinstance(normalizer, FunctionType)
        or normalizer.__globals__ is not builder.__globals__
        or normalizer.__name__ != "normalize_conversation"
    ):
        raise ValueError("K3 history encoding requires conversation normalization")
    image_placeholder = builder.__globals__.get("IMAGE_PLACEHOLDER", "")

    def normalize_history(messages):
        normalized = normalizer(messages)
        for message in normalized:
            if not isinstance(message, dict):
                continue
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function", call)
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments")
                if not isinstance(arguments, list):
                    continue
                rewritten = []
                for key, wire_type, value in arguments:
                    if wire_type == "string" and ARGUMENT_CLOSE in value:
                        value = json.dumps(value, ensure_ascii=False)
                        # Keep literal image text out of the media placeholder
                        # expander, using an ordinary reversible JSON escape.
                        if image_placeholder:
                            value = value.replace(
                                image_placeholder, "\\u003c" + image_placeholder[1:]
                            )
                        wire_type = "json-string"
                    rewritten.append((key, wire_type, value))
                function["arguments"] = rewritten
        return normalized

    builder_globals = dict(builder.__globals__)
    builder_globals["normalize_conversation"] = normalize_history
    template_globals = dict(template.__globals__)
    template_globals["build_chat_segments"] = _with_globals(builder, builder_globals)
    return _with_globals(template, template_globals), normalizer


def _history_arguments(messages):
    for message in messages:
        if isinstance(message, list):
            yield from _history_arguments(message)
        elif isinstance(message, dict):
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict):
                    function = call.get("function", call)
                    if isinstance(function, dict):
                        yield function.get("arguments")


def apply_chat_template(tokenizer, conversation, **kwargs):
    template = tokenizer.apply_chat_template
    # Ordinary histories take the original path, including exact safe raw and
    # non-string JSON literal rendering. Unicode escapes need normalization
    # before deciding whether they spell a real delimiter or literal text.
    possible = any(
        any(
            isinstance(value, str) and ARGUMENT_CLOSE in value
            for value in arguments.values()
        )
        if isinstance(arguments, dict)
        else isinstance(arguments, str)
        and (ARGUMENT_CLOSE in arguments or "\\u" in arguments)
        for arguments in _history_arguments(conversation)
    )
    if not possible:
        return template(conversation, **kwargs)
    function = getattr(template, "__func__", None)
    if not isinstance(function, FunctionType):
        raise ValueError("K3 marker history requires its segmented tokenizer")
    adapted, normalizer = _history_template(function)
    normalized = normalizer(conversation)
    if not any(
        wire_type == "string" and ARGUMENT_CLOSE in value
        for arguments in _history_arguments(normalized)
        if isinstance(arguments, list)
        for _key, wire_type, value in arguments
    ):
        return template(conversation, **kwargs)
    return adapted(tokenizer, conversation, **kwargs)
