"""Source-integrated adapter, converted from dsv41_message_roles.py.
No import hooks or external runtime code paths.
"""

import functools


import os


import sys


import traceback


_ENCODING = "sglang.srt.entrypoints.openai.encoding_dsv41"


_MARK = "_dsv41_message_roles"


_warned = False


def _system_only_user_turn_enabled():
    # Read per call: the value is fixed for the life of the process (env changes
    # need a container recreate anyway), and this keeps the flag testable.
    return os.environ.get("DSV41_SYSTEM_ONLY_USER_TURN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _needs_user_turn(messages):
    """True for the one shape the format has no generation point for.

    `render_message` emits the assistant header for a user/developer turn, for a
    system message at index > 0 (the documented mid-conversation system message),
    and for assistant continuations. A lone system message at index 0 matches
    none of them; every other trailing role does (a `tool` message is merged into
    the preceding user turn by `merge_tool_messages` before rendering).
    """
    return (
        len(messages) == 1
        and isinstance(messages[0], dict)
        and messages[0].get("role") == "system"
    )


def _prepare_messages(messages):
    if not isinstance(messages, list) or not messages:
        return messages

    out = messages
    last = len(out) - 1
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "developer":
            continue
        # A developer message that is the whole conversation is the message that
        # carries the assistant generation header. Mapping it to a system message
        # at index 0 would take that header away, so it is left alone and keeps
        # rendering exactly as it does today. Every other position maps cleanly.
        if index == 0 and index == last:
            continue
        if out is messages:
            out = list(messages)
        out[index] = dict(message, role="system")

    if _system_only_user_turn_enabled() and _needs_user_turn(out):
        out = list(out) + [{"role": "user", "content": ""}]
    return out


def _patch_encoding(module) -> None:
    original = module.encode_messages
    if getattr(original, _MARK, False):
        return

    @functools.wraps(original)
    def encode_messages(*args, **kwargs):
        try:
            if args:
                args = (_prepare_messages(args[0]),) + args[1:]
            elif "messages" in kwargs:
                kwargs["messages"] = _prepare_messages(kwargs["messages"])
        except Exception:  # noqa: BLE001 -- fail open: encode what upstream got
            global _warned
            if not _warned:
                _warned = True
                print(
                    "[dsv41-patches] message-role preparation raised; encoding the "
                    "messages unchanged",
                    file=sys.stderr,
                )
                traceback.print_exc()
        return original(*args, **kwargs)

    setattr(encode_messages, _MARK, True)
    module.encode_messages = encode_messages
