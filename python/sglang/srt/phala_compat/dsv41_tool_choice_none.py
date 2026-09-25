"""Source-integrated adapter, converted from dsv41_tool_choice_none.py.
No import hooks or external runtime code paths.
"""

import functools

_TARGET = "sglang.srt.entrypoints.openai.serving_chat"


_MARK = "_dsv41_tool_choice_none"


def drop_tool_definitions(request) -> bool:
    """Remove tool *definitions* from ``request`` when tool_choice is "none".

    Covers both carriers SGLang's encoder reads: the top-level ``tools`` array
    and the per-message ``tools`` field of a system/developer message.
    """
    if getattr(request, "tool_choice", None) != "none":
        return False
    dropped = bool(getattr(request, "tools", None))
    request.tools = None
    for message in getattr(request, "messages", None) or []:
        if getattr(message, "tools", None):
            message.tools = None
            dropped = True
    return dropped


def apply(module) -> None:
    cls = module.OpenAIServingChat
    original = cls._process_messages
    if getattr(original, _MARK, False):
        return

    @functools.wraps(original)
    def _process_messages(self, request, *args, **kwargs):
        drop_tool_definitions(request)
        return original(self, request, *args, **kwargs)

    setattr(_process_messages, _MARK, True)
    cls._process_messages = _process_messages
