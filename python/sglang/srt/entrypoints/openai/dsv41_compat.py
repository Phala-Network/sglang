"""Direct request compatibility helpers; no import hooks or runtime patches."""
import os

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

_ALLOWED_URL_SCHEMES = ("data:", "http://", "https://")

_MEDIA_PART_MODALITY = {
    "image_url": "image",
    "video_url": "video",
    "audio_url": "audio",
    "input_audio": "audio",
}

def _part_type(part):
    if isinstance(part, dict):
        return part.get("type")
    return getattr(part, "type", None)

def _part_url(part, part_type):
    holder = (
        part.get(part_type) if isinstance(part, dict) else getattr(part, part_type, None)
    )
    if isinstance(holder, str):
        return holder
    if isinstance(holder, dict):
        return holder.get("url")
    return getattr(holder, "url", None)

def _iter_media_parts(request):
    for message in getattr(request, "messages", None) or []:
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if not isinstance(content, list):
            continue
        for part in content:
            part_type = _part_type(part)
            modality = _MEDIA_PART_MODALITY.get(part_type)
            if modality is not None:
                yield part, part_type, modality

def _data_uri_size(url):
    """Decoded payload size of a ``data:`` URI, without decoding it."""
    head, separator, payload = url.partition(",")
    if not separator:
        return None  # malformed data URI; the loader reports it as a 400 now
    if ";base64" not in head.lower():
        return len(payload)
    return (len(payload) * 3) // 4 - payload[-2:].count("=")

def _supported_modalities(serving):
    tokenizer_manager = getattr(serving, "tokenizer_manager", None)
    model_config = getattr(tokenizer_manager, "model_config", None)
    mm_tokens = getattr(
        getattr(tokenizer_manager, "mm_processor", None), "mm_tokens", None
    )
    video = True
    if mm_tokens is not None:
        video = bool(
            getattr(mm_tokens, "video_token", None)
            or getattr(mm_tokens, "video_token_id", None)
        )
    return {
        "image": bool(getattr(model_config, "is_image_understandable_model", True)),
        "audio": bool(getattr(model_config, "is_audio_understandable_model", True)),
        "video": video,
    }

def _validate_media_parts(serving, request):
    supported = None
    max_bytes = 0
    for part, part_type, modality in _iter_media_parts(request):
        if supported is None:  # resolved once, and only for requests with media
            supported = _supported_modalities(serving)
            server_args = getattr(
                getattr(serving, "tokenizer_manager", None), "server_args", None
            )
            limit_mb = getattr(server_args, "media_url_max_file_size_mb", 0) or 0
            max_bytes = int(limit_mb) * 1024 * 1024

        if not supported.get(modality, True):
            return f"{modality} input is not supported by this model."

        url = _part_url(part, part_type)
        if not isinstance(url, str):
            continue
        if not url.startswith(_ALLOWED_URL_SCHEMES):
            return (
                f"Invalid {part_type}.url: expected a data: URI or an "
                "http(s):// URL. Bare base64 payloads and local paths are not "
                "accepted."
            )
        if max_bytes and url.startswith("data:"):
            size = _data_uri_size(url)
            if size is not None and size > max_bytes:
                return (
                    f"{part_type} data URI decodes to {size} bytes, over the "
                    f"{max_bytes} byte media size limit "
                    "(--media-url-max-file-size-mb)."
                )
    return None

def normalize_request(values):
    """Normalize aliases before upstream validates its documented float effort."""
    out = dict(values)
    r = dict(out.get("reasoning") or {}) if isinstance(out.get("reasoning"), dict) else {}
    for holder, keys in ((out, ("reasoning_effort",)), (r, ("effort", "reasoning_effort"))):
        for key in keys:
            value = holder.get(key)
            if isinstance(value, str) and value.strip().lower() == "default":
                holder.pop(key)
    truthy = {"1", "true", "yes", "y", "on"}
    enabled = r.get("enabled") if r.get("enabled") is not None else r.get("enable")
    if enabled is not None:
        enabled = enabled.strip().lower() in truthy if isinstance(enabled, str) else bool(enabled)
        if not enabled:
            r.pop("effort", None)
            r.pop("reasoning_effort", None)
            if out.get("reasoning_effort") is None:
                out["reasoning_effort"] = "none"
    # Explicit flat effort has precedence over nested effort.
    effort = out.get("reasoning_effort")
    if effort is not None:
        r.pop("effort", None)
        r.pop("reasoning_effort", None)
    else:
        effort = r.get("effort") if r.get("effort") is not None else r.get("reasoning_effort")
    if isinstance(effort, bool):
        raise ValueError("reasoning_effort must not be a boolean")
    if isinstance(effort, int) and effort != 0:
        if not 1 <= effort <= 100:
            raise ValueError("an integer reasoning_effort budget must be in [1, 100]")
        out.pop("reasoning_effort", None)
        r.pop("effort", None)
        r.pop("reasoning_effort", None)
        ctk = dict(out.get("chat_template_kwargs") or {})
        ctk.setdefault("reasoning_effort", effort)
        ctk.setdefault("thinking", True)
        ctk.setdefault("enable_thinking", True)
        out["chat_template_kwargs"] = ctk
    elif isinstance(out.get("chat_template_kwargs"), dict):
        out["chat_template_kwargs"] = dict(out["chat_template_kwargs"])
    if "exclude" in r and "reasoning_exclude" not in out:
        exclude = r["exclude"]
        out["reasoning_exclude"] = exclude.strip().lower() in truthy if isinstance(exclude, str) else bool(exclude)
    if isinstance(out.get("reasoning"), dict):
        out["reasoning"] = r
    if out.get("stream"):
        options = dict(out.get("stream_options") or {})
        options.update(include_usage=True, continuous_usage_stats=True)
        out["stream_options"] = options
    return out
