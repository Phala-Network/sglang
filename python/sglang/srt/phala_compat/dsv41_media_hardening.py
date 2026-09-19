"""Source-integrated adapter, converted from dsv41_media_hardening.py.
No import hooks or external runtime code paths.
"""

import functools


import sys


import traceback


_COMMON = "sglang.srt.utils.common"


_UTILS_PACKAGE = "sglang.srt.utils"


_BASE_PROCESSOR = "sglang.srt.multimodal.processors.base_processor"


_SERVING_CHAT = "sglang.srt.entrypoints.openai.serving_chat"


_MARK = "_dsv41_media_hardening"


_warned = False


_ALLOWED_URL_SCHEMES = ("data:", "http://", "https://")


_MEDIA_PART_MODALITY = {
    "image_url": "image",
    "video_url": "video",
    "audio_url": "audio",
    "input_audio": "audio",
}


def _widen_client_media_exceptions(module) -> None:
    from PIL import Image  # already imported by the time either target loads

    current = tuple(getattr(module, "CLIENT_MEDIA_EXCEPTIONS", ()))
    if not current:
        raise AttributeError(f"{module.__name__}.CLIENT_MEDIA_EXCEPTIONS is missing")
    extra = tuple(
        exc for exc in (OSError, Image.DecompressionBombError) if exc not in current
    )
    module.CLIENT_MEDIA_EXCEPTIONS = current + extra


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


def _patch_serving_chat(module) -> None:
    serving_class = module.OpenAIServingChat
    original = serving_class._validate_media_content
    if getattr(original, _MARK, False):
        return

    @functools.wraps(original)
    def _validate_media_content(self, request):
        error = original(self, request)
        if error:
            return error
        # Fail open on the request path too: a bug in the added checks must not
        # turn every media request into the 500 this module exists to remove.
        try:
            return _validate_media_parts(self, request)
        except Exception:  # noqa: BLE001
            global _warned
            if not _warned:
                _warned = True
                print(
                    "[dsv41-patches] media validation raised; skipping the added "
                    "checks for this and later requests' extra validation",
                    file=sys.stderr,
                )
                traceback.print_exc()
            return None

    setattr(_validate_media_content, _MARK, True)
    serving_class._validate_media_content = _validate_media_content
