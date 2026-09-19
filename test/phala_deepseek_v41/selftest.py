"""Tests against installed source-integrated adapters; no runtime package mounts."""

import sys
from types import SimpleNamespace

_failures = []
_checks = 0


def check(label, got, want):
    global _checks
    _checks += 1
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'}  {label}\n        got {got!r}" + ("" if ok else f"\n        want {want!r}"))
    if not ok:
        _failures.append(label)


def check_true(label, got):
    check(label, bool(got), True)


def section(name):
    print(f"\n--- {name} " + "-" * max(0, 60 - len(name)))


# ==========================================================================
section("1: common tool and media adapters composed")
# ==========================================================================
from sglang.srt.entrypoints.openai import protocol as P  # noqa: E402
from sglang.srt.entrypoints.openai import serving_chat as SC  # noqa: E402
from sglang.srt.multimodal.processors import base_processor as BP  # noqa: E402
from sglang.srt.managers.schedule_batch import Modality  # noqa: E402


check_true("tool_choice-none wrapper survived (dsv41_tool_choice_none)",
           getattr(SC.OpenAIServingChat._process_messages, "_dsv41_tool_choice_none", False))
check_true("media-hardening wrapper installed on the same class",
           getattr(SC.OpenAIServingChat._validate_media_content, "_dsv41_media_hardening", False))

# ==========================================================================
section("A/B: client media failures classify as 400, not 500")
# ==========================================================================
import base64  # noqa: E402
import struct  # noqa: E402
import zlib  # noqa: E402
from PIL import Image  # noqa: E402


def png(width, height):
    """1-bit grayscale PNG of the given size; a few KB whatever the pixel count."""
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    compressor = zlib.compressobj(9)
    row = b"\x00" + b"\x00" * ((width + 7) // 8)
    body = [compressor.compress(row) for _ in range(height)]
    body.append(compressor.flush())
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 1, 0, 0, 0, 0))
            + chunk(b"IDAT", b"".join(body)) + chunk(b"IEND", b""))


def data_uri(raw, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


check_true("base_processor.CLIENT_MEDIA_EXCEPTIONS covers OSError",
           OSError in BP.CLIENT_MEDIA_EXCEPTIONS)
check_true("base_processor.CLIENT_MEDIA_EXCEPTIONS covers DecompressionBombError",
           Image.DecompressionBombError in BP.CLIENT_MEDIA_EXCEPTIONS)

# PIL raises above 2x MAX_IMAGE_PIXELS; >89.5 MP only warns and still decodes.
bomb = png(14000, 13000)
check_true("bomb fixture is over PIL's hard limit",
           14000 * 13000 > 2 * Image.MAX_IMAGE_PIXELS)


def classify(data):
    try:
        BP.BaseMultimodalProcessor._load_single_item(data, Modality.IMAGE)
        return "loaded"
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__


# ValueError is what fast_load_mm_data re-raises as a 400; RuntimeError is the 500.
check("bare base64 JPEG -> 400 class",
      classify("/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVRPUg=="), "ValueError")
check("decompression bomb data URI -> 400 class", classify(data_uri(bomb)), "ValueError")
check("truncated data URI -> 400 class",
      classify(data_uri(b"\x89PNG\r\n\x1a\nbroken")), "ValueError")
check("unreachable http URL -> 400 class",
      classify("https://127.0.0.1:9/nope.png"), "ValueError")
check("valid tiny PNG data URI still loads", classify(data_uri(png(8, 8))), "loaded")

# ==========================================================================
section("A/C: request validation rejects unreadable and unsupported media")
# ==========================================================================
# Stand-in for the serving object, carrying the same capability flags the live
# replica reports at /get_model_info (image yes, audio no, no video token) and
# the same --media-url-max-file-size-mb as compose.yaml.
FAKE_CHAT = SimpleNamespace(
    tokenizer_manager=SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal=True,
            is_image_understandable_model=True,
            is_audio_understandable_model=False,
        ),
        mm_processor=SimpleNamespace(
            mm_tokens=SimpleNamespace(video_token=None, video_token_id=None)
        ),
        server_args=SimpleNamespace(media_url_max_file_size_mb=32),
    )
)
validate = SC.OpenAIServingChat._validate_media_content


def request(content, **extra):
    return P.ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": content}], **extra
    )


def media_error(content):
    return validate(FAKE_CHAT, request(content))


text_part = {"type": "text", "text": "what is this"}
good_png = data_uri(png(8, 8))

check("text only -> no error", media_error([text_part]), None)
check("data: URI image -> no error",
      media_error([text_part, {"type": "image_url", "image_url": {"url": good_png}}]), None)
check("https image -> no error",
      media_error([{"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}]), None)
check("bare base64 image -> 400",
      media_error([{"type": "image_url", "image_url": {"url": "/9j/4AAQSkZJRg=="}}]),
      "Invalid image_url.url: expected a data: URI or an http(s):// URL. "
      "Bare base64 payloads and local paths are not accepted.")
check("file:// image -> 400",
      media_error([{"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}]),
      "Invalid image_url.url: expected a data: URI or an http(s):// URL. "
      "Bare base64 payloads and local paths are not accepted.")
check("video_url -> 400",
      media_error([{"type": "video_url", "video_url": {"url": "https://example.com/nope.mp4"}}]),
      "video input is not supported by this model.")
check("audio_url -> 400",
      media_error([{"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}}]),
      "audio input is not supported by this model.")
check("input_audio -> 400",
      media_error([{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]),
      "audio input is not supported by this model.")

oversized = "data:image/png;base64," + "A" * (45 * 1024 * 1024)
check_true("data URI over --media-url-max-file-size-mb -> 400",
           str(media_error([{"type": "image_url", "image_url": {"url": oversized}}])
               ).endswith("byte media size limit (--media-url-max-file-size-mb)."))
under = "data:image/png;base64," + "A" * (30 * 1024 * 1024)  # ~22.5 MB decoded
check("30 MB base64 data URI still allowed (under the 32 MB limit)",
      media_error([{"type": "image_url", "image_url": {"url": under}}]), None)

# ==========================================================================
print(f"\n{_checks} checks, {len(_failures)} failed")
if _failures:
    for label in _failures:
        print(f"  FAILED: {label}")
    sys.exit(1)
print("all patches verified")
