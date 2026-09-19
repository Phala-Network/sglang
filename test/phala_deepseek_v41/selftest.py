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
section("1: patch layer loaded, and all four patch modules composed")
# ==========================================================================
from sglang.srt.entrypoints.openai import protocol as P  # noqa: E402
from sglang.srt.entrypoints.openai import serving_chat as SC  # noqa: E402
from sglang.srt.entrypoints.openai import usage_processor as UP  # noqa: E402
from sglang.srt.entrypoints.openai import encoding_dsv41 as ENC  # noqa: E402
from sglang.srt.multimodal.processors import base_processor as BP  # noqa: E402
from sglang.srt.managers.schedule_batch import Modality  # noqa: E402

from sglang.srt.phala_compat import dsv41_protocol_compat as PC  # noqa: E402

check("reasoning-effort table re-anchored (dsv41_reasoning_effort)",
      ENC.REASONING_EFFORT_MAPPINGS,
      {"minimal": 25, "low": 50, "medium": 62, "high": 75, "xhigh": 90, "max": 100})
check_true("tool_choice-none wrapper survived (dsv41_tool_choice_none)",
           getattr(SC.OpenAIServingChat._process_messages, "_dsv41_tool_choice_none", False))
check_true("media-hardening wrapper installed on the same class",
           getattr(SC.OpenAIServingChat._validate_media_content, "_dsv41_media_hardening", False))
check_true("reasoning-exclude wrappers installed on the same class",
           getattr(SC.OpenAIServingChat._build_chat_response, "_dsv41_protocol_compat", False)
           and getattr(SC.OpenAIServingChat._process_reasoning_stream, "_dsv41_protocol_compat", False))

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
section("D: reasoning protocol translation")
# ==========================================================================
def build(**extra):
    try:
        r = P.ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], **extra)
    except Exception as exc:  # noqa: BLE001
        return ("error", str(exc))
    return (r.reasoning_effort, r.chat_template_kwargs, r.reasoning_exclude)


OFF = ("none", {"thinking": False, "enable_thinking": False}, False)
ON_HIGH = ("high", {"thinking": True, "enable_thinking": True}, False)
ABSENT = (None, None, False)

check("reasoning.enabled=false disables thinking", build(reasoning={"enabled": False}), OFF)
check("reasoning.enable='false' disables thinking", build(reasoning={"enable": "false"}), OFF)
check("reasoning.enabled=false with nested effort still disables",
      build(reasoning={"enabled": False, "effort": "high"}), OFF)
check("reasoning.enabled=true unchanged",
      build(reasoning={"enabled": True}), (None, {"thinking": True, "enable_thinking": True}, False))
check("reasoning_effort='default' is treated as absent", build(reasoning_effort="default"), ABSENT)
check("reasoning.effort='default' is treated as absent", build(reasoning={"effort": "default"}), ABSENT)
check("reasoning_effort=50 routes to chat_template_kwargs",
      build(reasoning_effort=50),
      (None, {"reasoning_effort": 50, "thinking": True, "enable_thinking": True}, False))
check("reasoning.effort=50 routes to chat_template_kwargs",
      build(reasoning={"effort": 50}),
      (None, {"reasoning_effort": 50, "thinking": True, "enable_thinking": True}, False))
check("reasoning_effort=100 accepted",
      build(reasoning_effort=100),
      (None, {"reasoning_effort": 100, "thinking": True, "enable_thinking": True}, False))
check_true("reasoning_effort=101 is a 400 naming the range",
           "[1, 100]" in str(build(reasoning_effort=101)[1]))
check_true("reasoning_effort=-1 is a 400 naming the range",
           "[1, 100]" in str(build(reasoning_effort=-1)[1]))
check("reasoning.exclude is recorded",
      build(reasoning={"exclude": True, "effort": "high"}), ("high", {"thinking": True, "enable_thinking": True}, True))
# Regressions: everything that worked before must still work, unchanged.
check("reasoning_effort='high' unchanged", build(reasoning_effort="high"), ON_HIGH)
check("reasoning_effort='none' unchanged", build(reasoning_effort="none"), OFF)
check("reasoning_effort=0.5 float unchanged",
      build(reasoning_effort=0.5), (0.5, {"thinking": True, "enable_thinking": True}, False))
check("reasoning_effort=0 stays a float effort",
      build(reasoning_effort=0), (0.0, {"thinking": True, "enable_thinking": True}, False))
check("no reasoning fields at all unchanged", build(), ABSENT)
check("chat_template_kwargs budget still passes through untouched",
      build(chat_template_kwargs={"reasoning_effort": 50}), (None, {"reasoning_effort": 50}, False))
check("explicit chat_template_kwargs wins over a routed budget",
      build(reasoning_effort=50, chat_template_kwargs={"reasoning_effort": 7}),
      (None, {"reasoning_effort": 7, "thinking": True, "enable_thinking": True}, False))

# The encoder must accept what we route to it.
from sglang.srt.entrypoints.openai import chat_encoding  # noqa: E402

check("encoder accepts the routed budget", chat_encoding.parse_dsv41_reasoning_effort(50), 50)
check("encoder preserves both integer budget boundaries",
      [chat_encoding.parse_dsv41_reasoning_effort(value) for value in (1, 100)], [1, 100])
check("encoder preserves legal OpenAI fractional efforts",
      [chat_encoding.parse_dsv41_reasoning_effort(value) for value in (0.0, 0.5, 0.99)],
      [1, 50, 99])
check("invalid budget values do not become an accepted effort",
      [chat_encoding.parse_dsv41_reasoning_effort(value) for value in (True, 0, 101, -0.1, 1.0)],
      [None, None, None, None, None])
check("serving resolver uses the parsed integer instead of the deployment default",
      SC.OpenAIServingChat._resolve_dsv41_reasoning_effort(
          SimpleNamespace(_dsv41_default_reasoning_effort="high"), 50), 50)
check("serving resolver retains the deployment default when effort is absent",
      SC.OpenAIServingChat._resolve_dsv41_reasoning_effort(
          SimpleNamespace(_dsv41_default_reasoning_effort="high"), None), "high")

# ==========================================================================
section("D3: reasoning.exclude withholds reasoning_content")
# ==========================================================================
class _StubChat:
    """Stand-in for OpenAIServingChat carrying only the two patched methods."""

    def _build_chat_response(self, request, *args, **kwargs):
        return P.ChatCompletionResponse(
            id="x", created=1, model="m",
            choices=[P.ChatCompletionResponseChoice(
                index=0,
                message=P.ChatMessage(role="assistant", content="answer",
                                      reasoning_content="secret"),
                finish_reason="stop")],
            usage=P.UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def _process_reasoning_stream(self, index, delta, parsers, content, request, *a, **kw):
        return "secret", "answer"


PC._patch_reasoning_exclude(SimpleNamespace(OpenAIServingChat=_StubChat))
stub = _StubChat()
for label, excluded, want_reasoning in (("excluded", True, None), ("kept", False, "secret")):
    req = P.ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}],
                                  reasoning={"exclude": excluded})
    response = stub._build_chat_response(req, [], 0)
    check(f"non-stream reasoning_content {label}",
          response.choices[0].message.reasoning_content, want_reasoning)
    check(f"non-stream content intact when {label}", response.choices[0].message.content, "answer")
    check(f"stream reasoning delta {label}",
          stub._process_reasoning_stream(0, "d", {}, {}, req), (want_reasoning, "answer"))

# ==========================================================================
section("D4: continuous usage forced on every streaming request")
# ==========================================================================
def stream_opts(**extra):
    r = P.ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], **extra)
    so = r.stream_options
    return (so.include_usage, so.continuous_usage_stats) if so is not None else None


check("stream, no stream_options at all -> both forced true",
      stream_opts(stream=True), (True, True))
check("stream, stream_options={include_usage: true} only -> continuous_usage_stats added",
      stream_opts(stream=True, stream_options={"include_usage": True}), (True, True))
check("stream, stream_options={continuous_usage_stats: true} only -> include_usage added",
      stream_opts(stream=True, stream_options={"continuous_usage_stats": True}), (True, True))
check("stream, both already true -> left true, not re-created",
      stream_opts(stream=True,
                  stream_options={"include_usage": True, "continuous_usage_stats": True}),
      (True, True))
check("stream, both explicitly false -> forced true anyway (deployment policy, not client choice)",
      stream_opts(stream=True,
                  stream_options={"include_usage": False, "continuous_usage_stats": False}),
      (True, True))
check("non-stream request, no stream_options -> left absent",
      stream_opts(stream=False), None)
check("non-stream request with explicit stream_options -> left exactly as sent",
      stream_opts(stream=False, stream_options={"include_usage": False}), (False, False))
check("stream defaults to false when omitted -> stream_options left absent",
      stream_opts(), None)

# ==========================================================================
section("E: usage.completion_tokens_details.reasoning_tokens")
# ==========================================================================
usage = UP.UsageProcessor.calculate_token_usage(
    prompt_tokens=11, completion_tokens=9, reasoning_tokens=4
)
check("flat reasoning_tokens kept", usage.reasoning_tokens, 4)
check("nested details built", usage.completion_tokens_details, {"reasoning_tokens": 4})
check("continuous-usage stream dict carries the nested details",
      usage.model_dump()["completion_tokens_details"], {"reasoning_tokens": 4})

non_stream = P.ChatCompletionResponse(id="x", created=1, model="m", choices=[], usage=usage)
check("non-stream response keeps the nested details after serialization",
      non_stream.model_dump()["usage"]["completion_tokens_details"], {"reasoning_tokens": 4})
stream = P.ChatCompletionStreamResponse(id="x", created=1, model="m", choices=[], usage=usage)
check("final stream chunk keeps the nested details after serialization",
      stream.model_dump()["usage"]["completion_tokens_details"], {"reasoning_tokens": 4})
check("a non-reasoning response reports zero, not null",
      UP.UsageProcessor.calculate_token_usage(prompt_tokens=1, completion_tokens=1).model_dump()[
          "completion_tokens_details"], {"reasoning_tokens": 0})
completion = P.CompletionResponse(id="x", created=1, model="m", choices=[], usage=usage)
check("/v1/completions usage shape deliberately unchanged",
      "completion_tokens_details" in completion.model_dump()["usage"], False)

# ==========================================================================
section("F/G: message roles in the native V4.1 encoder")
# ==========================================================================
# Expected prompts are DeepSeek's own reference encoder's output
# (checkpoint encoding/encoding.py) for the equivalent message lists.
import os  # noqa: E402

EFFORT_PREFIX = ("<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cSystem\uff5c>Reasoning Effort: 75 "
                 "(range 1-100, the higher the value, the more thorough the reasoning)\n\n")
DEV_MESSAGES = [
    {"role": "user", "content": "I need help identifying where this came from."},
    {"role": "assistant", "content": "Sure, please send me the full text."},
    {"role": "developer", "content": "You must start your response with AFFIRMATIVE"},
    {"role": "user", "content": "Whats my name?"},
]
SYS_MESSAGES = [{"role": "system", "content": "You are a helpful assistant."}]


def render(messages, mode="thinking"):
    return ENC.encode_messages(messages, thinking_mode=mode, reasoning_effort=75)


check_true("message-roles wrapper installed on encode_messages",
           getattr(ENC.encode_messages, "_dsv41_message_roles", False))
check("mid-conversation developer renders as a <System> block (reference-identical)",
      render(DEV_MESSAGES),
      EFFORT_PREFIX
      + "<\uff5cUser\uff5c>I need help identifying where this came from."
        "<\uff5cAssistant\uff5c></think>Sure, please send me the full text."
        "<\uff5cend\u2581of\u2581sentence\uff5c>"
        "<\uff5cSystem\uff5c>You must start your response with AFFIRMATIVE"
        "<\uff5cUser\uff5c>Whats my name?<\uff5cAssistant\uff5c><think>")
check("the caller's message list is not mutated",
      [m["role"] for m in DEV_MESSAGES], ["user", "assistant", "developer", "user"])
check("a leading developer message also becomes the system prompt",
      render([{"role": "developer", "content": "Be terse."}, {"role": "user", "content": "hi"}]),
      EFFORT_PREFIX.replace("\n\n", "\n\n") + "Be terse."
      + "<\uff5cUser\uff5c>hi<\uff5cAssistant\uff5c><think>")
sole = render([{"role": "developer", "content": "Be terse."}])
check("a developer-only conversation keeps its generation header",
      sole.endswith("<\uff5cAssistant\uff5c><think>") and "Be terse." in sole, True)

os.environ.pop("DSV41_SYSTEM_ONLY_USER_TURN", None)
check("system-only conversation has no generation point by default (as upstream)",
      render(SYS_MESSAGES),
      EFFORT_PREFIX + "You are a helpful assistant.")
os.environ["DSV41_SYSTEM_ONLY_USER_TURN"] = "1"
check("DSV41_SYSTEM_ONLY_USER_TURN=1 adds the empty user turn (reference-identical)",
      render(SYS_MESSAGES),
      EFFORT_PREFIX + "You are a helpful assistant."
      + "<\uff5cUser\uff5c><\uff5cAssistant\uff5c><think>")
check("the opt-in does not touch a conversation that already has a user turn",
      render(SYS_MESSAGES + [{"role": "user", "content": "hi"}]),
      EFFORT_PREFIX + "You are a helpful assistant."
      + "<\uff5cUser\uff5c>hi<\uff5cAssistant\uff5c><think>")
os.environ.pop("DSV41_SYSTEM_ONLY_USER_TURN", None)

# ==========================================================================
section("F: the real FastAPI app builds and validates with the rebuilt models")
# ==========================================================================
# Strongest pre-restart check: registering /v1/chat/completions makes FastAPI build
# its own body field from ChatCompletionRequest, which is the path a live request
# takes. Costs ~20 s (imports the whole server module; still no GPU).
import sglang.srt.entrypoints.http_server as H  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402

chat_route = [r for r in H.app.routes
              if isinstance(r, APIRoute) and r.path == "/v1/chat/completions"][0]
body_field = chat_route.body_field
check("chat route body field is the patched model",
      "reasoning_exclude" in body_field.field_info.annotation.model_fields, True)


def validate_body(**extra):
    value, errors = body_field.validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra}, {}, loc=("body",))
    return "errors" if errors else (value.reasoning_effort, value.chat_template_kwargs)


check("through FastAPI: reasoning_effort='default'",
      validate_body(reasoning_effort="default"), (None, None))
check("through FastAPI: reasoning.enabled=false",
      validate_body(reasoning={"enabled": False}), ("none", {"thinking": False, "enable_thinking": False}))
check("through FastAPI: integer budget",
      validate_body(reasoning_effort=50),
      (None, {"reasoning_effort": 50, "thinking": True, "enable_thinking": True}))


def validate_stream_options(**extra):
    value, errors = body_field.validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra}, {}, loc=("body",))
    if errors:
        return "errors"
    so = value.stream_options
    return (so.include_usage, so.continuous_usage_stats) if so is not None else None


check("through FastAPI: stream with no stream_options -> both forced true",
      validate_stream_options(stream=True), (True, True))
check("through FastAPI: non-stream request -> stream_options untouched",
      validate_stream_options(stream=False), None)

# ==========================================================================
print(f"\n{_checks} checks, {len(_failures)} failed")
if _failures:
    for label in _failures:
        print(f"  FAILED: {label}")
    sys.exit(1)
print("all patches verified")
