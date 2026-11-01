"""CPU-only gate: JSON string data must survive XTML parsing unchanged."""

import inspect
import json

import pytest

from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOLS_OPEN,
)
from sglang.srt.parser.reasoning_parser import KimiK3Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def detector(thinking, **kwargs):
    # Run the identical data-preservation assertion on the pre-fix parser too.
    if "structured_json" in inspect.signature(KimiK3Detector).parameters:
        kwargs["structured_json"] = True
    return KimiK3Detector(force_reasoning=thinking, **kwargs)


@pytest.mark.parametrize(
    "marker",
    [THINK_OPEN, RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE, THINK_CLOSE, TOOLS_OPEN],
)
@pytest.mark.parametrize("thinking", [True, False])
def test_literal_marker_is_json_data(marker, thinking):
    value = {"text": "prefix " + marker + " suffix"}
    body = json.dumps(value, separators=(",", ":"))
    native = (
        (THINK_CLOSE if thinking else "")
        + RESPONSE_OPEN
        + body
        + RESPONSE_CLOSE
        + MESSAGE_CLOSE
    )
    parser = detector(thinking)
    assert json.loads(parser.detect_and_parse(native).normal_text) == value
    for width in (1, 7, 1000):
        parser = detector(thinking)
        content = "".join(
            (parser.parse_streaming_increment(native[i : i + width]).normal_text or "")
            for i in range(0, len(native), width)
        )
        assert json.loads(content) == value


@pytest.mark.parametrize(
    "value",
    [
        {"a": [{"b": 'escaped quote " and slash \\ ' + RESPONSE_CLOSE}], "c": True},
        [RESPONSE_OPEN, {"x": MESSAGE_CLOSE}],
        RESPONSE_CLOSE + ' " \\ end',
        12.5,
        True,
        None,
    ],
)
@pytest.mark.parametrize("thinking", [True, False])
def test_nested_escaped_and_scalar_json(value, thinking):
    body = json.dumps(value, separators=(",", ":"))
    native = (
        ("reason " + THINK_CLOSE if thinking else "")
        + RESPONSE_OPEN
        + body
        + RESPONSE_CLOSE
        + MESSAGE_CLOSE
    )
    for width in range(1, max(len(RESPONSE_OPEN), len(RESPONSE_CLOSE)) + 1):
        parser = detector(thinking)
        chunks = [
            parser.parse_streaming_increment(native[i : i + width])
            for i in range(0, len(native), width)
        ]
        chunks.append(parser.finish())
        assert json.loads("".join(x.normal_text for x in chunks)) == value
        assert "".join(x.reasoning_text for x in chunks) == (
            "reason " if thinking else ""
        )


def test_reasoning_json_is_not_content_and_reasoning_option_is_preserved():
    reasoning = '{"temporary":"reasoning example"}'
    body = '{"final":true}'
    for stream_reasoning in (True, False):
        parser = detector(True, stream_reasoning=stream_reasoning)
        first = parser.parse_streaming_increment(reasoning)
        assert not first.normal_text
        assert first.reasoning_text == (reasoning if stream_reasoning else "")
        second = parser.parse_streaming_increment(THINK_CLOSE + body)
        assert second.normal_text == body
        assert second.reasoning_text == ("" if stream_reasoning else reasoning)


def test_json_body_is_not_retained_and_choices_are_isolated():
    parsers = [detector(False), detector(True)]
    chunks = ['{"value":"' + "x" * 4096, '{"reason":"not final"}']
    a = parsers[0].parse_streaming_increment(chunks[0])
    b = parsers[1].parse_streaming_increment(chunks[1])
    assert a.normal_text == chunks[0] and not b.normal_text
    assert len(parsers[0]._json_content_parser.buffer) < len(RESPONSE_CLOSE)
    assert not parsers[0]._stream_text and not parsers[0]._streamed_reasoning
    a2 = parsers[0].parse_streaming_increment(RESPONSE_CLOSE + '"}' + RESPONSE_CLOSE)
    b2 = parsers[1].parse_streaming_increment(
        THINK_CLOSE + '{"ok":true}' + RESPONSE_CLOSE
    )
    assert json.loads(a.normal_text + a2.normal_text) == {
        "value": "x" * 4096 + RESPONSE_CLOSE
    }
    assert json.loads(b2.normal_text) == {"ok": True}


@pytest.mark.parametrize("backslashes", range(6))
@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("root", ["string", "object", "array"])
def test_exact_json_bytes_survive_every_escape_split(backslashes, thinking, root):
    text = (
        "\\" * backslashes
        + '" []{} '
        + THINK_OPEN
        + RESPONSE_CLOSE
        + THINK_CLOSE
        + " literal \\u003c and newline\n plus \u4e2d\u6587"
    )
    value = {
        "string": text,
        "object": {"nested": [text]},
        "array": [text, {"n": -12.5e3}],
    }[root]
    body = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    reason = "analysis " if thinking else ""
    native = (
        reason
        + (THINK_CLOSE if thinking else "")
        + RESPONSE_OPEN
        + body
        + RESPONSE_CLOSE
        + MESSAGE_CLOSE
    )
    for split in range(len(native) + 1):
        parser = detector(thinking)
        chunks = [
            parser.parse_streaming_increment(piece)
            for piece in (native[:split], "", native[split:])
        ]
        chunks.append(parser.finish())
        assert "".join(x.normal_text for x in chunks) == body, (
            root,
            backslashes,
            split,
        )
        assert "".join(x.reasoning_text for x in chunks) == reason


@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize(
    "fragment",
    [
        '{"value":"' + RESPONSE_CLOSE + "\\",
        '"' + MESSAGE_CLOSE + '\\\\"',
        '[{"nested":' + json.dumps(THINK_CLOSE) + ',"value":"unfinished',
        "true" + RESPONSE_CLOSE[:9],
        "12.5" + MESSAGE_CLOSE[:9],
    ],
)
def test_truncated_json_and_outer_markers_are_preserved(fragment, thinking):
    native = (THINK_CLOSE if thinking else "") + RESPONSE_OPEN + fragment
    for split in range(len(native) + 1):
        parser = detector(thinking)
        chunks = [
            parser.parse_streaming_increment(native[:split]),
            parser.parse_streaming_increment(native[split:]),
            parser.finish(),
        ]
        assert "".join(x.normal_text for x in chunks) == fragment, split


@pytest.mark.parametrize(
    "value",
    [
        [1, -12.5e3, True, None] * 256,
        {"nested": [{"empty": [], "object": {}}, ["brackets []{}", {"n": 42}]]},
        -1234567890.125,
    ],
)
def test_container_and_primitive_spans_keep_root_boundary(value):
    body = json.dumps(value, separators=(",", ":"))
    native = RESPONSE_OPEN + body + RESPONSE_CLOSE + MESSAGE_CLOSE
    for width in (1, 7, 256, len(native)):
        parser = detector(False)
        chunks = [
            parser.parse_streaming_increment(native[i : i + width])
            for i in range(0, len(native), width)
        ]
        chunks.append(parser.finish())
        assert "".join(x.normal_text for x in chunks) == body
        assert len(parser._json_content_parser.buffer) < len(RESPONSE_CLOSE)


def test_only_constrained_k3_chat_requests_select_mode():
    from types import SimpleNamespace

    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    for model in ("kimi_k3", "qwen3"):
        owner = SimpleNamespace(reasoning_parser=model)
        for kind, strict in [
            ("json_object", None),
            ("json_schema", True),
            ("json_schema", False),
            ("text", None),
        ]:
            request = SimpleNamespace(
                response_format=SimpleNamespace(
                    type=kind, json_schema=SimpleNamespace(strict=strict)
                )
            )
            expected = model == "kimi_k3" and (
                kind == "json_object" or kind == "json_schema" and strict is not False
            )
            assert (
                OpenAIServingChat._kimi_k3_structured_json(owner, request) == expected
            )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
