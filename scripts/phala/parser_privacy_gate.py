"""CPU parser canaries and same-environment differential corpus, without weights."""

import argparse
import asyncio
import hashlib
import importlib
import inspect
import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MARKER = "SYNTHETIC_PRIVATE_PARSER_CANARY"


def normalize(value):
    if hasattr(value, "model_dump"):
        return normalize(value.model_dump())
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: normalize(getattr(value, key)) for key in value.__dataclass_fields__
        }
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--record-baseline", action="store_true")
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report exists")
    from sglang_simulator.simulation.sglang.hook_bootstrap import (
        install_simulator_hooks,
    )

    install_simulator_hooks()
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
    from sglang.srt.function_call import mimo_detector as mimo
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.function_call.hermes_detector import HermesDetector
    from sglang.srt.function_call.kimik3_detector import KimiK3Detector
    from sglang.srt.function_call.kimik3_format import TOOLS_CLOSE, TOOLS_OPEN
    from sglang.srt.function_call.muse_glimmer_detector import MuseGlimmerDetector
    from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

    captured = io.StringIO()
    handler = logging.StreamHandler(captured)
    logging.getLogger().handlers = [handler]
    logging.getLogger().setLevel(logging.DEBUG)
    rows = []

    def check(name, invoke):
        captured.seek(0)
        captured.truncate(0)
        try:
            invoke()
            text = captured.getvalue()
            rows.append(
                {
                    "case": name,
                    "passed": bool(text.strip())
                    and MARKER not in text
                    and "Traceback (" not in text,
                    "marker_logged": MARKER in text,
                    "traceback_logged": "Traceback (" in text,
                }
            )
        except Exception as error:
            rows.append(
                {"case": name, "passed": False, "test_error_type": type(error).__name__}
            )

    for kind in ("integer", "number", "boolean", "object", "array"):
        tool = Tool.model_validate(
            {
                "type": "function",
                "function": {
                    "name": MARKER,
                    "parameters": {
                        "type": "object",
                        "properties": {MARKER: {"type": kind}},
                    },
                },
            }
        )
        check(
            "mimo-coercion-" + kind,
            lambda tool=tool: mimo._convert_param_value(MARKER, MARKER, MARKER, [tool]),
        )
        detector = Qwen3CoderDetector()
        values = (MARKER, MARKER, {MARKER: {"type": kind}}, MARKER)
        if len(inspect.signature(detector._convert_param_value).parameters) == 5:
            values += ([],)
        check(
            "qwen-coercion-" + kind,
            lambda detector=detector, values=values: detector._convert_param_value(
                *values
            ),
        )

    def fault(detector, method, target, text):
        with patch.object(detector, target, side_effect=RuntimeError(MARKER)):
            getattr(detector, method)(text, [])

    check(
        "hermes-exception",
        lambda: fault(
            HermesDetector(),
            "detect_and_parse",
            "parse_base_json",
            '<tool_call>{"name":"x","arguments":{}}</tool_call>',
        ),
    )
    kimi = KimiK3Detector()
    check(
        "kimi-stream-exception",
        lambda: fault(
            kimi, "parse_streaming_increment", "_parse_calls", kimi.bot_token + MARKER
        ),
    )
    check(
        "muse-unknown-tool", lambda: MuseGlimmerDetector()._emit_call(MARKER, {}, set())
    )
    check(
        "qwen-unknown-tool",
        lambda: Qwen3CoderDetector().parse_base_json(
            {"name": MARKER, "arguments": {}}, []
        ),
    )

    def request_fault():
        def fail(_):
            raise RuntimeError(MARKER)

        owner = SimpleNamespace(
            _validate_request=fail, create_error_response=lambda **kw: kw
        )
        result = asyncio.run(
            OpenAIServingBase.handle_request(owner, SimpleNamespace(), None)
        )
        assert result["status_code"] == 500

    check("serving-request-exception", request_fault)
    tools = [
        Tool.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                    },
                },
            }
        )
    ]
    samples = [
        "",
        "ordinary answer",
        '{"name":"test_tool","arguments":{"value":3}}',
        '<tool_call>{"name":"test_tool","arguments":{"value":3}}</tool_call>',
        "<tool_call><function=test_tool><parameter=value>3</parameter></function></tool_call>",
        "<tool_call>test_tool<arg_key>value</arg_key><arg_value>3</arg_value></tool_call>",
        "<tool_call>{invalid " + MARKER,
    ]
    behavior = []
    for name in (
        "qwen3_coder",
        "gemma4",
        "glm",
        "glm47",
        "muse",
        "kimi_k2",
        "kimi_k3",
        "hermes",
        "mimo",
    ):
        for index, sample in enumerate(samples):
            for streaming in (False, True):
                try:
                    detector = FunctionCallParser(tools, name)
                    value = (
                        [
                            detector.parse_stream_chunk(sample[i : i + 3])
                            for i in range(0, len(sample), 3)
                        ]
                        if streaming
                        else detector.parse_non_stream(sample)
                    )
                    value = normalize(value)
                except Exception as error:
                    value = {
                        "exception_type": type(error).__name__,
                        "message": str(error),
                    }
                behavior.append(
                    {
                        "parser": name,
                        "sample": index,
                        "streaming": streaming,
                        "result": value,
                    }
                )
    fingerprint = hashlib.sha256(
        json.dumps(behavior, sort_keys=True).encode()
    ).hexdigest()
    positive_inputs = {
        "qwen3_coder": "<tool_call><function=test_tool><parameter=value>3</parameter></function></tool_call>",
        "gemma4": "<|tool_call>call:test_tool{value:3}<tool_call|>",
        "glm": "<tool_call>test_tool\n<arg_key>value</arg_key><arg_value>3</arg_value></tool_call>",
        "glm47": "<tool_call>test_tool<arg_key>value</arg_key><arg_value>3</arg_value></tool_call>",
        "muse": '<|start|>assistant to=test_tool<|message|><atem:function_calls><atem:invoke name="test_tool"><atem:parameter name="value">3</atem:parameter></atem:invoke></atem:function_calls>',
        "kimi_k3": TOOLS_OPEN
        + '<|open|>call tool="test_tool" index="1"<|sep|><|open|>argument key="value" type="integer"<|sep|>3<|close|>argument<|sep|><|close|>call<|sep|>'
        + TOOLS_CLOSE,
        "hermes": '<tool_call>{"name":"test_tool","arguments":{"value":3}}</tool_call>',
    }
    positive = []
    for name, text in positive_inputs.items():
        try:
            _, calls = FunctionCallParser(tools, name).parse_non_stream(text)
            assert len(calls) == 1 and calls[0].name == "test_tool"
            assert json.loads(calls[0].parameters) == {"value": 3}
            for size in (1, 3, 11):
                detector = FunctionCallParser(tools, name)
                fragments = []
                for offset in range(0, len(text), size):
                    _, pieces = detector.parse_stream_chunk(
                        text[offset : offset + size]
                    )
                    fragments.extend(pieces)
                assert any(call.name == "test_tool" for call in fragments)
            positive.append({"parser": name, "passed": True})
        except Exception as error:
            positive.append(
                {"parser": name, "passed": False, "error_type": type(error).__name__}
            )
    equal = None
    if args.compare:
        before = json.loads(args.compare.read_text())
        equal = (
            before["behavior"] == behavior and before["behavior_sha256"] == fingerprint
        )
    module = importlib.import_module("sglang.srt.function_call.function_call_parser")
    report = {
        "scope": "source-mounted-cpu-parser-contracts-not-final-image-or-gpu",
        "record_only": args.record_baseline,
        "imported_source": module.__file__,
        "privacy_cases": rows,
        "privacy_passed": all(row["passed"] for row in rows),
        "positive_cases": positive,
        "positive_passed": all(row["passed"] for row in positive),
        "behavior": behavior,
        "behavior_sha256": fingerprint,
        "matches_baseline": equal,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    handler.close()
    print(
        json.dumps({key: value for key, value in report.items() if key != "behavior"})
    )
    if args.record_baseline:
        return 0
    return (
        0
        if report["privacy_passed"] and report["positive_passed"] and equal is not False
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
