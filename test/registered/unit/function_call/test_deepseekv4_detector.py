"""Unit tests for DeepSeekV4Detector DSML streaming — no server, no model loading."""

import json
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

DSML = "｜DSML｜"


def _wrapped(invoke: str) -> str:
    return f"<{DSML}tool_calls>\n{invoke}\n</{DSML}tool_calls>"


def _invoke(name: str, params: str = "") -> str:
    return f'<{DSML}invoke name="{name}">\n{params}\n</{DSML}invoke>'


def _param(name: str, is_string: str, value: str) -> str:
    return (
        f'<{DSML}parameter name="{name}" string="{is_string}">{value}</{DSML}parameter>'
    )


def _weather_call(city: str = "SF") -> str:
    return _wrapped(_invoke("get_weather", _param("city", "true", city)))


class TestDeepSeekV4Streaming(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            )
        ]

    def _feed(self, chunks):
        """Returns (normal_text, calls) accumulated over the chunks."""
        detector = DeepSeekV4Detector()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        return normal, calls

    def test_preamble_in_same_delta_as_tool_call(self):
        """Prose sharing a delta with the tool call must not be dropped, and the
        streaming and one-shot paths must agree on it."""
        text = "Let me check.\n" + _weather_call()
        normal, calls = self._feed([text])

        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual(
            normal, DeepSeekV4Detector().detect_and_parse(text, self.tools).normal_text
        )

    def test_preamble_before_bare_invoke_without_wrapper(self):
        """The bare `<｜DSML｜invoke …>` form has no tool_calls wrapper to walk
        back to, so the preamble is computed from the invoke itself."""
        text = "Checking.\n" + _invoke("get_weather", _param("city", "true", "SF"))
        normal, calls = self._feed([text])

        self.assertIn("Checking.", normal)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])

    def test_no_dsml_markers_leak_into_normal_text(self):
        text = "Prose.\n" + _weather_call()
        normal, _ = self._feed([text[i : i + 4] for i in range(0, len(text), 4)])

        self.assertNotIn(DSML, normal)

    def test_malformed_partial_json_falls_back_to_raw_value(self):
        """A partial non-string parameter must not escape as MalformedJSON."""
        detector = DeepSeekV4Detector()
        result = detector.parse_streaming_increment(
            f'<{DSML}tool_calls>\n<{DSML}invoke name="get_weather">\n'
            f'<{DSML}parameter name="city" string="false">{{"a"',
            self.tools,
        )

        self.assertEqual(result.calls, [])
        self.assertNotEqual(detector._buffer, "")

    def test_non_streaming_parses_every_tool_calls_section(self):
        """A turn with two tool_calls sections must yield both calls."""
        result = DeepSeekV4Detector().detect_and_parse(
            f"{_weather_call('SF')}\n{_weather_call('NY')}", self.tools
        )

        self.assertEqual(len(result.calls), 2)

    def test_non_streaming_repeated_calls_use_sequential_indices(self):
        """Repeated calls to one tool use response ordinals, not tool slots."""
        result = DeepSeekV4Detector().detect_and_parse(
            "\n".join(_weather_call(city) for city in ("SF", "NY", "LA")),
            self.tools,
        )

        self.assertEqual([call.name for call in result.calls], ["get_weather"] * 3)
        self.assertEqual([call.tool_index for call in result.calls], [0, 1, 2])

    def test_structural_tag_honors_parallel_tool_calls_false(self):
        import xgrammar as xgr

        structural_tag = DeepSeekV4Detector().get_structural_tag(
            self.tools,
            tool_choice="required",
            parallel_tool_calls=False,
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        xgr.Grammar.from_structural_tag(structural_tag)

        def stop_flags(node):
            flags = []
            if isinstance(node, dict):
                if node.get("type") == "tags_with_separator":
                    flags.append(node.get("stop_after_first"))
                for child in node.values():
                    flags.extend(stop_flags(child))
            elif isinstance(node, list):
                for child in node:
                    flags.extend(stop_flags(child))
            return flags

        flags = stop_flags(structural_tag.model_dump().get("format"))
        self.assertTrue(flags)
        self.assertTrue(all(flags))

        parallel_tag = DeepSeekV4Detector().get_structural_tag(
            self.tools,
            tool_choice="required",
            parallel_tool_calls=True,
        )
        parallel_flags = stop_flags(parallel_tag.model_dump().get("format"))
        self.assertTrue(parallel_flags)
        self.assertTrue(all(flag is False for flag in parallel_flags))

    def test_streaming_call_is_atomic_for_chunk_sizes(self):
        text = _weather_call("Paris")
        close = f"</{DSML}invoke>"
        prefix, suffix = text.split(close, 1)

        for chunk_size in (1, 7, len(prefix)):
            detector = DeepSeekV4Detector()
            for offset in range(0, len(prefix), chunk_size):
                result = detector.parse_streaming_increment(
                    prefix[offset : offset + chunk_size], self.tools
                )
                self.assertEqual(result.calls, [])

            result = detector.parse_streaming_increment(close + suffix, self.tools)
            self.assertEqual(len(result.calls), 1)
            self.assertEqual(result.calls[0].tool_index, 0)
            self.assertEqual(result.calls[0].name, "get_weather")
            self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Paris"})

    def test_truncated_call_emits_no_tool_delta(self):
        detector = DeepSeekV4Detector()
        text = _weather_call("Paris")
        prefix = text.split(f"</{DSML}invoke>", 1)[0]

        incremental = detector.parse_streaming_increment(prefix, self.tools)
        terminal = detector.finish(self.tools)

        self.assertEqual(incremental.calls, [])
        self.assertEqual(terminal.calls, [])

    def test_streaming_drops_unknown_then_parses_known(self):
        text = _wrapped(
            _invoke("missing", _param("city", "true", "bad"))
            + _invoke("get_weather", _param("city", "true", "Paris"))
        )
        result = DeepSeekV4Detector().parse_streaming_increment(text, self.tools)

        self.assertEqual([call.name for call in result.calls], ["get_weather"])
        self.assertEqual([call.tool_index for call in result.calls], [0])
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Paris"})

    def test_streaming_forwards_unknown_when_enabled(self):
        text = _wrapped(
            _invoke("missing", _param("city", "true", "bad"))
            + _invoke("get_weather", _param("city", "true", "Paris"))
        )
        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            result = DeepSeekV4Detector().parse_streaming_increment(text, self.tools)

        self.assertEqual(
            [call.name for call in result.calls], ["missing", "get_weather"]
        )
        self.assertEqual([call.tool_index for call in result.calls], [0, 1])

    def test_streaming_repeated_calls_use_sequential_indices(self):
        text = _wrapped(
            "".join(
                _invoke("get_weather", _param("city", "true", city))
                for city in ("SF", "NY", "LA")
            )
        )
        result = DeepSeekV4Detector().parse_streaming_increment(text, self.tools)

        self.assertEqual([call.tool_index for call in result.calls], [0, 1, 2])

    def test_parse_error_neither_swallows_nor_duplicates(self):
        """An unexpected parse error must retain the buffer for retry; only
        the preamble (text before the first DSML tag) is emitted as
        normal_text so the tool-call text is neither swallowed permanently
        nor duplicated across deltas."""
        detector = DeepSeekV4Detector()

        with patch.object(
            DeepSeekV4Detector,
            "_parse_parameters_from_xml",
            side_effect=RuntimeError("boom"),
        ):
            first = detector.parse_streaming_increment(_weather_call(), self.tools)
            # Buffer is retained for retry — NOT cleared
            self.assertNotEqual(detector._buffer, "")

        # Mock removed — the retained buffer should now parse successfully
        # on the next delta, proving the retry works.
        second = detector.parse_streaming_increment(" tail", self.tools)

        # _weather_call() has no preamble, so first.normal_text is empty.
        self.assertEqual(first.calls, [])
        # The tool call is emitted as a call, NOT as normal_text
        self.assertNotIn("get_weather", second.normal_text)
        self.assertTrue(any(c.name == "get_weather" for c in second.calls))


if __name__ == "__main__":
    import unittest

    unittest.main()
