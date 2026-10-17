"""CPU-only request-aware Xing4 structured response reasoning regressions."""

import random
import unittest

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.parser.reasoning_parser import ReasoningParser, Xing4_0Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1, "base-a-test-cpu")


def chunkings(text):
    for split in range(len(text) + 1):
        yield [text[:split], text[split:]]
    yield list(text)
    for seed in range(8):
        rng = random.Random(seed)
        chunks, start = [""], 0
        while start < len(text):
            end = start + rng.randint(1, 13)
            chunks.extend((text[start:end], ""))
            start = end
        yield chunks


class TestXing4StructuredReasoning(unittest.TestCase):
    def assert_parsing(self, factory, text, expected):
        parser = factory()
        self.assertEqual(parser.parse_non_stream(text), expected)
        for chunks in chunkings(text):
            with self.subTest(chunks=chunks):
                parser = factory()
                results = [parser.parse_stream_chunk(chunk) for chunk in chunks]
                results.append(parser.parse_stream_end())
                self.assertEqual(
                    tuple("".join(part[i] or "" for part in results) for i in (0, 1)),
                    expected,
                )
                self.assertEqual(parser.parse_stream_end(), ("", ""))

    def test_structured_request_matrix(self):
        bodies = [
            '{"value":"<think>literal</think>"}',
            ' {"value":"</think><think><think>","unicode":"雪"}\n',
            '{"value":"<think>"}',
            '{"ok":true}',
            # No schema-aware body parsing: ordinary text is also verbatim.
            "normal text <think>literal</think> and trailing <thi",
            "<think>body marker at the start</think>",
        ]
        for format_type in ("json_schema", "json_object"):
            response_format = {"type": format_type}
            if format_type == "json_schema":
                response_format["json_schema"] = {
                    "name": "body",
                    "schema": {"type": "object"},
                }
            for thinking in (False, True):
                for stream_reasoning in (False, True):
                    for force_nonempty in (False, True):
                        request = ChatCompletionRequest(
                            model="xing4", messages=[{"role": "user", "content": "test"}],
                            response_format=response_format,
                            chat_template_kwargs={
                                "enable_thinking": thinking,
                                "force_nonempty_content": force_nonempty,
                            },
                        )

                        def factory():
                            return ReasoningParser(
                                "xing4_0",
                                request=request,
                                force_reasoning=thinking,
                                stream_reasoning=stream_reasoning,
                            )

                        with self.subTest(
                            format=format_type,
                            thinking=thinking,
                            stream=stream_reasoning,
                            force=force_nonempty,
                        ):
                            self.assertTrue(factory().detector.structured_output)
                            self.assertFalse(factory().detector._force_nonempty_content)
                            for body in bodies:
                                prefixes = (
                                    ("analysis</think>", "<think>analysis</think>")
                                    if thinking
                                    else ("",)
                                )
                                for prefix in prefixes:
                                    self.assert_parsing(
                                        factory,
                                        prefix + body,
                                        ("analysis" if thinking else "", body),
                                    )
                            if thinking:
                                for text, reasoning in (
                                    ("NOT JSON", "NOT JSON"),
                                    ("<think>NOT JSON", "NOT JSON"),
                                    ("unfinished</thi", "unfinished</thi"),
                                    ("<thi", "<thi"),
                                    ("", ""),
                                ):
                                    self.assert_parsing(factory, text, (reasoning, ""))

    def test_explicit_detector_force_nonempty_is_disabled(self):
        for force in (False, True):
            for stream in (False, True):
                detector = Xing4_0Detector(
                    force_reasoning=True,
                    force_nonempty_content=force,
                    stream_reasoning=stream,
                    structured_output=True,
                )
                self.assertFalse(detector._force_nonempty_content)
                self.assertEqual(detector.detect_and_parse("NOT JSON").normal_text, "")
                result = detector.parse_streaming_increment("NOT JSON")
                final = detector.finish()
                self.assertEqual(result.normal_text + final.normal_text, "")
                self.assertEqual(result.reasoning_text + final.reasoning_text, "NOT JSON")

    def test_nonstructured_legacy_and_other_models(self):
        for response_format in (None, {"type": "text"}):
            request = ChatCompletionRequest(
                model="xing4", messages=[{"role": "user", "content": "test"}],
                response_format=response_format,
            )
            parser = ReasoningParser("xing4_0", request=request, force_reasoning=True)
            self.assertFalse(parser.detector.structured_output)
            self.assertEqual(parser.parse_non_stream("NOT JSON"), ("", "NOT JSON"))
            parser = ReasoningParser("xing4_0", request=request, force_reasoning=False)
            self.assertEqual(
                parser.parse_non_stream("before<think>inside</think>after"),
                ("before<think>inside", "after"),
            )
        for stream_reasoning in (False, True):
            parser = ReasoningParser(
                "xing4_0", force_reasoning=True, stream_reasoning=stream_reasoning
            )
            expected_reasoning = "NOT JSON" if stream_reasoning else ""
            self.assertEqual(
                parser.parse_stream_chunk("NOT JSON"), (expected_reasoning, "")
            )
            self.assertEqual(parser.parse_stream_end(), ("", "NOT JSON"))

            # Ordinary chat intentionally retains the legacy marker detection,
            # even when a subsequent chunk opens a second reasoning block.
            parser = ReasoningParser(
                "xing4_0", force_reasoning=True, stream_reasoning=stream_reasoning
            )
            self.assertEqual(parser.parse_stream_chunk("analysis</think>"),
                             ("analysis", ""))
            self.assertEqual(
                parser.parse_stream_chunk('{"value":"<think>literal</think>"}'),
                ('{"value":"literal', '"}'),
            )
        request = ChatCompletionRequest(
            model="other", messages=[{"role": "user", "content": "test"}],
            response_format={"type": "json_object"},
        )
        parser = ReasoningParser("qwen3", request=request, force_reasoning=False)
        self.assertFalse(hasattr(parser.detector, "structured_output"))
        self.assertEqual(
            parser.parse_non_stream("before<think>inside</think>after"),
            ("before<think>inside", "after"),
        )


if __name__ == "__main__":
    unittest.main()
