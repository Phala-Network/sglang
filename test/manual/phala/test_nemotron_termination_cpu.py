"""Execute Nemotron termination/budget source without importing GPU infrastructure."""

import ast
import itertools
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import test_nemotron_token_context_cpu as token_context
from test_nemotron_token_context_cpu import (
    LiteralTokenizer,
    load_source_methods,
    ordinary,
)

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    for name in names:
        body = tree.body
        for part in name.split("."):
            node = next(n for n in body if getattr(n, "name", None) == part)
            body = getattr(node, "body", [])
        selected.append(node)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias("annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *selected], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )


class TerminationTests(unittest.TestCase):
    def parser(self, stream=True, force=True, tokenizer=None):
        parser = load_source_methods()["ReasoningParser"](
            "nemotron_3",
            force_reasoning=True,
            stream_reasoning=stream,
            tokenizer=tokenizer,
        )
        parser.detector._force_nonempty_content = force
        return parser

    def test_unfinished_thought_never_becomes_content_on_cut(self):
        text = "Example <tool_call><function=f></function></tool_call> unfinished"
        for finish, stream, force, width in itertools.product(
            ("length", "abort"), (False, True), (False, True), (1, 7, 1000)
        ):
            with self.subTest(finish=finish, stream=stream, force=force, width=width):
                parser = self.parser(stream, force)
                self.assertEqual(parser.parse_non_stream(text, finish), (text, ""))
                parser = self.parser(stream, force)
                parts = [
                    parser.parse_stream_chunk(text[i : i + width])
                    for i in range(0, len(text), width)
                ]
                parts.append(parser.parse_stream_end(finish))
                self.assertEqual("".join(r for r, _ in parts), text)
                self.assertEqual("".join(c for _, c in parts), "")

    def test_normal_eof_retains_compatibility_without_token_context(self):
        text = "Need a tool.<tool_call>f</tool_call>"
        for stream, width in itertools.product((False, True), (1, 7, 1000)):
            parser = self.parser(stream, False)
            parts = [
                parser.parse_stream_chunk(text[i : i + width])
                for i in range(0, len(text), width)
            ]
            parts.append(parser.parse_stream_end("stop"))
            self.assertEqual("".join(r for r, _ in parts), "Need a tool.")
            self.assertEqual("".join(c for _, c in parts), "<tool_call>f</tool_call>")

    def test_real_closer_preserves_final_on_length(self):
        text = (
            "Example <tool_call>quoted</tool_call></think><tool_call>real</tool_call>"
        )
        for stream, width in itertools.product((False, True), (1, 7, 1000)):
            parser = self.parser(stream)
            parts = [
                parser.parse_stream_chunk(text[i : i + width])
                for i in range(0, len(text), width)
            ]
            parts.append(parser.parse_stream_end("length"))
            self.assertEqual(
                "".join(r for r, _ in parts), "Example <tool_call>quoted</tool_call>"
            )
            self.assertEqual(
                "".join(c for _, c in parts), "<tool_call>real</tool_call>"
            )

    def test_other_detector_keeps_existing_force_content(self):
        parser = load_source_methods()["ReasoningParser"]("qwen3", force_reasoning=True)
        parser.detector._force_nonempty_content = True
        self.assertEqual(parser.parse_non_stream("thought", "length"), ("", "thought"))


class TokenTerminationTests(unittest.TestCase):
    setUpClass = token_context.TokenContextTests.__dict__["setUpClass"]
    setUp = token_context.TokenContextTests.setUp

    def test_force_content_cannot_promote_unclosed_real_token_thought(self):
        text = "Example <tool_call>quoted</tool_call>"
        for finish, stream in itertools.product(
            ("stop", "length", "abort"), (False, True)
        ):
            parser = self.parser(
                "nemotron_3",
                force_reasoning=True,
                tokenizer=LiteralTokenizer(),
                stream_reasoning=stream,
            )
            parser.detector._force_nonempty_content = True
            self.assertEqual(
                parser.parse_non_stream(text, finish, output_ids=ordinary(text)),
                (text, ""),
            )
            parser = self.parser(
                "nemotron_3",
                force_reasoning=True,
                tokenizer=LiteralTokenizer(),
                stream_reasoning=stream,
            )
            parser.detector._force_nonempty_content = True
            parts = [parser.parse_stream_chunk(text, output_ids=ordinary(text))]
            parts.append(parser.parse_stream_end(finish))
            self.assertEqual("".join(r for r, _ in parts), text)
            self.assertEqual("".join(c for _, c in parts), "")


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.namespace = {}
        load_functions(
            SRT / "entrypoints/openai/serving_chat.py",
            [
                "apply_nemotron_structured_output_reasoning_budget",
                "nemotron_response_format_template_kwargs",
            ],
            self.namespace,
        )

    def request(self, **kwargs):
        values = dict(
            response_format=NS(type="json_object"),
            tools=None,
            tool_choice=None,
            input_ids=None,
            continue_final_message=False,
            reasoning_effort=None,
            chat_template_kwargs={},
            custom_params={},
            max_completion_tokens=None,
            max_tokens=8192,
        )
        values.update(kwargs)
        return NS(**values)

    def test_reservation_boundaries_and_explicit_controls(self):
        apply = self.namespace["apply_nemotron_structured_output_reasoning_budget"]
        for total, expected in (
            (1, 0),
            (128, 0),
            (512, 256),
            (8192, 4096),
            (16384, 12288),
        ):
            req = self.request(max_tokens=total, custom_params={"other": 3})
            apply(req, "nemotron_3")
            self.assertEqual(
                req.custom_params, {"other": 3, "thinking_budget": expected}
            )
        for overrides in (
            {"reasoning_effort": "none"},
            {"chat_template_kwargs": {"thinking": False}},
            {"chat_template_kwargs": {"enable_thinking": False}},
            {"input_ids": [1]},
            {"continue_final_message": True},
            {"max_tokens": None},
            {"response_format": None},
            {"custom_params": {"thinking_budget": -1}},
        ):
            req = self.request(**overrides)
            original = dict(req.custom_params)
            apply(req, "nemotron_3")
            self.assertEqual(req.custom_params, original)
        req = self.request()
        apply(req, "qwen3")
        self.assertEqual(req.custom_params, {})

    def test_effective_tools_and_completion_budget_precedence(self):
        apply = self.namespace["apply_nemotron_structured_output_reasoning_budget"]
        req = self.request(
            response_format=None, max_tokens=10, max_completion_tokens=512
        )
        apply(req, "nemotron_3", structured_tools=True)
        self.assertEqual(req.custom_params, {"thinking_budget": 256})
        req = self.request(response_format=None, tools=[1], tool_choice="none")
        apply(req, "nemotron_3")
        self.assertEqual(req.custom_params, {})

    def test_template_receives_exact_response_contract(self):
        payload = {"type": "json_schema", "json_schema": {"name": "answer"}}
        req = self.request(
            response_format=NS(
                type="json_schema", model_dump=Mock(return_value=payload)
            )
        )
        self.assertEqual(
            self.namespace["nemotron_response_format_template_kwargs"](
                req, "nemotron_3"
            ),
            {"response_format": payload},
        )

    def test_budget_enables_filter_or_aborts(self):
        class Grammar:
            pass

        namespace = dict(
            ReasonerGrammarObject=Grammar,
            get_request_reasoning_end_token_ids=lambda *a, **kw: None,
        )
        load_functions(
            SRT / "constrained/grammar_manager.py",
            [
                "GrammarManager._get_request_thinking_bounds",
                "GrammarManager._apply_request_reasoning_config",
            ],
            namespace,
        )
        owner = NS(scheduler=NS(model_config=NS()))
        owner._get_request_thinking_bounds = lambda req: namespace[
            "_get_request_thinking_bounds"
        ](owner, req)
        for budget, supports_filter in itertools.product((-1, 0, 100), (False, True)):
            grammar = Grammar()
            grammar.token_filter_fn = Mock() if supports_filter else None
            grammar.enable_token_filter = False
            req = NS(
                grammar=grammar,
                sampling_params=NS(custom_params={"thinking_budget": budget}),
                set_finish_with_abort=Mock(),
            )
            namespace["_apply_request_reasoning_config"](owner, req)
            self.assertEqual(grammar.max_think_tokens, budget)
            self.assertEqual(
                req.set_finish_with_abort.called, budget >= 0 and not supports_filter
            )
            self.assertEqual(
                grammar.enable_token_filter, budget >= 0 and supports_filter
            )


if __name__ == "__main__":
    unittest.main()
