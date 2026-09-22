"""CPU source-method regressions; no mocked parser implementation or GPU imports."""

import ast
import importlib.util
import inspect
import itertools
import json
import os
import re
import sys
import typing
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_source_methods():
    source = SRT / "parser/reasoning_parser.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    wanted = {
        "StreamingParseResult",
        "BaseReasoningFormatDetector",
        "Nemotron3Detector",
        "Qwen3Detector",
        "ReasoningParser",
    }
    classes = [
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name in wanted
    ]
    for node in classes:
        if node.name == "ReasoningParser":
            # Only the registry is narrowed; all executable methods are actual source.
            node.body = [
                child for child in node.body if not isinstance(child, ast.AnnAssign)
            ]
            node.body.insert(
                0,
                ast.parse(
                    'DetectorMap = {"nemotron_3": Nemotron3Detector, "qwen3": Qwen3Detector}'
                ).body[0],
            )
    namespace = dict(
        vars(typing),
        inspect=inspect,
        re=re,
        ChatCompletionRequest=type("ChatCompletionRequest", (), {}),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    serving = ast.parse(
        (SRT / "entrypoints/openai/serving_chat.py").read_text(encoding="utf-8")
    )
    owner = next(
        n
        for n in serving.body
        if isinstance(n, ast.ClassDef) and n.name == "OpenAIServingChat"
    )
    methods = [
        n
        for n in owner.body
        if isinstance(n, ast.FunctionDef) and n.name == "_process_reasoning_stream"
    ]
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


class LiteralTokenizer:
    chat_template = "{# phala-nemotron-literal-message-tokens-v1 #}"

    @staticmethod
    def convert_tokens_to_ids(text):
        return {"<think>": 12, "</think>": 13, "<tool_call>": 14, "</tool_call>": 15}[
            text
        ]

    @staticmethod
    def decode(ids, **kwargs):
        controls = {
            12: "<think>",
            13: "</think>",
            14: "<tool_call>",
            15: "</tool_call>",
        }
        return "".join(controls.get(i, chr(i - 1000) if i >= 1000 else "") for i in ids)


def ordinary(text):
    return [1000 + ord(c) for c in text]


THOUGHT = (
    'Example "<think>x</think><tool_call>not executable</tool_call>". Keep thinking.'
)
ANSWER = 'Answer "<think>literal</think><tool_call>literal</tool_call>".'


class TokenContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_source_methods()
        cls.parser = cls.namespace["ReasoningParser"]
        cls.tokenizer = LiteralTokenizer()
        cls.boundary = module_from_file(
            "sglang.srt.parser.nemotron_token_boundary",
            SRT / "parser/nemotron_token_boundary.py",
        )

    def setUp(self):
        self.modules = patch.dict(
            sys.modules, {"sglang.srt.parser.nemotron_token_boundary": self.boundary}
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_nonstream_real_control_vs_literal_text_and_truncation(self):
        for opening, complete in itertools.product((False, True), repeat=2):
            with self.subTest(opening=opening, complete=complete):
                ids = ([12] if opening else []) + ordinary(THOUGHT)
                if complete:
                    ids += [13] + ordinary(ANSWER)
                parser = self.parser(
                    "nemotron_3", force_reasoning=True, tokenizer=self.tokenizer
                )
                self.assertEqual(
                    parser.parse_non_stream(self.tokenizer.decode(ids), output_ids=ids),
                    (THOUGHT, ANSWER if complete else ""),
                )

    def test_literal_opening_without_reasoning_stays_content(self):
        text = "<think>literal</think><tool_call>literal</tool_call>"
        parser = self.parser(
            "nemotron_3", force_reasoning=False, tokenizer=self.tokenizer
        )
        self.assertEqual(
            parser.parse_non_stream(text, output_ids=ordinary(text)), ("", text)
        )

    def test_stream_actual_serving_method_all_boundaries(self):
        for (
            width,
            incremental,
            stream_reasoning,
            opening,
            complete,
        ) in itertools.product(
            (1, 3, 7, 29, 10000),
            (False, True),
            (False, True),
            (False, True),
            (False, True),
        ):
            with self.subTest(
                width=width,
                incremental=incremental,
                stream=stream_reasoning,
                opening=opening,
                complete=complete,
            ):
                ids = ([12] if opening else []) + ordinary(THOUGHT)
                if complete:
                    ids += [13] + ordinary(ANSWER)
                serving = SimpleNamespace(
                    reasoning_parser="nemotron_3",
                    template_manager=SimpleNamespace(force_reasoning=False),
                    tokenizer_manager=SimpleNamespace(
                        tokenizer=self.tokenizer,
                        server_args=SimpleNamespace(
                            incremental_streaming_output=incremental
                        ),
                    ),
                    _get_reasoning_from_request=lambda _: True,
                    _tool_call_parsing_active=lambda _: False,
                )
                request = SimpleNamespace(stream_reasoning=stream_reasoning)
                states, results = {}, []
                for start in range(0, len(ids), width):
                    end = min(start + width, len(ids))
                    results.append(
                        self.namespace["_process_reasoning_stream"](
                            serving,
                            0,
                            self.tokenizer.decode(ids[start:end]),
                            states,
                            {
                                "output_ids": ids[start:end]
                                if incremental
                                else ids[:end]
                            },
                            request,
                            "stop" if end == len(ids) else None,
                        )
                    )
                self.assertEqual("".join(r or "" for r, _ in results), THOUGHT)
                self.assertEqual(
                    "".join(c or "" for _, c in results), ANSWER if complete else ""
                )

    def test_non_nemotron_token_ids_do_not_change_text_path(self):
        for text in (THOUGHT, "<think>real</think>answer", "plain", "<tool_call>call"):
            parser = self.parser("qwen3", tokenizer=self.tokenizer)
            original = self.parser("qwen3", tokenizer=self.tokenizer)
            self.assertEqual(
                parser.parse_non_stream(text, output_ids=ordinary(text)),
                original.parse_non_stream(text),
            )
            parser = self.parser("qwen3", tokenizer=self.tokenizer)
            original = self.parser("qwen3", tokenizer=self.tokenizer)
            for chunk in (text[:3], text[3:]):
                self.assertEqual(
                    parser.parse_stream_chunk(
                        chunk, output_ids=ordinary(chunk), incremental_output=True
                    ),
                    original.parse_stream_chunk(chunk),
                )

    def test_literal_start_of_reasoning_is_not_stripped_on_finish(self):
        text = "<think>literal</think><tool_call>literal"
        for stream_reasoning in (False, True):
            parser = self.parser(
                "nemotron_3",
                force_reasoning=True,
                tokenizer=self.tokenizer,
                stream_reasoning=stream_reasoning,
            )
            results = [
                parser.parse_stream_chunk(
                    c, output_ids=ordinary(c), incremental_output=True
                )
                for c in text
            ]
            results.append(parser.parse_stream_end())
            self.assertEqual("".join(r for r, _ in results), text)
            self.assertEqual("".join(c for _, c in results), "")

    def test_cumulative_output_ids_must_not_move_backwards(self):
        parser = self.parser(
            "nemotron_3", force_reasoning=True, tokenizer=self.tokenizer
        )
        parser.parse_stream_chunk("abc", output_ids=ordinary("abc"))
        with self.assertRaisesRegex(ValueError, "backwards"):
            parser.parse_stream_chunk("", output_ids=ordinary("ab"))


class LiteralInputGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = module_from_file(
            "nemotron_literal_tokens",
            SRT / "entrypoints/openai/nemotron_literal_tokens.py",
        )

    def test_opt_out_and_no_literals_leave_ids_and_renderer_untouched(self):
        ids = [12, 13]
        for template, content in (
            ("plain", "<think>data"),
            (LiteralTokenizer.chat_template, "plain"),
        ):
            tokenizer = SimpleNamespace(chat_template=template)
            renderer = unittest.mock.Mock(
                side_effect=AssertionError("render must not run")
            )
            self.assertIs(
                self.helper.encode_nemotron_message_literals(
                    tokenizer,
                    [{"role": "user", "content": content}],
                    content,
                    ids,
                    renderer,
                ),
                ids,
            )

    def test_serving_input_model_and_template_guards_are_wired(self):
        parsed = ast.parse(
            (SRT / "entrypoints/openai/serving_chat.py").read_text(encoding="utf-8")
        )
        guard = next(
            node
            for node in ast.walk(parsed)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "encode_nemotron_message_literals"
                for statement in node.body
                for child in ast.walk(statement)
            )
        )
        for model, continuation, multimodal in itertools.product(
            ("nemotron_3", "qwen3", "muse"), (False, True), (False, True)
        ):
            active = eval(
                compile(ast.Expression(guard.test), "<real-input-guard>", "eval"),
                {
                    "self": SimpleNamespace(reasoning_parser=model),
                    "request": SimpleNamespace(continue_final_message=continuation),
                    "is_multimodal": multimodal,
                },
            )
            self.assertEqual(
                active, model == "nemotron_3" and not continuation and not multimodal
            )

    def test_nonstream_serving_routes_ids_only_to_nemotron(self):
        source = ast.parse(
            (SRT / "entrypoints/openai/serving_chat.py").read_text(encoding="utf-8")
        )
        call = next(
            node
            for node in ast.walk(source)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "parse_non_stream"
            and node.keywords
        )
        for model in ("nemotron_3", "qwen3", "muse"):
            parser = SimpleNamespace(
                parse_non_stream=unittest.mock.Mock(return_value=("", "answer"))
            )
            eval(
                compile(ast.Expression(call), "<real-serving-call>", "eval"),
                {
                    "parser": parser,
                    "self": SimpleNamespace(reasoning_parser=model),
                    "text": "answer",
                    "ret_item": {"output_ids": [12, 13]},
                },
            )
            self.assertEqual(
                parser.parse_non_stream.call_args.kwargs,
                {"output_ids": [12, 13]} if model == "nemotron_3" else {},
            )


class RealTokenizerTests(TokenContextTests):
    """Optional pinned-artifact CPU checks; no network requests in this test."""

    @classmethod
    def setUpClass(cls):
        path = os.environ.get("NEMOTRON_TOKENIZER_JSON")
        if not path:
            raise unittest.SkipTest("NEMOTRON_TOKENIZER_JSON not supplied")
        from tokenizers import Tokenizer

        super().setUpClass()
        backend = Tokenizer.from_file(path)

        class ActualTokenizer:
            chat_template = LiteralTokenizer.chat_template
            backend_tokenizer = backend

            @staticmethod
            def convert_tokens_to_ids(text):
                return backend.token_to_id(text)

            @staticmethod
            def decode(ids, skip_special_tokens=True, **kwargs):
                return backend.decode(ids, skip_special_tokens=skip_special_tokens)

        cls.actual = ActualTokenizer()
        cls.helper = module_from_file(
            "nemotron_literal_tokens",
            SRT / "entrypoints/openai/nemotron_literal_tokens.py",
        )

    def test_actual_markers_survive_default_decode(self):
        for marker in ("<think>", "</think>"):
            token_id = self.actual.convert_tokens_to_ids(marker)
            self.assertEqual(self.actual.decode([token_id]), marker)

    def actual_ids(self):
        from tokenizers import Tokenizer

        marker_ids, _ = self.helper._literal_marker_pieces(self.actual)
        config = json.loads(self.actual.backend_tokenizer.to_str())
        config["added_tokens"] = [
            item
            for item in config["added_tokens"]
            if item["id"] not in marker_ids.values()
        ]
        ordinary_backend = Tokenizer.from_str(json.dumps(config))
        thought = (
            "<think>quoted </think> 中文 \U0001f600 <tool_call>not a call</tool_call>."
        )
        answer = 'Final 中文 \U0001f642 {"literal":"<think></think><tool_call>"}'
        ids = [marker_ids["<think>"]]
        ids += ordinary_backend.encode(thought, add_special_tokens=False).ids
        ids += [marker_ids["</think>"]]
        ids += ordinary_backend.encode(answer, add_special_tokens=False).ids
        return thought, answer, ids

    def test_actual_byte_level_unicode_and_incremental_id_boundaries(self):
        thought, answer, ids = self.actual_ids()
        for width, incremental, stream_reasoning in itertools.product(
            (1, 3, 7, 29, 10000), (False, True), (False, True)
        ):
            with self.subTest(
                width=width, incremental=incremental, stream=stream_reasoning
            ):
                parser = self.parser(
                    "nemotron_3",
                    force_reasoning=True,
                    tokenizer=self.actual,
                    stream_reasoning=stream_reasoning,
                )
                emitted, results = "", []
                for start in range(0, len(ids), width):
                    end = min(start + width, len(ids))
                    decoded = self.actual.decode(ids[:end])
                    if end < len(ids):
                        decoded = decoded.rstrip("\ufffd")
                    self.assertTrue(decoded.startswith(emitted))
                    delta, emitted = decoded[len(emitted) :], decoded
                    results.append(
                        parser.parse_stream_chunk(
                            delta,
                            output_ids=ids[start:end] if incremental else ids[:end],
                            incremental_output=incremental,
                        )
                    )
                results.append(parser.parse_stream_end())
                self.assertEqual("".join(r for r, _ in results), thought)
                self.assertEqual("".join(c for _, c in results), answer)
                parser = self.parser(
                    "nemotron_3", force_reasoning=True, tokenizer=self.actual
                )
                self.assertEqual(
                    parser.parse_non_stream(emitted, output_ids=ids), (thought, answer)
                )

    def test_real_end_token_precedes_its_split_text_chunks(self):
        thought, answer, ids = self.actual_ids()
        text = self.actual.decode(ids)
        for width, incremental in itertools.product((1, 3, 7), (False, True)):
            parser = self.parser(
                "nemotron_3", force_reasoning=True, tokenizer=self.actual
            )
            results = []
            for start in range(0, len(text), width):
                token_ids = ids if start == 0 or not incremental else []
                results.append(
                    parser.parse_stream_chunk(
                        text[start : start + width],
                        output_ids=token_ids,
                        incremental_output=incremental,
                    )
                )
            results.append(parser.parse_stream_end())
            self.assertEqual("".join(r for r, _ in results), thought)
            self.assertEqual("".join(c for _, c in results), answer)

    def test_actual_bpe_prompt_literals_preserve_rendered_bytes(self):
        def render(messages):
            return (
                "".join(
                    "<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n"
                    for m in messages
                )
                + "<|im_start|>assistant\n<think>"
            )

        marker_ids, _ = self.helper._literal_marker_pieces(self.actual)
        for role in ("user", "system", "assistant", "tool"):
            messages = [
                {
                    "role": role,
                    "content": '中文 \U0001f600 "<think>x</think><tool_call>data</tool_call>"',
                }
            ]
            rendered = render(messages)
            ids = self.actual.backend_tokenizer.encode(
                rendered, add_special_tokens=False
            ).ids
            output = self.helper.encode_nemotron_message_literals(
                self.actual, messages, rendered, ids, render
            )
            self.assertEqual(
                self.actual.decode(output, skip_special_tokens=False), rendered
            )
            for marker, token_id in marker_ids.items():
                self.assertEqual(
                    output.count(token_id), ids.count(token_id) - 1, marker
                )
            self.assertEqual(
                output,
                self.helper.encode_nemotron_message_literals(
                    self.actual, messages, rendered, ids, render
                ),
            )


if __name__ == "__main__":
    unittest.main()
