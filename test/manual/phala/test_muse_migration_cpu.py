"""Actual Muse/source methods without torch/native serving imports.

Protocol envelopes and environment are test doubles. Schema generation, parser
selection, Muse parsing and JSON Schema validation are real source/library code.
This is not native grammar compilation, a tokenizer test, or SSE acceptance.
"""

import ast
import copy
import hashlib
import inspect
import json
import logging
import os
import re
import unittest
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"
FUTURE = ast.ImportFrom(
    module="__future__", names=[ast.alias(name="annotations")], level=0
)


@dataclass
class ToolCallItem:
    tool_index: int
    parameters: str
    name: str | None = None


@dataclass
class StreamingParseResult:
    normal_text: str = ""
    calls: list = field(default_factory=list)


class ToolChoice(SimpleNamespace):
    pass


def source_nodes(path, names=None):
    nodes = ast.parse((SRT / path).read_text(encoding="utf-8")).body
    return [
        n
        for n in nodes
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.Assign, ast.AnnAssign))
        and (names is None or getattr(n, "name", None) in names)
    ]


def execute(nodes, ns):
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[FUTURE, *nodes], type_ignores=[])
            ),
            "<actual Muse source>",
            "exec",
        ),
        ns,
    )


def load():
    ns = dict(
        json=json,
        logging=logging,
        re=re,
        inspect=inspect,
        logger=logging.getLogger(__name__),
        ToolChoice=ToolChoice,
        ToolCallItem=ToolCallItem,
        StreamingParseResult=StreamingParseResult,
        ToolStrictLevel=SimpleNamespace(FUNCTION=1, PARAMETER=2),
        Glm47MoeDetector=type("Glm47MoeDetector", (), {}),
        envs=SimpleNamespace(
            SGLANG_FORWARD_UNKNOWN_TOOLS=SimpleNamespace(get=lambda: False),
            SGLANG_TOOL_STRICT_LEVEL=SimpleNamespace(get=lambda: 0),
        ),
    )
    execute(source_nodes("function_call/muse_glimmer_format.py"), ns)
    base = source_nodes(
        "function_call/base_format_detector.py", {"BaseFormatDetector"}
    )[0]
    base.bases = []
    base.body = [
        n
        for n in base.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {
            "__init__",
            "get_auto_tool_call_structural_tag",
            "get_structural_tag",
            "get_structural_tag_name",
            "supports_structural_tag",
            "parses_required_natively",
            "parses_constrained_output_natively",
        }
    ]
    for n in base.body:
        n.decorator_list = []
    execute([base], ns)
    execute(source_nodes("function_call/muse_glimmer_detector.py"), ns)
    execute(
        source_nodes(
            "function_call/utils.py",
            {"_get_tool_schema", "_get_tool_schema_defs", "get_json_schema_constraint"},
        ),
        ns,
    )
    parser = source_nodes(
        "function_call/function_call_parser.py", {"FunctionCallParser"}
    )[0]
    parser.body = [n for n in parser.body if isinstance(n, ast.FunctionDef)]
    execute([parser], ns)
    ns["FunctionCallParser"].ToolCallParserEnum = {
        "muse": ns["MuseGlimmerDetector"],
        "control": ns["BaseFormatDetector"],
    }
    return ns


NS = load()
Parser = NS["FunctionCallParser"]


def tool(name):
    return SimpleNamespace(
        function=SimpleNamespace(
            name=name,
            description="Weather lookup",
            strict=False,
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    )


class MuseConstraintTests(unittest.TestCase):
    def setUp(self):
        self.tools = [tool("weather"), tool("other")]

    def test_required_has_cardinality_and_schema(self):
        parser = Parser(self.tools, "muse", constrained_output=True)
        self.assertFalse(parser.detector.parses_required_natively())
        kind, schema = parser.get_structure_constraint(
            "required", parallel_tool_calls=False
        )
        self.assertEqual(kind, "json_schema")
        validator = Draft202012Validator(schema)
        self.assertFalse(validator.is_valid([]))
        valid = [{"name": "weather", "parameters": {"city": "Paris"}}]
        self.assertTrue(validator.is_valid(valid))
        self.assertFalse(validator.is_valid(valid * 2))
        self.assertFalse(validator.is_valid([{"name": "unknown", "parameters": {}}]))

    def test_named_choice_constrains_only_requested_tool(self):
        _, schema = Parser(self.tools, "muse").get_structure_constraint(
            ToolChoice(function=SimpleNamespace(name="weather"))
        )
        validator = Draft202012Validator(schema)
        self.assertTrue(
            validator.is_valid([{"name": "weather", "parameters": {"city": "Paris"}}])
        )
        self.assertFalse(
            validator.is_valid([{"name": "other", "parameters": {"city": "Paris"}}])
        )

    def test_named_choice_survives_native_required_capability(self):
        with patch.object(
            NS["MuseGlimmerDetector"], "parses_required_natively", return_value=True
        ):
            self.assertIsNone(
                Parser(self.tools, "muse").get_structure_constraint("required")
            )
            kind, schema = Parser(self.tools, "muse").get_structure_constraint(
                ToolChoice(function=SimpleNamespace(name="weather"))
            )
            self.assertEqual(kind, "json_schema")
            self.assertEqual(schema["items"]["properties"]["name"]["enum"], ["weather"])

    def test_auto_does_not_force_json_or_execute_quoted_json(self):
        parser = Parser(self.tools, "muse")
        self.assertIsNone(parser.get_structure_constraint("auto"))
        text = (
            ' to=user<|message|>[{"name":"weather","parameters":{"city":"X"}}]<|eot|>'
        )
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(calls, [])
        self.assertIn('"weather"', normal)

    def test_json_channel_and_bare_body_all_chunk_splits(self):
        body = '[{"name":"weather","parameters":{"city":"Paris"}}]'
        for prefix in ("", " to=weather<|message|>", " to=user<|message|>"):
            text = prefix + body
            for split in range(len(text) + 1):
                parser = Parser(self.tools, "muse", constrained_output=True)
                results = [
                    parser.parse_stream_chunk(text[:split]),
                    parser.parse_stream_chunk(text[split:]),
                    parser.parse_stream_end(),
                ]
                self.assertEqual("".join(x[0] for x in results), "")
                calls = [c for _, cs in results for c in cs]
                self.assertEqual(
                    [(c.name, json.loads(c.parameters)) for c in calls],
                    [("weather", {"city": "Paris"})],
                )

    def test_self_channel_never_executes_json_even_when_constrained(self):
        parser = Parser(self.tools, "muse", constrained_output=True)
        body = '[{"name":"weather","parameters":{"city":"X"}}]'
        normal, calls = parser.parse_non_stream(
            " to=self<|message|>" + body + "<|eom|>"
        )
        self.assertEqual(calls, [])
        self.assertEqual(normal, body)

    def test_unknown_json_does_not_execute(self):
        parser = Parser(self.tools, "muse", constrained_output=True)
        _, calls = parser.parse_non_stream('[{"name":"missing","parameters":{}}]')
        self.assertEqual(calls, [])

    def test_unrelated_detector_default_contract_is_unchanged(self):
        parser = Parser(self.tools, "control", constrained_output=True)
        self.assertFalse(parser.detector.parses_constrained_output_natively())
        self.assertFalse(hasattr(parser.detector, "_constrained_output"))
        self.assertTrue(parser.detector.supports_structural_tag())

    def test_shared_chat_responses_use_constrained_native_route(self):
        for path in (
            "entrypoints/openai/serving_chat.py",
            "entrypoints/openai/serving_responses.py",
        ):
            text = (SRT / path).read_text(encoding="utf-8")
            self.assertIn("constrained_output=is_required", text)
            self.assertIn("parses_constrained_output_natively()", text)


def load_reasoner():
    ns = dict(logging=logging, logger=logging.getLogger(__name__))
    execute(source_nodes("utils/token_sequence_matcher.py"), ns)
    base = source_nodes("constrained/base_grammar_backend.py", {"BaseGrammarObject"})[0]
    base.body = [
        n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    ]
    execute([base], ns)
    execute(
        source_nodes(
            "constrained/reasoner_grammar_backend.py", {"ReasonerGrammarObject"}
        ),
        ns,
    )
    return ns["ReasonerGrammarObject"]


class MuseGrammarTests(unittest.TestCase):
    def make(self, channel=True, reasoning=True, **kwargs):
        grammar = MagicMock()
        obj = load_reasoner()(
            grammar,
            [7, 8],
            channel_header_end_ids=[12, 13] if channel else None,
            channel_reasoning_header_ids=[10, 11, 12, 13] if channel else None,
            **kwargs,
        )
        obj.maybe_init_reasoning(reasoning)
        return obj, grammar

    def feed(self, obj, tokens):
        for token in tokens:
            obj.accept_token(token)

    def test_channel_header_not_masked_or_accepted_until_complete(self):
        obj, grammar = self.make()
        self.feed(obj, [99, 7, 8])
        for token in [50, 51, 12, 13]:
            obj.fill_vocab_mask(None, 0)
            obj.accept_token(token)
        grammar.fill_vocab_mask.assert_not_called()
        grammar.accept_token.assert_not_called()
        obj.fill_vocab_mask(None, 0)
        obj.accept_token(100)
        grammar.fill_vocab_mask.assert_called_once_with(None, 0)
        grammar.accept_token.assert_called_once_with(100)

    def test_repeated_self_channel_does_not_activate_grammar(self):
        obj, grammar = self.make()
        self.feed(obj, [99, 7, 8, 10, 11, 12, 13])
        self.assertTrue(obj._is_thinking())
        self.feed(obj, [98, 7, 8, 50, 12, 13, 100])
        grammar.accept_token.assert_called_once_with(100)

    def test_self_prefix_tool_name_does_not_disable_grammar(self):
        obj, grammar = self.make()
        self.feed(obj, [99, 7, 8, 10, 11, 55, 12, 13, 100])
        grammar.accept_token.assert_called_once_with(100)

    def test_rollback_each_boundary_restores_exact_state_and_counts(self):
        tokens = [99, 7, 8, 10, 11, 12, 13, 98, 7, 8, 50, 12, 13, 100, 101]
        for keep in range(len(tokens) + 1):
            obj, grammar = self.make()
            reference, _ = self.make()
            self.feed(reference, tokens[:keep])
            self.feed(obj, tokens)
            obj.rollback(len(tokens) - keep)
            self.assertEqual(obj._snapshot_state(), reference._snapshot_state())
            steps = len(tokens) - max(keep, len(tokens) - 2)
            if steps:
                grammar.rollback.assert_called_once_with(steps)
            else:
                grammar.rollback.assert_not_called()

    def test_copy_does_not_share_histories(self):
        obj, grammar = self.make(min_think_tokens=3)
        self.feed(obj, [99, 7, 8, 50, 12])
        clone = obj.copy()
        self.assertEqual(clone._snapshot_state(), obj._snapshot_state())
        self.assertEqual(clone.min_think_tokens, 3)
        clone.accept_token(13)
        self.assertTrue(clone._is_generation())
        self.assertTrue(obj._waiting_for_channel_header)
        clone.rollback(1)
        self.assertEqual(clone._snapshot_state(), obj._snapshot_state())
        self.assertIsNot(clone._state_history, obj._state_history)
        grammar.copy.assert_called_once()

    def test_rejected_inner_token_does_not_advance_wrapper(self):
        obj, grammar = self.make()
        self.feed(obj, [99, 7, 8, 50, 12, 13])
        before = (obj._snapshot_state(), list(obj._state_history), obj.current_token)
        grammar.accept_token.side_effect = ValueError("native rejection")
        with self.assertRaisesRegex(ValueError, "native rejection"):
            obj.accept_token(100)
        self.assertEqual(
            (obj._snapshot_state(), obj._state_history, obj.current_token), before
        )
        grammar.accept_token.side_effect = None
        obj.accept_token(101)
        obj.rollback(1)
        self.assertEqual(obj._snapshot_state(), before[0])
        grammar.rollback.assert_called_once_with(1)

    def test_long_final_output_keeps_only_header_snapshots(self):
        obj, _ = self.make()
        self.feed(obj, [99, 7, 8, 50, 12, 13])
        prefix = len(obj._state_history)
        # No inner grammar double: this measures the actual wrapper bookkeeping.
        obj.grammar = None
        for _ in range(100_000):
            obj.accept_token(100)
        self.assertEqual(obj.tokens_after_end, 100_000)
        self.assertEqual(len(obj._state_history), prefix)
        for _ in range(100):
            clone = obj.copy()
            self.assertEqual(len(clone._state_history), prefix)
            clone.rollback(200)
            self.assertEqual(clone.tokens_after_end, 99_800)
        obj.rollback(100_001)
        self.assertTrue(obj._waiting_for_channel_header)
        self.assertEqual(len(obj._state_history), prefix - 1)

    def test_bounded_header_timeout_and_unlimited_setting(self):
        obj, grammar = self.make(max_channel_header_tokens=2)
        self.feed(obj, [7, 8, 50, 51, 100])
        grammar.accept_token.assert_called_once_with(100)
        unlimited, grammar = self.make(max_channel_header_tokens=-1)
        self.feed(unlimited, [7, 8] + [50] * 32)
        self.assertTrue(unlimited._waiting_for_channel_header)
        grammar.accept_token.assert_not_called()

    def test_non_muse_direct_transition_and_rollback_unchanged(self):
        obj, grammar = self.make(channel=False)
        self.feed(obj, [99, 7, 8, 100, 101])
        self.assertEqual(
            [c.args[0] for c in grammar.accept_token.call_args_list], [100, 101]
        )
        obj.rollback(3)
        grammar.rollback.assert_called_once_with(2)
        self.assertTrue(obj._is_thinking())
        self.assertEqual(obj._matched_think_end_tokens, 1)
        self.assertEqual(obj.tokens_in_think, 1)

    def test_direct_final_masks_first_token(self):
        obj, grammar = self.make(reasoning=False)
        obj.fill_vocab_mask(None, 0)
        obj.accept_token(100)
        grammar.fill_vocab_mask.assert_called_once_with(None, 0)
        grammar.accept_token.assert_called_once_with(100)

    def test_request_specific_terminator_stays_predecode_only(self):
        obj, _ = self.make(channel=False)
        obj.set_request_think_end_ids([20, 21])
        self.feed(obj, [20, 21])
        self.assertTrue(obj._is_generation())
        with self.assertRaises(RuntimeError):
            obj.set_request_think_end_ids([22])

    def test_thinking_budget_filter_preserved_for_non_muse(self):
        filters = MagicMock()
        obj, _ = self.make(
            channel=False,
            min_think_tokens=2,
            max_think_tokens=3,
            enable_token_filter=True,
            token_filter_fn=filters,
        )
        obj.fill_vocab_mask(None, 0)
        self.assertIs(filters.call_args.args[3], False)
        self.feed(obj, [98, 99, 100])
        obj.fill_vocab_mask(None, 0)
        self.assertEqual(filters.call_args.args[1], [7])
        self.assertIs(filters.call_args.args[3], True)


def load_chat():
    ns = dict(
        NS,
        copy=copy,
        Enum=Enum,
        MessageProcessingResult=SimpleNamespace,
        process_content_for_template_format=lambda message, *args: message,
        normalize_hunyuan_reasoning_effort=lambda *args: None,
        _CHAT_TEMPLATE_CLIENT_ERRORS=(ValueError,),
    )
    ns["envs"] = SimpleNamespace(
        SGLANG_DEFAULT_THINKING=SimpleNamespace(get=lambda: False)
    )
    functions = {
        "normalize_muse_history",
        "normalize_muse_reasoning",
        "apply_muse_structured_output_reasoning_default",
        "muse_format_template_kwargs",
        "normalize_tool_content",
        "normalize_assistant_tool_call_arguments",
        "ThinkingMode",
    }
    execute(source_nodes("entrypoints/openai/serving_chat.py", functions), ns)
    ns["_COMPLETE_THINK_CONTENT_RE"] = re.compile(
        r"\A\s*<think>(?P<reasoning>.*?)</think>\s*\Z", re.DOTALL
    )
    cls = source_nodes("entrypoints/openai/serving_chat.py", {"OpenAIServingChat"})[0]
    cls.bases = []
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {
            "_process_messages",
            "_apply_jinja_template",
            "_get_reasoning_from_request",
            "_render_and_encode_chat_template",
        }
    ]
    execute([cls], ns)
    # Preserve the actual request normalizer and its existing exclusion precedence.
    request = source_nodes("entrypoints/openai/protocol.py", {"ChatCompletionRequest"})[
        0
    ]
    normalizer = next(
        n
        for n in request.body
        if isinstance(n, ast.FunctionDef) and n.name == "normalize_reasoning_inputs"
    )
    normalizer.decorator_list = []
    execute([normalizer], ns)
    return ns


CHAT = load_chat()


def request(**kwargs):
    values = dict(
        response_format=None,
        reasoning_effort=None,
        include_reasoning=None,
        chat_template_kwargs=None,
        stop=[],
        input_ids=None,
        tools=[],
        parallel_tool_calls=True,
        continue_final_message=False,
        messages=[
            SimpleNamespace(
                model_dump=lambda: {
                    "role": "user",
                    "content": 'Literal 中文 "quote" \\ backslash',
                }
            )
        ],
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


class MuseHistoryTests(unittest.TestCase):
    def test_history_control_promotion_alias_and_no_input_mutation(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "developer", "content": "rule"},
            {"role": "assistant", "content": None, "reasoning": "legacy"},
            {"role": "system", "content": "second rule"},
            {"role": "assistant", "content": "<think>private</think>"},
        ]
        original = copy.deepcopy(messages)
        normalized = CHAT["normalize_muse_history"](messages)
        self.assertEqual(messages, original)
        self.assertEqual(
            normalized[0], {"role": "system", "content": "rule\n\nsecond rule"}
        )
        self.assertEqual(normalized[2]["reasoning_content"], "legacy")
        self.assertNotIn("reasoning", normalized[2])
        self.assertEqual(normalized[3]["content"], "")
        self.assertEqual(normalized[3]["reasoning_content"], "private")

    def test_empty_canonical_reasoning_wins_and_embedded_think_is_literal(self):
        messages = [
            {
                "role": "assistant",
                "content": "A <think>x</think> example.",
                "reasoning_content": "",
                "reasoning": "legacy",
            }
        ]
        normalized = CHAT["normalize_muse_history"](messages)
        self.assertEqual(normalized[0]["content"], messages[0]["content"])
        self.assertEqual(normalized[0]["reasoning_content"], "")

    def test_protocol_alias_real_pydantic_validation_and_serialization(self):
        from typing import List, Literal, Optional, Tuple, Union, get_args

        from pydantic import (
            BaseModel,
            Field,
            ValidationError,
            field_validator,
            model_validator,
        )

        ns = dict(
            BaseModel=BaseModel,
            Field=Field,
            field_validator=field_validator,
            model_validator=model_validator,
            Optional=Optional,
            Union=Union,
            List=List,
            Literal=Literal,
            Tuple=Tuple,
            get_args=get_args,
            ToolCall=dict,
            Tool=dict,
            ChatCompletionMessageContentPart=dict,
            ChatCompletionMessageContentThinkingPart=type("ThinkingPart", (), {}),
        )
        # Keep role validators and actual Pydantic field definitions unchanged.
        ns["_GenericMessageRole"] = Literal[
            "system", "assistant", "tool", "function", "developer", "latest_reminder"
        ]
        ns["_GENERIC_MESSAGE_ROLES"] = get_args(ns["_GenericMessageRole"])
        execute(
            source_nodes(
                "entrypoints/openai/protocol.py", {"ChatCompletionMessageGenericParam"}
            ),
            ns,
        )
        model = ns["ChatCompletionMessageGenericParam"]
        model.model_rebuild(_types_namespace=ns)
        msg = model(role="assistant", reasoning="legacy", content=None)
        self.assertEqual(msg.reasoning_content, "legacy")
        self.assertNotIn("reasoning", msg.model_dump())
        self.assertEqual(
            model(
                role="assistant", reasoning="legacy", reasoning_content=""
            ).reasoning_content,
            "",
        )
        with self.assertRaises(ValidationError):
            model(role="developer", reasoning="private")

    def test_strength_and_default_precedence(self):
        for effort in ("none", "low", "medium", "high", "xhigh", "max"):
            req = request(reasoning_effort=effort)
            CHAT["normalize_muse_reasoning"](req, "muse")
            self.assertEqual(req.chat_template_kwargs["reasoning_strength"], effort)
        for val in (False, "false", "off"):
            req = request(chat_template_kwargs={"enable_thinking": val})
            CHAT["normalize_muse_reasoning"](req, "muse")
            self.assertEqual(req.chat_template_kwargs["reasoning_strength"], "none")
        req = request(
            reasoning_effort="low", chat_template_kwargs={"reasoning_strength": "high"}
        )
        CHAT["normalize_muse_reasoning"](req, "muse")
        self.assertEqual(req.chat_template_kwargs["reasoning_strength"], "high")
        req = request(response_format=SimpleNamespace(type="json_object"))
        CHAT["apply_muse_structured_output_reasoning_default"](req, "muse")
        self.assertEqual(req.chat_template_kwargs["reasoning_strength"], "none")
        for control in (
            {"reasoning_effort": "high"},
            {"include_reasoning": True},
            {"reasoning_max_tokens": 100},
            {"chat_template_kwargs": {"thinking": True}},
        ):
            req = request(
                response_format=SimpleNamespace(type="json_schema"), **control
            )
            CHAT["apply_muse_structured_output_reasoning_default"](req, "muse")
            self.assertNotEqual(
                (req.chat_template_kwargs or {}).get("reasoning_strength"), "none"
            )

    def test_non_muse_helpers_are_noops(self):
        for parser in (None, "qwen3", "nemotron_3", "kimi_k3", "deepseek-v4"):
            req = request(
                reasoning_effort="none",
                response_format=SimpleNamespace(type="json_object"),
            )
            CHAT["normalize_muse_reasoning"](req, parser)
            CHAT["apply_muse_structured_output_reasoning_default"](req, parser)
            self.assertIsNone(req.chat_template_kwargs)
            self.assertEqual(
                CHAT["muse_format_template_kwargs"](req, parser, ("json_schema", {})),
                {},
            )

    def test_protocol_aliases_preserve_explicit_and_visibility_precedence(self):
        normalize = CHAT["normalize_reasoning_inputs"]
        for raw in ({"enable_thinking": False}, {"thinking": {"type": "disabled"}}):
            result = normalize(None, raw)
            self.assertFalse(result["chat_template_kwargs"]["thinking"])
        result = normalize(
            None,
            {"reasoning": {"enabled": True, "exclude": True}, "enable_thinking": False},
        )
        self.assertTrue(result["chat_template_kwargs"]["thinking"])
        self.assertTrue(result["reasoning_exclude"])
        self.assertNotIn("reasoning_strength", result["chat_template_kwargs"])
        result = normalize(
            None, {"reasoning": {"exclude": False}, "include_reasoning": False}
        )
        self.assertFalse(result["reasoning_exclude"])
        self.assertNotIn("chat_template_kwargs", result)


@unittest.skipUnless(
    os.environ.get("MUSE_TEMPLATE"), "requires pinned historical Muse template"
)
class MuseTemplateCallChainTests(unittest.TestCase):
    def make_serving(self):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        data = Path(os.environ["MUSE_TEMPLATE"]).read_bytes()
        self.assertEqual(
            hashlib.sha256(data).hexdigest(),
            "900db3effc316e33295ec3d7dfa2df83ea2735228cba73adba8fecc2e83343f7",
        )
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
        template = env.from_string(data.decode())

        def render(messages, **kwargs):
            self.kwargs = kwargs
            self.rendered = template.render(
                messages=messages, bos_token="<s>", **kwargs
            )
            return self.rendered

        serving = CHAT["OpenAIServingChat"]()
        serving.reasoning_parser = serving.tool_call_parser = "muse"
        serving.is_gpt_oss = serving.is_gemma4 = False
        serving.chat_encoding_spec = None
        serving.default_chat_template_kwargs = {}
        serving._tokenizer_auto_adds_specials = False
        serving.template_manager = SimpleNamespace(
            reasoning_config=None,
            chat_template_name=None,
            jinja_template_content_format="string",
            jinja_template_may_reorder_tool_results=False,
        )
        serving.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(
                apply_chat_template=render,
                encode=lambda text, **kw: list(text.encode()),
            )
        )
        serving._effective_tools = lambda req: req.tools
        serving._effective_tool_choice = lambda req: getattr(req, "tool_choice", "auto")
        serving._request_tools_for_prompt = lambda req: [
            {"type": "function", "function": vars(t.function)} for t in req.tools
        ]
        serving._patch_reasoning_skip_special_tokens = lambda req: None
        serving._fold_qwen35_system_messages = lambda messages: messages
        serving._apply_qwen35_reasoning_effort_guidance = lambda messages, effort: (
            messages
        )
        serving._expose_qwen35_reasoning_tool_history = lambda messages: None
        serving._filter_message_tools_for_prompt = lambda messages, req: None
        serving._uses_qwen35_chat_template = lambda: False
        serving._encode_messages = lambda *args, **kwargs: None
        serving._handle_last_assistant_message = lambda messages, req: (messages, "")
        return serving

    def test_real_process_jinja_render_chain_carries_required_schema(self):
        serving = self.make_serving()
        req = request(
            tools=[tool("weather")],
            tool_choice="required",
            reasoning_effort="none",
            skip_special_tokens=True,
        )
        result = serving._process_messages(req, False)
        self.assertEqual(
            self.kwargs["_phala_muse_tool_schema"], result.tool_call_constraint[1]
        )
        self.assertIn("one JSON array", self.rendered)
        self.assertNotIn('<atem:invoke name="$FUNCTION_NAME">', self.rendered)
        self.assertTrue(self.rendered.endswith("<|start|>assistant to=user<|message|>"))
        self.assertFalse(result.require_reasoning)
        self.assertIn('Literal 中文 "quote" \\ backslash', self.rendered)

    def test_real_chain_json_default_and_explicit_reasoning(self):
        for effort, expected in ((None, False), ("high", True)):
            serving = self.make_serving()
            fmt = SimpleNamespace(
                type="json_object", model_dump=lambda **kw: {"type": "json_object"}
            )
            result = serving._process_messages(
                request(
                    response_format=fmt,
                    reasoning_effort=effort,
                    skip_special_tokens=True,
                ),
                False,
            )
            self.assertIn("one valid JSON object", self.rendered)
            self.assertEqual(result.require_reasoning, expected)
            self.assertEqual(
                self.kwargs["_phala_muse_response_format"], {"type": "json_object"}
            )

    def test_plain_auto_keeps_atem_instructions_and_default_reasoning(self):
        serving = self.make_serving()
        result = serving._process_messages(
            request(
                tools=[tool("weather")], tool_choice="auto", skip_special_tokens=True
            ),
            False,
        )
        self.assertNotIn("_phala_muse_tool_schema", self.kwargs)
        self.assertIn('<atem:invoke name="$FUNCTION_NAME">', self.rendered)
        self.assertTrue(result.require_reasoning)


if __name__ == "__main__":
    unittest.main()
