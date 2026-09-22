"""Actual Muse/source methods without torch/native serving imports.

Protocol envelopes and environment are test doubles. Schema generation, parser
selection, Muse parsing and JSON Schema validation are real source/library code.
This is not native grammar compilation, a tokenizer test, or SSE acceptance.
"""
import ast
import inspect
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"
FUTURE = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)


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
    return [n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.Assign, ast.AnnAssign))
            and (names is None or getattr(n, "name", None) in names)]


def execute(nodes, ns):
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[FUTURE, *nodes], type_ignores=[])), "<actual Muse source>", "exec"), ns)


def load():
    ns = dict(json=json, logging=logging, re=re, inspect=inspect,
              logger=logging.getLogger(__name__), ToolChoice=ToolChoice,
              ToolCallItem=ToolCallItem, StreamingParseResult=StreamingParseResult,
              ToolStrictLevel=SimpleNamespace(FUNCTION=1, PARAMETER=2),
              Glm47MoeDetector=type("Glm47MoeDetector", (), {}),
              envs=SimpleNamespace(
                  SGLANG_FORWARD_UNKNOWN_TOOLS=SimpleNamespace(get=lambda: False),
                  SGLANG_TOOL_STRICT_LEVEL=SimpleNamespace(get=lambda: 0)))
    execute(source_nodes("function_call/muse_glimmer_format.py"), ns)
    base = source_nodes("function_call/base_format_detector.py", {"BaseFormatDetector"})[0]
    base.bases = []
    base.body = [n for n in base.body if isinstance(n, ast.FunctionDef) and n.name in {
        "__init__", "get_auto_tool_call_structural_tag", "get_structural_tag", "get_structural_tag_name",
        "supports_structural_tag", "parses_required_natively", "parses_constrained_output_natively"}]
    for n in base.body:
        n.decorator_list = []
    execute([base], ns)
    execute(source_nodes("function_call/muse_glimmer_detector.py"), ns)
    execute(source_nodes("function_call/utils.py", {
        "_get_tool_schema", "_get_tool_schema_defs", "get_json_schema_constraint"}), ns)
    parser = source_nodes("function_call/function_call_parser.py", {"FunctionCallParser"})[0]
    parser.body = [n for n in parser.body if isinstance(n, ast.FunctionDef)]
    execute([parser], ns)
    ns["FunctionCallParser"].ToolCallParserEnum = {
        "muse": ns["MuseGlimmerDetector"], "control": ns["BaseFormatDetector"]}
    return ns


NS = load()
Parser = NS["FunctionCallParser"]


def tool(name):
    return SimpleNamespace(function=SimpleNamespace(
        name=name, strict=False, parameters={
            "type": "object", "properties": {"city": {"type": "string"}},
            "required": ["city"]}))


class MuseConstraintTests(unittest.TestCase):
    def setUp(self):
        self.tools = [tool("weather"), tool("other")]

    def test_required_has_cardinality_and_schema(self):
        parser = Parser(self.tools, "muse", constrained_output=True)
        self.assertFalse(parser.detector.parses_required_natively())
        kind, schema = parser.get_structure_constraint("required", parallel_tool_calls=False)
        self.assertEqual(kind, "json_schema")
        validator = Draft202012Validator(schema)
        self.assertFalse(validator.is_valid([]))
        valid = [{"name": "weather", "parameters": {"city": "Paris"}}]
        self.assertTrue(validator.is_valid(valid))
        self.assertFalse(validator.is_valid(valid * 2))
        self.assertFalse(validator.is_valid([{"name": "unknown", "parameters": {}}]))

    def test_named_choice_constrains_only_requested_tool(self):
        _, schema = Parser(self.tools, "muse").get_structure_constraint(
            ToolChoice(function=SimpleNamespace(name="weather")))
        validator = Draft202012Validator(schema)
        self.assertTrue(validator.is_valid([{"name": "weather", "parameters": {"city": "Paris"}}]))
        self.assertFalse(validator.is_valid([{"name": "other", "parameters": {"city": "Paris"}}]))

    def test_auto_does_not_force_json_or_execute_quoted_json(self):
        parser = Parser(self.tools, "muse")
        self.assertIsNone(parser.get_structure_constraint("auto"))
        text = ' to=user<|message|>[{"name":"weather","parameters":{"city":"X"}}]<|eot|>'
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(calls, [])
        self.assertIn('"weather"', normal)

    def test_json_channel_and_bare_body_all_chunk_splits(self):
        body = '[{"name":"weather","parameters":{"city":"Paris"}}]'
        for prefix in ("", " to=weather<|message|>", " to=user<|message|>"):
            text = prefix + body
            for split in range(len(text) + 1):
                parser = Parser(self.tools, "muse", constrained_output=True)
                results = [parser.parse_stream_chunk(text[:split]),
                           parser.parse_stream_chunk(text[split:]),
                           parser.parse_stream_end()]
                self.assertEqual("".join(x[0] for x in results), "")
                calls = [c for _, cs in results for c in cs]
                self.assertEqual([(c.name, json.loads(c.parameters)) for c in calls],
                                 [("weather", {"city": "Paris"})])

    def test_self_channel_never_executes_json_even_when_constrained(self):
        parser = Parser(self.tools, "muse", constrained_output=True)
        body = '[{"name":"weather","parameters":{"city":"X"}}]'
        normal, calls = parser.parse_non_stream(" to=self<|message|>" + body + "<|eom|>")
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
        for path in ("entrypoints/openai/serving_chat.py", "entrypoints/openai/serving_responses.py"):
            text = (SRT / path).read_text(encoding="utf-8")
            self.assertIn("constrained_output=is_required", text)
            self.assertIn("parses_constrained_output_natively()", text)


def load_reasoner():
    ns = dict(logging=logging, logger=logging.getLogger(__name__))
    execute(source_nodes("utils/token_sequence_matcher.py"), ns)
    base = source_nodes("constrained/base_grammar_backend.py", {"BaseGrammarObject"})[0]
    base.body = [n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    execute([base], ns)
    execute(source_nodes("constrained/reasoner_grammar_backend.py", {"ReasonerGrammarObject"}), ns)
    return ns["ReasonerGrammarObject"]


class MuseGrammarTests(unittest.TestCase):
    def make(self, channel=True, reasoning=True, **kwargs):
        grammar = MagicMock()
        obj = load_reasoner()(grammar, [7, 8], channel_header_end_ids=[12, 13] if channel else None,
                              channel_reasoning_header_ids=[10, 11] if channel else None, **kwargs)
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
        self.assertEqual([c.args[0] for c in grammar.accept_token.call_args_list], [100, 101])
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
        obj, _ = self.make(channel=False, min_think_tokens=2, max_think_tokens=3,
                           enable_token_filter=True, token_filter_fn=filters)
        obj.fill_vocab_mask(None, 0)
        self.assertIs(filters.call_args.args[3], False)
        self.feed(obj, [98, 99, 100])
        obj.fill_vocab_mask(None, 0)
        self.assertEqual(filters.call_args.args[1], [7])
        self.assertIs(filters.call_args.args[3], True)


if __name__ == "__main__":
    unittest.main()
