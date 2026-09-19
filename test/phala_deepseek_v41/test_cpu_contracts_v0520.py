"""Finite CPU protocol/usage checks without importing the GPU runtime."""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / "python/sglang/srt/phala_compat" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


roles = load("dsv41_message_roles")
protocol = load("dsv41_protocol_compat")
tools = load("dsv41_tool_choice_none")


class ProtocolContracts(unittest.TestCase):
    def test_developer_preserves_caller_and_standalone_generation_role(self):
        solo = [{"role": "developer", "content": "one"}]
        self.assertIs(roles._prepare_messages(solo), solo)
        original = solo + [{"role": "user", "content": "two"}]
        actual = roles._prepare_messages(original)
        self.assertEqual(actual[0]["role"], "system")
        self.assertEqual(original[0]["role"], "developer")

    def test_system_only_user_turn_is_opt_in(self):
        original = [{"role": "system", "content": "one"}]
        with patch.dict(os.environ, {"DSV41_SYSTEM_ONLY_USER_TURN": "0"}):
            self.assertIs(roles._prepare_messages(original), original)
        with patch.dict(os.environ, {"DSV41_SYSTEM_ONLY_USER_TURN": "1"}):
            self.assertEqual(
                roles._prepare_messages(original)[-1], {"role": "user", "content": ""}
            )
        self.assertEqual(len(original), 1)

    def test_reasoning_off_does_not_override_explicit_flat_effort(self):
        self.assertEqual(
            protocol._normalize({"reasoning": {"enabled": False}})["reasoning_effort"],
            "none",
        )
        self.assertEqual(
            protocol._normalize(
                {"reasoning_effort": "high", "reasoning": {"enabled": False}}
            )["reasoning_effort"],
            "high",
        )

    def test_integer_budget_and_fractional_effort_are_distinct(self):
        out = protocol._normalize({"reasoning_effort": 75})
        self.assertEqual(out["chat_template_kwargs"]["reasoning_effort"], 75)
        self.assertTrue(out["chat_template_kwargs"]["thinking"])
        original = {"reasoning_effort": 0.75}
        self.assertIs(protocol._normalize(original), original)
        for bad in (-1, 101):
            with self.assertRaises(ValueError):
                protocol._normalize({"reasoning_effort": bad})

    def test_exclude_is_output_setting_and_does_not_disable_generation(self):
        out = protocol._normalize({"reasoning": {"effort": "high", "exclude": True}})
        self.assertTrue(out["reasoning_exclude"])
        self.assertNotIn("reasoning_effort", out)
        self.assertEqual(out["reasoning"]["effort"], "high")

    def test_stream_usage_is_continuous_without_mutating_caller(self):
        original = {"stream": True, "stream_options": {"include_usage": False}}
        out = protocol._normalize(original)
        self.assertEqual(
            out["stream_options"],
            {"include_usage": True, "continuous_usage_stats": True},
        )
        self.assertFalse(original["stream_options"]["include_usage"])

    def test_none_removes_both_tool_definition_carriers(self):
        request = SimpleNamespace(
            tool_choice="none",
            tools=[{"name": "a"}],
            messages=[SimpleNamespace(tools=[{"name": "b"}])],
        )
        self.assertTrue(tools.drop_tool_definitions(request))
        self.assertIsNone(request.tools)
        self.assertIsNone(request.messages[0].tools)
        request.tool_choice = "required"
        request.tools = [{"name": "a"}]
        self.assertFalse(tools.drop_tool_definitions(request))
        self.assertEqual(len(request.tools), 1)


class UsageCapContracts(unittest.TestCase):
    def test_actual_req_method_caps_only_the_emitted_finishing_prefix(self):
        path = ROOT / "python/sglang/srt/managers/schedule_batch.py"
        parsed = ast.parse(path.read_text(encoding="utf-8"))
        req = next(
            n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "Req"
        )
        fn = next(
            n
            for n in req.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_cap_reasoning_tokens_at_finished_len"
        )
        module = ast.Module(body=[fn], type_ignores=[])
        namespace = {}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        for finish, counted, expected in (
            (8, 11, 8),
            (8, 5, 5),
            (None, 11, 11),
            (0, 3, 0),
        ):
            with self.subTest(finish=finish, counted=counted):
                value = SimpleNamespace(finished_len=finish, reasoning_tokens=counted)
                namespace[fn.name](value)
                self.assertEqual(value.reasoning_tokens, expected)


if __name__ == "__main__":
    unittest.main()
