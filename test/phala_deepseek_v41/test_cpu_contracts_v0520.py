"""Native DS encoder/async contracts; no historical monkey-patch package."""

import ast
import asyncio
import contextvars
import importlib.util
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative=None):
    path = (
        ROOT / "python" / relative
        if relative is not None
        else ROOT / "python/sglang/srt/phala_compat" / (name + ".py")
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    packages = {}
    for package, package_path in (
        ("sglang", ROOT / "python/sglang"),
        ("sglang.srt", ROOT / "python/sglang/srt"),
        ("sglang.srt.phala_compat", ROOT / "python/sglang/srt/phala_compat"),
    ):
        namespace = types.ModuleType(package)
        namespace.__path__ = [str(package_path)]
        packages[package] = namespace
    with patch.dict(sys.modules, {name: module, **packages}):
        spec.loader.exec_module(module)
    return module


encoding = load("encoding_dsv41", "sglang/srt/entrypoints/openai/encoding_dsv41.py")
batcher_module = load(
    "async_batcher", "sglang/srt/managers/async_dynamic_batch_tokenizer.py"
)
roles = load("dsv41_message_roles")
protocol = load("dsv41_protocol_compat")
tools = load("dsv41_tool_choice_none")


class ProtocolContracts(unittest.TestCase):
    def test_current_effort_parser_boundaries(self):
        path = ROOT / "python/sglang/srt/entrypoints/openai/chat_encoding.py"
        parsed = ast.parse(path.read_text(encoding="utf-8"))
        fn = next(
            n
            for n in parsed.body
            if getattr(n, "name", None) == "parse_dsv41_reasoning_effort"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias("annotations")], level=0
                ),
                fn,
            ],
            type_ignores=[],
        )
        ns = {"encoding_dsv41": encoding}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
        parse = ns[fn.name]
        self.assertEqual([parse(x) for x in (1, 50, 100)], [1, 50, 100])
        self.assertEqual([parse(x) for x in (0.0, 0.5, 0.99)], [1, 50, 99])
        self.assertEqual([parse(x) for x in (True, 0, 101, -0.1, 1.0)], [None] * 5)

    def test_developer_preserves_caller_and_standalone_generation_role(self):
        solo = [{"role": "developer", "content": "one"}]
        self.assertIs(encoding._prepare_messages(solo), solo)
        original = solo + [{"role": "user", "content": "two"}]
        actual = encoding._prepare_messages(original)
        self.assertEqual(actual[0]["role"], "system")
        self.assertEqual(original[0]["role"], "developer")

    def test_standalone_developer_retains_generation_header(self):
        prompt = encoding.encode_messages(
            [{"role": "developer", "content": "one"}], thinking_mode="thinking"
        )
        self.assertIn("<｜Assistant｜><think>", prompt)

    def test_system_only_user_turn_is_opt_in(self):
        original = [{"role": "system", "content": "one"}]
        with patch.dict(os.environ, {"DSV41_SYSTEM_ONLY_USER_TURN": "0"}):
            self.assertIs(encoding._prepare_messages(original), original)
        with patch.dict(os.environ, {"DSV41_SYSTEM_ONLY_USER_TURN": "1"}):
            self.assertEqual(
                encoding._prepare_messages(original)[-1],
                {"role": "user", "content": ""},
            )
            self.assertIn(
                "<｜Assistant｜>",
                encoding.encode_messages(original, thinking_mode="thinking"),
            )
        self.assertEqual(len(original), 1)

    def test_historical_effort_mapping_is_native(self):
        expected = dict(minimal=25, low=50, medium=62, high=75, xhigh=90, max=100)
        self.assertEqual(encoding.REASONING_EFFORT_MAPPINGS, expected)
        for effort, budget in expected.items():
            with self.subTest(effort=effort):
                self.assertIn(
                    str(budget), encoding.render_reasoning_effort(0, "thinking", effort)
                )
        self.assertEqual(encoding.render_reasoning_effort(1, "thinking", 75), "")
        self.assertEqual(encoding.render_reasoning_effort(0, "chat", 75), "")

    def test_invalid_effort_rejected(self):
        for effort in (-1, 0, 101, 0.75, True):
            with self.subTest(effort=effort), self.assertRaises(ValueError):
                encoding.render_reasoning_effort(0, "thinking", effort)


class AsyncContracts(unittest.IsolatedAsyncioTestCase):
    async def test_worker_preserves_context_and_is_serial(self):
        batcher = batcher_module.AsyncDynamicbatchTokenizer(None)
        request_id = contextvars.ContextVar("ds_cpu_request_id")
        request_id.set("request")
        main_thread = threading.get_ident()
        try:
            results = await asyncio.gather(
                *[
                    batcher.run_sync(
                        lambda i=i: (i, request_id.get(), threading.get_ident())
                    )
                    for i in range(12)
                ]
            )
            self.assertEqual([r[0] for r in results], list(range(12)))
            self.assertEqual({r[1] for r in results}, {"request"})
            self.assertEqual(len({r[2] for r in results}), 1)
            self.assertNotEqual(results[0][2], main_thread)
        finally:
            batcher._executor.shutdown()

    async def test_worker_exception_does_not_poison_next_operation(self):
        batcher = batcher_module.AsyncDynamicbatchTokenizer(None)
        try:

            def fail():
                raise ValueError("fixture")

            with self.assertRaisesRegex(ValueError, "fixture"):
                await batcher.run_sync(fail)
            self.assertEqual(await batcher.run_sync(lambda: 42), 42)
        finally:
            batcher._executor.shutdown()


class CompatProtocolContracts(unittest.TestCase):
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
        original_budget = {"reasoning_effort": 75}
        self.assertIs(protocol._normalize(original_budget), original_budget)
        nested_budget = protocol._normalize({"reasoning": {"effort": 75}})
        self.assertEqual(nested_budget["reasoning_effort"], 75)
        self.assertNotIn("effort", nested_budget["reasoning"])
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
