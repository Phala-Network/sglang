"""Run guarded historical Qwen source methods without native serving imports."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "python/sglang/srt/entrypoints/openai/serving_chat.py"
METHODS = {
    "_uses_qwen35_chat_template",
    "_fold_qwen35_system_messages",
    "_apply_qwen35_reasoning_effort_guidance",
    "_qwen35_reasoning_effort_token_range",
    "_expose_qwen35_reasoning_tool_history",
}
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
constants = [
    node for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id.startswith("_QWEN35_")
            for target in node.targets)
]
owner = next(node for node in tree.body
             if isinstance(node, ast.ClassDef) and node.name == "OpenAIServingChat")
methods = [node for node in owner.body
           if isinstance(node, ast.FunctionDef) and node.name in METHODS]
assert {node.name for node in methods} == METHODS
serving_config = SimpleNamespace(enable_strict_thinking=True)
namespace = {
    "copy": copy,
    "get_serving": lambda: serving_config,
}
body = [
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    *constants,
    ast.ClassDef(name="Subject", bases=[], keywords=[], body=methods, decorator_list=[]),
]
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
             str(SOURCE), "exec"), namespace)


def subject(model_type):
    instance = namespace["Subject"]()
    instance.tokenizer_manager = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))
    )
    return instance


class QwenHistoricalSourceSemantics(unittest.TestCase):
    def tearDown(self):
        serving_config.enable_strict_thinking = True

    def test_gemma_and_qwen2_do_not_receive_qwen35_behavior(self):
        messages = [{"role": "user", "content": "question"},
                    {"role": "system", "content": "late instruction"}]
        for model_type in ("gemma4", "gemma4_text", "qwen2"):
            with self.subTest(model_type=model_type):
                target = subject(model_type)
                self.assertIs(target._fold_qwen35_system_messages(messages), messages)
                self.assertIs(target._apply_qwen35_reasoning_effort_guidance(messages, "medium"), messages)
                self.assertIsNone(target._qwen35_reasoning_effort_token_range("medium", 10000))

    def test_qwen_system_folding_preserves_input(self):
        messages = [{"role": "system", "content": "first"},
                    {"role": "user", "content": "question"},
                    {"role": "system", "content": "second"}]
        before = copy.deepcopy(messages)
        actual = subject("qwen3_5")._fold_qwen35_system_messages(messages)
        self.assertEqual(actual[0]["content"], "first\n\nsecond")
        self.assertEqual(actual[1], messages[1])
        self.assertEqual(messages, before)

    def test_explicit_medium_aliases_default_and_request_cap(self):
        target = subject("qwen3_5")
        self.assertEqual(target._qwen35_reasoning_effort_token_range("medium", 10000), (128, 4096))
        self.assertEqual(target._qwen35_reasoning_effort_token_range("minimal", 10000), (32, 64))
        for effort in ("high", "max"):
            self.assertEqual(target._qwen35_reasoning_effort_token_range(effort, 10000), (384, 8192))
        self.assertEqual(target._qwen35_reasoning_effort_token_range(None, 10000), (384, 8192))
        self.assertLessEqual(target._qwen35_reasoning_effort_token_range("medium", 256, 40)[1], 40)

    def test_strict_thinking_opt_out_has_no_forced_budget(self):
        serving_config.enable_strict_thinking = False
        self.assertIsNone(subject("qwen3_5")._qwen35_reasoning_effort_token_range("medium", 10000))

    def test_tool_result_guidance_and_preserved_reasoning_history(self):
        target = subject("qwen3_5")
        messages = [{"role": "tool", "content": "result"}]
        guided = target._apply_qwen35_reasoning_effort_guidance(messages, "medium")
        self.assertIn("Do not repeat the completed tool call", guided[0]["content"])
        history = [{"role": "assistant", "content": None, "tool_calls": [{}],
                    "reasoning_content": "prior thought"}]
        target._expose_qwen35_reasoning_tool_history(history)
        self.assertEqual(history[0]["content"], "Prior reasoning:\nprior thought")
        self.assertIsNone(history[0]["reasoning_content"])


if __name__ == "__main__":
    unittest.main()
