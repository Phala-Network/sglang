"""Source-method CPU checks for the v0.5.20 DeepSeek V4.1 chat migration."""

import ast
import copy
import importlib.util
import logging
import sys
import types
import typing
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SRT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    packages = {}
    for package, package_path in (
        ("sglang", SRT.parent),
        ("sglang.srt", SRT),
        ("sglang.srt.phala_compat", SRT / "phala_compat"),
    ):
        namespace = types.ModuleType(package)
        namespace.__path__ = [str(package_path)]
        packages[package] = namespace
    with patch.dict(sys.modules, {name: module, **packages}):
        spec.loader.exec_module(module)
    return module


class NamedChoice:
    def __init__(self, name):
        self.function = SimpleNamespace(name=name)


class AllowedChoice:
    def __init__(self, names):
        self.allowed_tools = SimpleNamespace(
            mode="auto",
            tools=[SimpleNamespace(function=SimpleNamespace(name=n)) for n in names],
        )


class Tool:
    def __init__(self, name):
        self.function = SimpleNamespace(name=name)

    def model_dump(self, exclude_unset=False, exclude_none=False, **kwargs):
        function = {
            "parameters": {"type": "object"},
            "description": "description",
            "name": self.function.name,
        }
        if not exclude_unset:
            function["strict"] = False
        if not exclude_none:
            function["optional"] = None
        return {"type": "function", "function": function}


class ChatMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = load_module(
            "encoding_dsv41", SRT / "entrypoints/openai/encoding_dsv41.py"
        )
        cls.namespace = dict(
            vars(typing),
            encoding_dsv41=cls.encoder,
            encoding_dsv4=SimpleNamespace(),
            logging=logging,
            Path=Path,
            ast=ast,
        )
        parsed = ast.parse(
            (SRT / "entrypoints/openai/chat_encoding.py").read_text(encoding="utf-8")
        )
        nodes = [
            n for n in parsed.body if not isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        exec(
            compile(
                ast.Module(nodes, type_ignores=[]), "<actual-chat-encoding>", "exec"
            ),
            cls.namespace,
        )
        parsed = ast.parse(
            (SRT / "entrypoints/openai/serving_chat.py").read_text(encoding="utf-8")
        )
        owner = next(
            n
            for n in parsed.body
            if isinstance(n, ast.ClassDef) and n.name == "OpenAIServingChat"
        )
        methods = [
            n
            for n in owner.body
            if isinstance(n, ast.FunctionDef)
            and n.name
            in (
                "_allowed_tool_names",
                "_effective_tool_choice",
                "_request_tools_for_prompt",
            )
        ]
        cls.namespace.update(
            ChatCompletionRequest=object,
            ToolChoice=NamedChoice,
            AllowedToolsChoice=AllowedChoice,
        )
        wrapper = ast.ClassDef(
            name="Serving", bases=[], keywords=[], body=methods, decorator_list=[]
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module([wrapper], type_ignores=[])),
                "<actual-serving-methods>",
                "exec",
            ),
            cls.namespace,
        )

    def test_model_type_precedes_ambiguous_v4_architecture(self):
        resolve = self.namespace["resolve_chat_encoding_spec"]
        self.assertEqual(
            resolve(
                hf_config=SimpleNamespace(
                    architectures=["DeepseekV4ForCausalLM"], model_type="deepseek_v41"
                ),
                tokenizer=None,
            ),
            "dsv41",
        )
        self.assertEqual(
            resolve(
                hf_config=SimpleNamespace(
                    architectures=["DeepseekV4ForCausalLM"], model_type="deepseek_v4"
                ),
                tokenizer=None,
            ),
            "dsv4",
        )
        for name in (
            "Qwen3ForCausalLM",
            "Gemma4ForConditionalGeneration",
            "NemotronHForCausalLM",
        ):
            self.assertIsNone(
                resolve(hf_config=SimpleNamespace(architectures=[name]), tokenizer=None)
            )

    def test_explicit_parser_routing_is_retained(self):
        for name, spec in (
            ("deepseekv41", "dsv41"),
            ("deepseekv4", "dsv4"),
            ("deepseekv32", "dsv32"),
            ("kimi_k3", "kimi_k3"),
        ):
            self.assertEqual(
                self.namespace["resolve_chat_encoding_spec"](
                    hf_config=None, tokenizer=None, tool_call_parser=name
                ),
                spec,
            )

    def test_budget_validation_is_model_local_and_rejects_invalid_types(self):
        parse = self.namespace["parse_dsv41_reasoning_effort"]
        for value in (None, True, False, 0, 101, -1, 1.0, {}, [], "invalid"):
            self.assertIsNone(parse(value), value)
        for value in (1, 50, 100, "low", "medium", "high", "xhigh", "max"):
            self.assertEqual(parse(value), value)
        self.assertEqual(parse(0.5), 50)
        self.assertEqual(parse(0.0), 1)

    def test_environment_invalid_budget_fails_at_startup(self):
        default = self.namespace["default_dsv41_reasoning_effort_from_env"]
        self.assertEqual(default(""), "high")
        self.assertEqual(default(" 75 "), 75)
        self.assertEqual(default("medium"), "medium")
        for value in ("0", "101", "nan", "invalid"):
            with self.assertRaises(ValueError):
                default(value)

    def test_allowed_and_named_tools_remain_filtered_before_ds_serialization(self):
        serving = self.namespace["Serving"]()
        for choice, expected in (
            (AllowedChoice(["b"]), ["b"]),
            (NamedChoice("a"), ["a"]),
            ("auto", ["a", "b"]),
        ):
            request = SimpleNamespace(tools=[Tool("a"), Tool("b")], tool_choice=choice)
            tools = serving._request_tools_for_prompt(
                request, exclude_unset=True, exclude_none=True
            )
            result = [self.namespace["dsv41_tool_payload"](tool) for tool in tools]
            self.assertEqual([t["function"]["name"] for t in result], expected)
            for tool in result:
                self.assertEqual(
                    list(tool["function"]), ["name", "description", "parameters"]
                )
                self.assertNotIn("strict", tool["function"])
                self.assertNotIn("optional", tool["function"])

    def test_non_ds_prompt_default_fields_remain_unchanged(self):
        request = SimpleNamespace(tools=[Tool("a")], tool_choice="auto")
        output = self.namespace["Serving"]()._request_tools_for_prompt(request)
        self.assertFalse(output[0]["function"]["strict"])
        self.assertIsNone(output[0]["function"]["optional"])

    def test_ds_payload_order_does_not_mutate_input(self):
        payload = Tool("a").model_dump(exclude_unset=True, exclude_none=True)
        original = copy.deepcopy(payload)
        self.namespace["dsv41_tool_payload"](payload)
        self.assertEqual(payload, original)

    def test_encoder_does_not_invent_empty_system_message(self):
        messages = [{"role": "user", "content": "hello"}]
        original = copy.deepcopy(messages)
        text = self.encoder.encode_messages(messages, thinking_mode="chat")
        self.assertNotIn(self.encoder.SYSTEM_SP_TOKEN, text)
        self.assertIn(self.encoder.USER_SP_TOKEN, text)
        self.assertEqual(messages, original)

    def test_encoder_retains_image_order_and_placeholder(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.invalid/a.png"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.invalid/b.png"},
                    },
                ],
            }
        ]
        text, media = self.encoder.encode_messages(
            messages, thinking_mode="chat", return_multi_modal_data=True
        )
        self.assertEqual(text.count(self.encoder.IMAGE_PLACEHOLDER), 2)
        self.assertEqual(
            [image["url"] for image in media["images"]],
            ["https://example.invalid/a.png", "https://example.invalid/b.png"],
        )

    def test_encoder_spaced_dsml_and_reasoning_history(self):
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "plan",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "x",
                        "function": {"name": "lookup", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "content": "done", "tool_call_id": "x"},
            {"role": "user", "content": "continue"},
        ]
        text = self.encoder.encode_messages(
            messages, thinking_mode="thinking", drop_thinking=False
        )
        self.assertIn("<\uff5cDSML\uff5c calls>", text)
        self.assertIn('<\uff5cDSML\uff5c invoke name="lookup">', text)
        self.assertIn("plan", text)
        self.assertIn("<tool_result>done</tool_result>", text)


if __name__ == "__main__":
    unittest.main()
