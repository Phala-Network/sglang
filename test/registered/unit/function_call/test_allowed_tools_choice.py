import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.protocol import (  # noqa: E402
    AllowedToolsChoice,
    ChatCompletionRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Call {name}",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }


def make_allowed_choice(mode: str, *names: str) -> dict:
    return {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": mode,
            "tools": [
                {"type": "function", "function": {"name": name}} for name in names
            ],
        },
    }


def make_request(
    tool_choice: object, response_format: dict | None = None
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "Use one tool."}],
        tools=[make_tool("get_weather"), make_tool("delete_file")],
        tool_choice=tool_choice,
        response_format=response_format,
    )


class AllowedToolsProtocolTest(unittest.TestCase):
    def test_accepts_and_normalizes_official_nested_shape(self):
        for mode in ("auto", "required"):
            with self.subTest(mode=mode):
                request = make_request(make_allowed_choice(mode, "get_weather"))
                self.assertIsInstance(request.tool_choice, AllowedToolsChoice)
                self.assertEqual(request.effective_tool_choice(), mode)
                self.assertEqual(
                    request.tool_choice.allowed_tools.tools[0].function.name,
                    "get_weather",
                )

    def test_rejects_invalid_empty_and_duplicate_subsets(self):
        invalid_choices = [
            make_allowed_choice("sometimes", "get_weather"),
            make_allowed_choice("required"),
            make_allowed_choice("auto", "get_weather", "get_weather"),
        ]
        for choice in invalid_choices:
            with self.subTest(choice=choice), self.assertRaises(ValidationError):
                make_request(choice)

    def test_required_subset_conflicts_with_response_format(self):
        request = make_request(
            make_allowed_choice("required", "get_weather"),
            response_format={"type": "json_object"},
        )
        with self.assertRaisesRegex(ValueError, "allowed_tools mode 'required'"):
            request.to_sampling_params(
                stop=[],
                model_generation_config={},
                tool_call_constraint=("structural_tag", object()),
            )


class AllowedToolsServingTest(unittest.TestCase):
    def setUp(self):
        self.chat = object.__new__(OpenAIServingChat)
        self.chat._grammar_backend = "xgrammar"
        self.chat.tokenizer_manager = SimpleNamespace(
            model_config=SimpleNamespace(is_multimodal=True),
            server_args=SimpleNamespace(
                context_length=262144,
                allow_auto_truncate=False,
            ),
        )

    def test_filters_parser_and_prompt_tools_to_allowed_subset(self):
        request = make_request(make_allowed_choice("required", "get_weather"))
        self.assertEqual(
            [tool.function.name for tool in self.chat._effective_tools(request)],
            ["get_weather"],
        )
        self.assertEqual(
            [
                tool["function"]["name"]
                for tool in self.chat._request_tools_for_prompt(request)
            ],
            ["get_weather"],
        )

    def test_filters_message_level_tools_before_template_render(self):
        request = make_request(make_allowed_choice("auto", "get_weather"))
        messages = [
            {
                "role": "system",
                "content": "Tools",
                "tools": [make_tool("get_weather"), make_tool("delete_file")],
            }
        ]
        self.chat._filter_message_tools_for_prompt(messages, request)
        self.assertEqual(
            [tool["function"]["name"] for tool in messages[0]["tools"]],
            ["get_weather"],
        )

    def test_unknown_allowed_tool_fails_closed(self):
        request = make_request(make_allowed_choice("required", "missing_tool"))
        self.assertEqual(
            self.chat._validate_request(request),
            "Allowed tool(s) not found in tools list: missing_tool.",
        )


if __name__ == "__main__":
    unittest.main()
