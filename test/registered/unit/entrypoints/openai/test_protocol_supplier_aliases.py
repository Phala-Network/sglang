import unittest

from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def tool(name: str):
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


class TestAllowedToolsChoice(unittest.TestCase):
    def request(self, tool_choice, *, tools=None, messages=None):
        return ChatCompletionRequest(
            model="test-model",
            messages=messages or [{"role": "user", "content": "Use a tool."}],
            tools=tools,
            tool_choice=tool_choice,
        )

    def test_nested_openai_shape_filters_and_reuses_required(self):
        request = self.request(
            {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "required",
                    "tools": [{"type": "function", "function": {"name": "weather"}}],
                },
            },
            tools=[tool("weather"), tool("search")],
        )
        self.assertEqual(request.tool_choice, "required")
        self.assertEqual([item.function.name for item in request.tools], ["weather"])

    def test_flat_router_alias_filters_and_reuses_auto(self):
        request = self.request(
            {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "search"}],
            },
            tools=[tool("weather"), tool("search")],
        )
        self.assertEqual(request.tool_choice, "auto")
        self.assertEqual([item.function.name for item in request.tools], ["search"])

    def test_message_level_tools_are_filtered(self):
        request = self.request(
            {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "required",
                    "tools": [{"type": "function", "function": {"name": "search"}}],
                },
            },
            messages=[
                {
                    "role": "system",
                    "content": "Use tools.",
                    "tools": [tool("weather"), tool("search")],
                },
                {"role": "user", "content": "Search."},
            ],
        )
        self.assertEqual(request.tool_choice, "required")
        self.assertEqual(
            [item.function.name for item in request.messages[0].tools], ["search"]
        )

    def test_unknown_or_duplicate_allowed_name_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "not present"):
            self.request(
                {
                    "type": "allowed_tools",
                    "allowed_tools": {
                        "mode": "required",
                        "tools": [
                            {"type": "function", "function": {"name": "missing"}}
                        ],
                    },
                },
                tools=[tool("weather")],
            )
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            self.request(
                {
                    "type": "allowed_tools",
                    "allowed_tools": {
                        "mode": "required",
                        "tools": [
                            {"type": "function", "function": {"name": "weather"}},
                            {"type": "function", "function": {"name": "weather"}},
                        ],
                    },
                },
                tools=[tool("weather")],
            )


class TestThinkingAliases(unittest.TestCase):
    def request(self, **kwargs):
        return ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Think."}],
            **kwargs,
        )

    def assert_toggle(self, request, enabled: bool):
        self.assertEqual(request.chat_template_kwargs["thinking"], enabled)
        self.assertEqual(request.chat_template_kwargs["enable_thinking"], enabled)

    def test_top_level_enable_thinking(self):
        self.assert_toggle(self.request(enable_thinking=True), True)
        self.assert_toggle(self.request(enable_thinking=False), False)

    def test_top_level_thinking_type(self):
        self.assert_toggle(self.request(thinking={"type": "enabled"}), True)
        self.assert_toggle(self.request(thinking={"type": "disabled"}), False)

    def test_chat_template_aliases_are_synchronized(self):
        self.assert_toggle(
            self.request(chat_template_kwargs={"enable_thinking": True}), True
        )
        self.assert_toggle(
            self.request(chat_template_kwargs={"thinking": False}), False
        )

    def test_reasoning_and_effort_precedence(self):
        self.assert_toggle(
            self.request(enable_thinking=True, reasoning={"enabled": False}), False
        )
        self.assert_toggle(
            self.request(
                enable_thinking=False,
                reasoning={"enabled": False},
                reasoning_effort="high",
            ),
            True,
        )

    def test_explicit_chat_template_value_wins_alias_default(self):
        request = self.request(
            enable_thinking=True,
            chat_template_kwargs={"thinking": False},
        )
        self.assertFalse(request.chat_template_kwargs["thinking"])
        self.assertFalse(request.chat_template_kwargs["enable_thinking"])

    def test_include_reasoning_legacy_visibility_and_default_enable(self):
        included = self.request(include_reasoning=True)
        self.assertTrue(included.include_reasoning)
        self.assertFalse(included.reasoning_exclude)
        self.assert_toggle(included, True)

        excluded = self.request(include_reasoning=False)
        self.assertFalse(excluded.include_reasoning)
        self.assertTrue(excluded.reasoning_exclude)
        # Excluding the returned trace must not itself disable internal
        # reasoning. With no other control the model/template default remains.
        self.assertIsNone(excluded.chat_template_kwargs)

    def test_nested_reasoning_exclude_wins_legacy_visibility(self):
        request = self.request(
            include_reasoning=True,
            reasoning={"enabled": True, "exclude": True},
        )
        self.assertTrue(request.reasoning_exclude)
        self.assert_toggle(request, True)

        request = self.request(
            include_reasoning=False,
            reasoning={"effort": "high", "exclude": False},
        )
        self.assertFalse(request.reasoning_exclude)
        self.assertEqual(request.reasoning_effort, "high")
        self.assert_toggle(request, True)

    def test_explicit_reasoning_disable_wins_include_reasoning(self):
        request = self.request(
            include_reasoning=True,
            reasoning={"enabled": False},
        )
        self.assertFalse(request.reasoning_exclude)
        self.assert_toggle(request, False)

    def test_reasoning_visibility_controls_require_booleans(self):
        for value in (1, 0, "true", "false", {}, []):
            with self.subTest(include_reasoning=value):
                with self.assertRaisesRegex(
                    ValidationError, "include_reasoning must be a boolean"
                ):
                    self.request(include_reasoning=value)

        for value in (1, 0, "true", "false", {}, []):
            with self.subTest(reasoning_exclude=value):
                with self.assertRaisesRegex(
                    ValidationError, "reasoning.exclude must be a boolean"
                ):
                    self.request(reasoning={"exclude": value})

    def test_invalid_alias_values_are_rejected(self):
        with self.assertRaisesRegex(ValidationError, "invalid enable_thinking"):
            self.request(enable_thinking={"bad": True})
        with self.assertRaisesRegex(ValidationError, "invalid thinking.type"):
            self.request(thinking={"type": "sometimes"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
