import unittest

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import (
    apply_muse_structured_output_reasoning_default,
)


def make_request(**overrides):
    body = {
        "model": "meta/muse-glimmer-30b",
        "messages": [{"role": "user", "content": "Return JSON."}],
    }
    body.update(overrides)
    return ChatCompletionRequest.model_validate(body)


class TestMuseReasoningAliases(unittest.TestCase):
    def assert_reasoning_disabled(self, **overrides):
        request = make_request(**overrides)
        self.assertEqual(
            request.chat_template_kwargs.get("reasoning_strength"), "none"
        )

    def test_top_level_enable_thinking_false(self):
        self.assert_reasoning_disabled(enable_thinking=False)

    def test_thinking_type_disabled(self):
        self.assert_reasoning_disabled(thinking={"type": "disabled"})

    def test_chat_template_enable_thinking_false(self):
        self.assert_reasoning_disabled(
            chat_template_kwargs={"enable_thinking": False}
        )

    def test_chat_template_thinking_false(self):
        self.assert_reasoning_disabled(chat_template_kwargs={"thinking": False})

    def test_standard_reasoning_enabled_wins_over_alias(self):
        request = make_request(
            reasoning={"enabled": True}, thinking={"type": "disabled"}
        )
        self.assertNotEqual(
            request.chat_template_kwargs.get("reasoning_strength"), "none"
        )
        self.assertIs(request.chat_template_kwargs.get("thinking"), True)

    def test_reasoning_effort_wins_over_disabled_alias(self):
        request = make_request(reasoning_effort="high", enable_thinking=False)
        self.assertEqual(request.reasoning_effort, "high")
        self.assertEqual(
            request.chat_template_kwargs.get("reasoning_strength"), "high"
        )

    def test_explicit_reasoning_strength_is_preserved(self):
        request = make_request(
            enable_thinking=False,
            chat_template_kwargs={"reasoning_strength": "max"},
        )
        self.assertEqual(
            request.chat_template_kwargs.get("reasoning_strength"), "max"
        )


class TestMuseStructuredOutputReasoningDefault(unittest.TestCase):
    def test_json_object_defaults_to_direct_final(self):
        request = make_request(response_format={"type": "json_object"})
        apply_muse_structured_output_reasoning_default(request, "muse")
        self.assertEqual(request.chat_template_kwargs, {"reasoning_strength": "none"})

    def test_json_schema_defaults_to_direct_final(self):
        request = make_request(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                },
            }
        )
        apply_muse_structured_output_reasoning_default(request, "muse")
        self.assertEqual(request.chat_template_kwargs, {"reasoning_strength": "none"})

    def test_plain_chat_keeps_checkpoint_default(self):
        request = make_request()
        apply_muse_structured_output_reasoning_default(request, "muse")
        self.assertIsNone(request.chat_template_kwargs)

    def test_other_reasoning_parser_is_unchanged(self):
        request = make_request(response_format={"type": "json_object"})
        apply_muse_structured_output_reasoning_default(request, "qwen3")
        self.assertIsNone(request.chat_template_kwargs)

    def test_explicit_effort_is_preserved(self):
        request = make_request(
            response_format={"type": "json_object"}, reasoning_effort="low"
        )
        before = dict(request.chat_template_kwargs)
        apply_muse_structured_output_reasoning_default(request, "muse")
        self.assertEqual(request.reasoning_effort, "low")
        self.assertEqual(request.chat_template_kwargs, before)

    def test_explicit_reasoning_enabled_is_preserved(self):
        request = make_request(
            response_format={"type": "json_object"},
            reasoning={"enabled": True},
        )
        apply_muse_structured_output_reasoning_default(request, "muse")
        self.assertIs(request.chat_template_kwargs.get("thinking"), True)
        self.assertNotEqual(
            request.chat_template_kwargs.get("reasoning_strength"), "none"
        )

    def test_include_reasoning_true_keeps_checkpoint_default(self):
        request = make_request(
            response_format={"type": "json_schema", "json_schema": None},
            include_reasoning=True,
        )
        apply_muse_structured_output_reasoning_default(request, "muse")
        self.assertIsNone(request.chat_template_kwargs)

    def test_include_reasoning_false_still_defaults_to_direct_final(self):
        request = make_request(
            response_format={"type": "json_object"},
            include_reasoning=False,
        )
        apply_muse_structured_output_reasoning_default(request, "muse")
        self.assertEqual(request.chat_template_kwargs, {"reasoning_strength": "none"})


if __name__ == "__main__":
    unittest.main()
