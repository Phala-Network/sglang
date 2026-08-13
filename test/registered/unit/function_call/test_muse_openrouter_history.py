import unittest

from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import ChatCompletionMessageGenericParam
from sglang.srt.entrypoints.openai.serving_chat import normalize_muse_history


class TestMuseOpenRouterHistory(unittest.TestCase):
    def test_reasoning_alias_is_preserved(self):
        message = ChatCompletionMessageGenericParam.model_validate(
            {
                "role": "assistant",
                "content": None,
                "reasoning": 'The secret word is "pistachio".',
            }
        )
        self.assertEqual(message.reasoning_content, 'The secret word is "pistachio".')
        self.assertNotIn("reasoning", message.model_dump())

    def test_standard_reasoning_content_wins_over_alias(self):
        message = ChatCompletionMessageGenericParam.model_validate(
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "standard",
                "reasoning": "legacy",
            }
        )
        self.assertEqual(message.reasoning_content, "standard")

    def test_reasoning_is_rejected_outside_assistant_role(self):
        with self.assertRaises(ValidationError):
            ChatCompletionMessageGenericParam.model_validate(
                {"role": "developer", "content": "x", "reasoning": "hidden"}
            )

    def test_complete_think_content_becomes_reasoning(self):
        messages = normalize_muse_history(
            [
                {
                    "role": "assistant",
                    "content": '<think>The secret word is "sassafras".</think>',
                    "tool_calls": [],
                }
            ]
        )
        self.assertEqual(messages[0]["reasoning_content"], 'The secret word is "sassafras".')
        self.assertEqual(messages[0]["content"], "")

    def test_embedded_think_tag_stays_content(self):
        content = "The literal example is <think>not private</think>."
        messages = normalize_muse_history(
            [{"role": "assistant", "content": content}]
        )
        self.assertEqual(messages[0]["content"], content)
        self.assertNotIn("reasoning_content", messages[0])

    def test_interleaved_developer_message_is_promoted(self):
        messages = normalize_muse_history(
            [
                {"role": "user", "content": "I need help."},
                {"role": "assistant", "content": "Send the text."},
                {
                    "role": "developer",
                    "content": "You must start your response with AFFIRMATIVE",
                },
                {"role": "user", "content": "Whats my name?"},
            ]
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(
            messages[0]["content"],
            "You must start your response with AFFIRMATIVE",
        )
        self.assertNotIn("developer", [message["role"] for message in messages])
        self.assertEqual(messages[-1]["content"], "Whats my name?")


if __name__ == "__main__":
    unittest.main()
