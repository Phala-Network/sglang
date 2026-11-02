"""Exercise allowed-tools output policy using real serving and Qwen parsing."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def call(name, value="ok"):
    return (
        f"<tool_call>\n<function={name}>\n"
        f"<parameter=value>{value}</parameter>\n"
        "</function>\n</tool_call>"
    )


def request(*, allowed=True, stream=False, mode="auto"):
    names = ("get_weather", "get_time", "old_tool")
    return ChatCompletionRequest(
        model="test",
        messages=[
            {"role": "user", "content": "Check the weather."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "old",
                        "type": "function",
                        "function": {
                            "name": "old_tool",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "old", "content": "Historical result"},
            {"role": "user", "content": "Use only the currently allowed tools."},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                        },
                    },
                },
            }
            for name in names
        ],
        tool_choice={
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": mode,
                "tools": [
                    {"type": "function", "function": {"name": name}}
                    for name in names[:2]
                ],
            },
        }
        if allowed
        else mode,
        stream=stream,
        stream_options={"include_usage": True} if stream else None,
    )


def result(text, *, finished=True, tokens=12, index=0):
    return {
        "text": text,
        "index": index,
        "meta_info": {
            "id": "chatcmpl-test",
            "prompt_tokens": 5,
            "completion_tokens": tokens,
            "cached_tokens": 0,
            "weight_version": "fixture-version",
            "finish_reason": {"type": "stop", "matched": None} if finished else None,
        },
    }


class AllowedToolsOutputTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.chat = object.__new__(OpenAIServingChat)
        self.chat.tool_call_parser = "qwen3_coder"
        self.chat.reasoning_parser = None
        self.chat.chat_encoding_spec = None
        self.chat.tokenizer_manager = SimpleNamespace(tokenizer=None)
        serving = SimpleNamespace(
            return_input_ids=False,
            return_output_ids=False,
            enable_cache_report=False,
            incremental_streaming_output=False,
            stream_response_default_include_usage=False,
        )
        self.context = patch(
            "sglang.srt.entrypoints.openai.serving_chat.get_serving",
            return_value=serving,
        )
        self.context.start()
        self.addCleanup(self.context.stop)

    def test_nonstream_historical_tool_outside_subset_is_error(self):
        for mode in ("auto", "required"):
            with self.subTest(mode=mode):
                response = self.chat._build_chat_response(
                    request(mode=mode),
                    [result(call("get_weather") + call("old_tool", "private"))],
                    0,
                )
                self.assertEqual(getattr(response, "status_code", None), 500)
                body = json.loads(response.body)
                self.assertEqual(body["type"], "InternalServerError")
                self.assertIn("allowed_tools", body["message"])
                self.assertNotIn("choices", body)
                self.assertNotIn("private", response.body.decode())

    def test_nonstream_allowed_parallel_tools_keep_finish_and_usage(self):
        response = self.chat._build_chat_response(
            request(),
            [result(call("get_weather") + call("get_time"))],
            0,
        )
        self.assertEqual(
            [c.function.name for c in response.choices[0].message.tool_calls],
            ["get_weather", "get_time"],
        )
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")
        self.assertEqual(response.usage.completion_tokens, 12)

    def test_nonstream_without_subset_keeps_existing_tool_behavior(self):
        response = self.chat._build_chat_response(
            request(allowed=False),
            [result(call("old_tool"))],
            0,
        )
        self.assertEqual(
            response.choices[0].message.tool_calls[0].function.name, "old_tool"
        )
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")

    async def stream(self, req, outputs):
        closed = []

        async def backend(*_args):
            try:
                for output in outputs:
                    yield output
            finally:
                closed.append(True)

        self.chat.tokenizer_manager.generate_request = backend
        chunks = [
            item async for item in self.chat._generate_chat_stream(None, req, None)
        ]
        self.assertEqual(closed, [True])
        self.assertEqual(chunks.count("data: [DONE]\n\n"), 1)
        return [json.loads(item[6:]) for item in chunks if item != "data: [DONE]\n\n"]

    async def test_stream_rejects_disallowed_name_before_name_or_arguments_emit(self):
        for prefix in ("", call("get_weather")):
            with self.subTest(prior_legal_call=bool(prefix)):
                outputs = [result(prefix, finished=False)] if prefix else []
                outputs += [result(prefix + call("old_tool", "private"))]
                chunks = await self.stream(request(stream=True), outputs)
                errors = [chunk["error"] for chunk in chunks if "error" in chunk]
                self.assertEqual(len(errors), 1)
                self.assertEqual(errors[0]["code"], 500)
                self.assertEqual(errors[0]["type"], "InternalServerError")
                self.assertIn("allowed_tools", errors[0]["message"])
                serialized = json.dumps(chunks)
                self.assertNotIn("old_tool", serialized)
                self.assertNotIn("private", serialized)
                choices = [
                    choice for chunk in chunks for choice in chunk.get("choices", [])
                ]
                self.assertFalse(any(choice.get("finish_reason") for choice in choices))
                self.assertFalse(any(chunk.get("usage") for chunk in chunks))
                if prefix:
                    self.assertIn("get_weather", serialized)

    async def test_stream_allowed_parallel_tools_finish_and_usage_unchanged(self):
        text = call("get_weather") + call("get_time")
        chunks = await self.stream(
            request(stream=True),
            [
                result(text[:36], finished=False, tokens=3),
                result(text),
            ],
        )
        choices = [choice for chunk in chunks for choice in chunk.get("choices", [])]
        calls = [
            call
            for choice in choices
            for call in choice.get("delta", {}).get("tool_calls", [])
        ]
        self.assertEqual(
            [
                call["function"]["name"]
                for call in calls
                if call["function"].get("name")
            ],
            ["get_weather", "get_time"],
        )
        self.assertEqual(
            [
                choice["finish_reason"]
                for choice in choices
                if choice.get("finish_reason")
            ],
            ["tool_calls"],
        )
        self.assertEqual(
            [
                chunk["usage"]["completion_tokens"]
                for chunk in chunks
                if chunk.get("usage")
            ],
            [12],
        )
        self.assertFalse(any("error" in chunk for chunk in chunks))

    async def test_stream_without_subset_keeps_existing_tool_behavior(self):
        chunks = await self.stream(
            request(allowed=False, stream=True), [result(call("old_tool"))]
        )
        self.assertIn("old_tool", json.dumps(chunks))
        self.assertFalse(any("error" in chunk for chunk in chunks))

    async def test_stream_orphan_other_index_does_not_borrow_valid_name(self):
        parser = SimpleNamespace(
            parse_stream_chunk=lambda _: (
                "",
                [
                    ToolCallItem(tool_index=0, name="get_weather", parameters="{}"),
                    ToolCallItem(tool_index=1, parameters="private"),
                ],
            )
        )
        chunks = [
            item
            async for item in self.chat._process_tool_call_stream(
                0,
                "",
                {0: parser},
                result(""),
                request(stream=True),
                {},
            )
        ]
        self.assertEqual(len(chunks), 1)
        self.assertNotIn("private", "".join(chunks))

    def test_terminal_argument_flush_cannot_bypass_subset(self):
        parser = SimpleNamespace(
            prev_tool_call_arr=[
                {"name": "old_tool", "arguments": {"value": "private"}}
            ],
            streamed_args_for_tool=[""],
        )
        with self.assertRaisesRegex(ValueError, "allowed_tools"):
            self.chat._check_for_unstreamed_tool_args(
                parser, result(""), request(stream=True), 0
            )

    async def test_allowed_terminal_flush_preserves_admitted_arguments(self):
        parser = SimpleNamespace(
            parse_stream_chunk=lambda _: (
                "",
                [
                    ToolCallItem(
                        tool_index=0,
                        name="get_weather",
                        parameters='{"value":',
                    )
                ],
            ),
            prev_tool_call_arr=[{"name": "get_weather", "arguments": {"value": "ok"}}],
            streamed_args_for_tool=['{"value":'],
        )
        req = request(stream=True)
        chunks = [
            item
            async for item in self.chat._process_tool_call_stream(
                0,
                "",
                {0: parser},
                result(""),
                req,
                {},
            )
        ]
        trailing = self.chat._check_for_unstreamed_tool_args(parser, result(""), req, 0)
        self.assertIsNotNone(trailing)
        chunks.append(trailing)
        arguments = "".join(
            json.loads(item[6:])["choices"][0]["delta"]["tool_calls"][0]["function"][
                "arguments"
            ]
            for item in chunks
        )
        self.assertEqual(json.loads(arguments), {"value": "ok"})

    async def test_stream_choice_does_not_borrow_another_choices_admitted_index(self):
        parsers = {
            0: SimpleNamespace(
                parse_stream_chunk=lambda _: (
                    "",
                    [
                        ToolCallItem(tool_index=0, name="get_weather", parameters="{}"),
                    ],
                )
            ),
            1: SimpleNamespace(
                parse_stream_chunk=lambda _: (
                    "",
                    [
                        ToolCallItem(tool_index=0, parameters="private"),
                    ],
                )
            ),
        }
        has_calls = {}
        emitted = []
        for index in (0, 1):
            emitted.extend(
                [
                    item
                    async for item in self.chat._process_tool_call_stream(
                        index,
                        "",
                        parsers,
                        result("", index=index),
                        request(stream=True),
                        has_calls,
                    )
                ]
            )
        self.assertEqual(len(emitted), 1)
        self.assertEqual(has_calls, {0: True})
        self.assertNotIn("private", "".join(emitted))


if __name__ == "__main__":
    unittest.main()
