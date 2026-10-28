"""Exercise preferred defaults through real OpenAI conversion and tokenization."""

import json
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_serving_chat import _MockTemplateManager, _MockTokenizerManager

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    MessageProcessingResult,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestPreferredSamplingDefaults(unittest.TestCase):
    def setUp(self):
        self.addCleanup(reset_context)

    def _handlers(self, preferred=None, model_defaults=None):
        reset_context()
        publish(
            ServerArgs(model_path="dummy", preferred_sampling_params=preferred),
            role="tokenizer",
        )
        manager = _MockTokenizerManager()
        manager.preferred_sampling_params = preferred
        manager.model_config.get_default_sampling_params.return_value = (
            model_defaults or {}
        )
        template = _MockTemplateManager()
        chat = OpenAIServingChat(manager, template)
        completion = OpenAIServingCompletion(manager, template)
        return manager, chat, completion

    def _convert(self, handler, overrides=None):
        overrides = overrides or {}
        if isinstance(handler, OpenAIServingChat):
            request = ChatCompletionRequest(
                model="fixture",
                messages=[{"role": "user", "content": "fixture"}],
                max_tokens=2000,
                **overrides,
            )
            processed = MessageProcessingResult(
                prompt="fixture",
                prompt_ids=[7, 8],
                image_data=None,
                audio_data=None,
                video_data=None,
                modalities=[],
                stop=[],
            )
            with (
                patch.object(handler, "_process_messages", return_value=processed),
                patch.object(
                    handler, "_can_reuse_text_only_prompt_ids", return_value=True
                ),
            ):
                adapted, _ = handler._convert_to_internal_request(request)
        else:
            request = CompletionRequest(
                model="fixture", prompt=[7, 8], max_tokens=2000, **overrides
            )
            adapted, _ = handler._convert_to_internal_request(request)
        return request, adapted

    def _tokenize(self, adapted, preferred):
        manager = object.__new__(TokenizerManager)
        manager.preferred_sampling_params = preferred
        manager.sampling_params_class = SamplingParams
        manager.tokenizer = Mock()
        manager.model_config = SimpleNamespace(vocab_size=512)
        manager.rid_to_state = defaultdict(lambda: SimpleNamespace(time_stats=Mock()))
        return manager._create_tokenized_object(adapted, "fixture", [7, 8])

    def test_real_cli_parses_existing_opt_in(self):
        preferred = {"repetition_penalty": 1.05}
        args = prepare_server_args(
            ["--model-path", "dummy", "--preferred-sampling-params", json.dumps(preferred)]
        )
        self.assertEqual(args.preferred_sampling_params, preferred)

    def test_omitted_chat_and_completion_penalty_reaches_real_tokenizer(self):
        preferred = {"repetition_penalty": 1.05}
        _, chat, completion = self._handlers(preferred)
        for handler in (chat, completion):
            with self.subTest(handler=type(handler).__name__):
                _, adapted = self._convert(handler)
                self.assertEqual(adapted.sampling_params["repetition_penalty"], 1.05)
                native = self._tokenize(adapted, preferred)
                self.assertEqual(native.sampling_params.repetition_penalty, 1.05)
                self.assertEqual(native.sampling_params.max_new_tokens, 2000)

    def test_explicit_one_zero_and_false_keep_priority(self):
        preferred = {
            "repetition_penalty": 1.05,
            "frequency_penalty": 0.1,
            "temperature": 0.8,
            "ignore_eos": True,
        }
        _, chat, completion = self._handlers(preferred)
        explicit = {
            "repetition_penalty": 1.0,
            "frequency_penalty": 0,
            "temperature": 0,
            "ignore_eos": False,
        }
        for handler in (chat, completion):
            with self.subTest(handler=type(handler).__name__):
                request, adapted = self._convert(handler, explicit)
                before = request.model_dump()
                native = self._tokenize(adapted, preferred)
                self.assertEqual(native.sampling_params.repetition_penalty, 1.0)
                self.assertEqual(native.sampling_params.frequency_penalty, 0)
                self.assertEqual(native.sampling_params.temperature, 1.0)
                self.assertEqual(native.sampling_params.top_k, 1)
                self.assertFalse(native.sampling_params.ignore_eos)
                self.assertEqual(adapted.sampling_params["temperature"], 0)
                self.assertEqual(request.model_dump(), before)

    def test_no_preferred_keeps_existing_defaults_and_model_priority(self):
        model_defaults = {"temperature": 0.9, "top_p": 0.85, "repetition_penalty": 1.1}
        _, chat, completion = self._handlers(model_defaults=model_defaults)
        _, chat_request = self._convert(chat)
        _, completion_request = self._convert(completion)
        self.assertEqual(chat_request.sampling_params["temperature"], 0.9)
        self.assertEqual(chat_request.sampling_params["repetition_penalty"], 1.1)
        self.assertEqual(completion_request.sampling_params["temperature"], 1.0)
        self.assertEqual(completion_request.sampling_params["repetition_penalty"], 1.0)

    def test_all_seven_numeric_defaults_reach_both_adapters(self):
        preferred = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.1,
            "repetition_penalty": 1.05,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
        }
        _, chat, completion = self._handlers(preferred)
        for handler in (chat, completion):
            with self.subTest(handler=type(handler).__name__):
                _, adapted = self._convert(handler)
                native = self._tokenize(adapted, preferred)
                for name, expected in preferred.items():
                    self.assertEqual(getattr(native.sampling_params, name), expected)

    def test_boolean_default_behavior_is_unchanged(self):
        preferred = {"repetition_penalty": 1.05, "ignore_eos": True}
        _, chat, completion = self._handlers(preferred)
        for handler in (chat, completion):
            with self.subTest(handler=type(handler).__name__):
                _, adapted = self._convert(handler)
                self.assertFalse(
                    self._tokenize(adapted, preferred).sampling_params.ignore_eos
                )

    def test_real_message_template_processing_preserves_omitted_fields(self):
        preferred = {"repetition_penalty": 1.05}
        manager, chat, _ = self._handlers(preferred)
        chat.template_manager.chat_template_name = None
        chat.template_manager.jinja_template_content_format = "string"
        manager.tokenizer.apply_chat_template.return_value = "rendered prompt"
        manager.tokenizer.apply_chat_template.reset_mock()
        request = ChatCompletionRequest(
            model="fixture",
            messages=[{"role": "user", "content": "Original request."}],
            temperature=0,
            max_tokens=2000,
            chat_template_kwargs={"enable_thinking": False},
        )
        before = request.model_dump()
        explicit_before = set(request.model_fields_set)
        adapted, _ = chat._convert_to_internal_request(request)
        manager.tokenizer.apply_chat_template.assert_called_once()
        self.assertEqual(request.model_fields_set, explicit_before)
        self.assertNotIn("repetition_penalty", request.model_fields_set)
        self.assertEqual(request.model_dump(), before)
        self.assertEqual(adapted.sampling_params["repetition_penalty"], 1.05)
        self.assertEqual(adapted.sampling_params["temperature"], 0)
        self.assertEqual(adapted.sampling_params["max_new_tokens"], 2000)

    def test_preferred_beats_model_only_when_request_omits_value(self):
        preferred = {"temperature": 0.7, "repetition_penalty": 1.05}
        _, chat, _ = self._handlers(
            preferred, {"temperature": 0.9, "top_p": 0.85, "repetition_penalty": 1.1}
        )
        _, adapted = self._convert(chat)
        self.assertEqual(adapted.sampling_params["temperature"], 0.7)
        self.assertEqual(adapted.sampling_params["repetition_penalty"], 1.05)
        self.assertEqual(adapted.sampling_params["top_p"], 0.85)
        _, explicit = self._convert(
            chat, {"temperature": 0, "repetition_penalty": 1.0}
        )
        self.assertEqual(explicit.sampling_params["temperature"], 0)
        self.assertEqual(explicit.sampling_params["repetition_penalty"], 1.0)

    def test_nullable_chat_value_can_use_preferred_default(self):
        preferred = {"repetition_penalty": 1.05}
        _, chat, _ = self._handlers(preferred)
        _, adapted = self._convert(chat, {"repetition_penalty": None})
        self.assertEqual(adapted.sampling_params["repetition_penalty"], 1.05)

    def test_budget_choice_count_and_response_schema_are_preserved(self):
        preferred = {
            "repetition_penalty": 1.05,
            "max_new_tokens": 99,
            "n": 3,
            "json_schema": '{"type":"string"}',
        }
        _, chat, completion = self._handlers(preferred)
        for handler in (chat, completion):
            with self.subTest(handler=type(handler).__name__):
                _, adapted = self._convert(
                    handler, {"response_format": {"type": "json_object"}}
                )
                native = self._tokenize(adapted, preferred)
                self.assertEqual(native.sampling_params.max_new_tokens, 2000)
                self.assertEqual(native.sampling_params.n, 1)
                self.assertEqual(
                    json.loads(native.sampling_params.json_schema), {"type": "object"}
                )

    def test_native_sampling_merge_remains_unchanged(self):
        preferred = {
            "repetition_penalty": 1.05,
            "frequency_penalty": 0.1,
            "ignore_eos": True,
        }
        self._handlers(preferred)
        omitted = GenerateReqInput(input_ids=[7, 8], sampling_params={})
        native = self._tokenize(omitted, preferred)
        self.assertEqual(native.sampling_params.repetition_penalty, 1.05)
        explicit = GenerateReqInput(
            input_ids=[7, 8],
            sampling_params={
                "repetition_penalty": 1.0,
                "frequency_penalty": 0,
                "ignore_eos": False,
            },
        )
        native = self._tokenize(explicit, preferred)
        self.assertEqual(native.sampling_params.repetition_penalty, 1.0)
        self.assertEqual(native.sampling_params.frequency_penalty, 0)
        self.assertFalse(native.sampling_params.ignore_eos)


if __name__ == "__main__":
    unittest.main(verbosity=2)
