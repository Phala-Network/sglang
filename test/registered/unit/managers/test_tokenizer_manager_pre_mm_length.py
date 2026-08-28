"""Regression tests for rejecting over-context requests before MM processing."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.test_utils import CustomTestCase


class _MMProcessorEntered(RuntimeError):
    pass


def _make_manager(*, allow_auto_truncate=False):
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.context_len = 8
    manager.num_reserved_tokens = 1
    manager.validate_total_tokens = True
    manager.allow_auto_truncate = allow_auto_truncate
    manager.tokenizer = MagicMock()
    manager.max_req_input_len = 8
    manager.server_args = SimpleNamespace(
        language_model_only=False,
        language_only=False,
    )
    manager.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[]),
    )
    manager._validate_mm_limits = MagicMock()
    manager.mm_processor = MagicMock()
    manager.mm_processor.prefer_tokenized_input = True
    manager.mm_processor.process_mm_data_async = AsyncMock(
        side_effect=_MMProcessorEntered("mm processor entered")
    )
    return manager


def _make_request(*, input_tokens, max_new_tokens=0, image_data=b"not-an-image"):
    request = MagicMock(spec=GenerateReqInput)
    request.text = "ignored because input_ids are supplied"
    request.input_embeds = None
    request.input_ids = [1] * input_tokens
    request.image_data = image_data
    request.video_data = None
    request.audio_data = None
    request.mm_content_hashes = None
    request.sampling_params = {"max_new_tokens": max_new_tokens}
    request.contains_mm_input.return_value = True
    return request


def _run_tokenize(manager, request):
    with patch(
        "sglang.srt.managers.tokenizer_manager.get_disagg",
        return_value=SimpleNamespace(
            language_only=False,
            encoder_transfer_backend=None,
        ),
    ):
        return asyncio.run(manager._tokenize_one_request(request))


class TestPreMultimodalLengthValidation(CustomTestCase):
    def test_text_budget_at_context_rejects_before_media_decode(self):
        manager = _make_manager()
        request = _make_request(input_tokens=7)

        with self.assertRaisesRegex(ValueError, "longer than"):
            _run_tokenize(manager, request)

        manager._validate_mm_limits.assert_not_called()
        manager.mm_processor.process_mm_data_async.assert_not_awaited()

    def test_corrupt_media_is_not_touched_when_text_is_already_over_context(self):
        manager = _make_manager()
        request = _make_request(
            input_tokens=8,
            image_data=b"corrupt-jpeg-payload",
        )

        with self.assertRaisesRegex(ValueError, "longer than"):
            _run_tokenize(manager, request)

        manager.mm_processor.process_mm_data_async.assert_not_awaited()

    def test_input_plus_requested_output_rejects_before_media_decode(self):
        manager = _make_manager()
        request = _make_request(input_tokens=5, max_new_tokens=3)

        with self.assertRaisesRegex(ValueError, "Requested token count exceeds"):
            _run_tokenize(manager, request)

        manager.mm_processor.process_mm_data_async.assert_not_awaited()

    def test_request_at_total_budget_reaches_media_processor(self):
        manager = _make_manager()
        request = _make_request(input_tokens=5, max_new_tokens=2)

        with self.assertRaises(_MMProcessorEntered):
            _run_tokenize(manager, request)

        manager._validate_mm_limits.assert_called_once_with(request)
        manager.mm_processor.process_mm_data_async.assert_awaited_once()

    def test_auto_truncate_still_defers_to_post_mm_validation(self):
        manager = _make_manager(allow_auto_truncate=True)
        request = _make_request(input_tokens=8)

        with self.assertRaises(_MMProcessorEntered):
            _run_tokenize(manager, request)

        manager.mm_processor.process_mm_data_async.assert_awaited_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
