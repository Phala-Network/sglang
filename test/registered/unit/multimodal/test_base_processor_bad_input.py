"""Unit tests for bad-input classification in ``BaseMultimodalProcessor`` media loading.

Media the client supplied but that cannot be fetched or decoded must raise
``ValueError``; anything else must stay a ``RuntimeError``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import asyncio
import base64
import binascii
import errno
import io
import random
import struct
import traceback
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import requests
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils.common import CLIENT_MEDIA_EXCEPTIONS
from sglang.test.test_utils import CustomTestCase

MODALITIES = (Modality.IMAGE, Modality.AUDIO, Modality.VIDEO)


class _StubProcessor(BaseMultimodalProcessor):
    # gpu_image_decode=False keeps the decode on PIL so the test needs no GPU. The
    # abstract methods are never called: only the _load_single_item classmethod is.
    gpu_image_decode = False


def _session_raising(exc):
    session = MagicMock()
    session.get.side_effect = exc
    return session


class TestBadInputIsClientError(CustomTestCase):
    def _assert_client_error(self, data, modality):
        with self.assertRaises(ValueError):
            _StubProcessor._load_single_item(data, modality)

    def test_unfetchable_url_every_modality(self):
        # All three loaders fetch through get_mm_http_session(); HTTPError,
        # ConnectionError and Timeout all subclass RequestException.
        for exc in (
            requests.exceptions.HTTPError("404 from media host"),
            requests.exceptions.ConnectionError("dns failure"),
            requests.exceptions.Timeout("read timed out"),
        ):
            for modality in MODALITIES:
                with self.subTest(exc=type(exc).__name__, modality=modality):
                    with patch(
                        "sglang.srt.utils.common.get_mm_http_session",
                        return_value=_session_raising(exc),
                    ):
                        self._assert_client_error("https://media.host/clip", modality)

    def test_invalid_base64(self):
        self._assert_client_error("!!!not-base64!!!", Modality.IMAGE)

    def test_undecodable_image_bytes(self):
        # PIL raises UnidentifiedImageError, an OSError -- not a ValueError.
        self._assert_client_error(b"definitely not an image", Modality.IMAGE)

    def test_lazy_pil_decode_failure(self):
        malformed_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLJSwAAAABJRU5ErkJggg=="
        )
        for media in (malformed_png, Image.open(io.BytesIO(malformed_png))):
            with self.subTest(media_type=type(media).__name__):
                with self.assertRaisesRegex(
                    ValueError, "Could not load client media"
                ) as ctx:
                    _StubProcessor._load_single_item(media, Modality.IMAGE)
                self.assertIsInstance(ctx.exception.__context__.__context__, OSError)
                self.assertTrue(ctx.exception.__suppress_context__)

    def test_undecodable_audio_bytes(self):
        # soundfile raises LibsndfileError, a RuntimeError -- not a ValueError.
        self._assert_client_error(b"definitely not audio", Modality.AUDIO)

    def test_undecodable_video_bytes(self):
        # Decoder is patched so no codec backend needs to be installed.
        with patch(
            "sglang.srt.utils.common.VideoDecoderWrapper",
            side_effect=RuntimeError("invalid data found when processing input"),
        ):
            self._assert_client_error(b"definitely not a video", Modality.VIDEO)


class TestServerFaultStaysServerError(CustomTestCase):
    """``load_video`` catches the decoder broadly; these are the exclusions."""

    def _assert_server_error(self, side_effect):
        with patch(
            "sglang.srt.utils.common.VideoDecoderWrapper", side_effect=side_effect
        ):
            with self.assertRaises(RuntimeError):
                _StubProcessor._load_single_item(b"payload", Modality.VIDEO)

    def test_missing_decoder_backend(self):
        self._assert_server_error(ImportError("no decoder backend installed"))

    def test_decoder_oom(self):
        self._assert_server_error(MemoryError("out of memory"))

    def test_image_source_os_error(self):
        with patch(
            "sglang.srt.utils.common.get_image_bytes",
            side_effect=OSError("too many open files"),
        ):
            with self.assertRaisesRegex(RuntimeError, "server error"):
                _StubProcessor._load_single_item("file:///image.png", Modality.IMAGE)

    def test_pil_system_oserror_stays_server_fault(self):
        image = MagicMock(mode="RGB")
        image.load.side_effect = OSError(errno.EMFILE, "private-fixture")
        with patch(
            "sglang.srt.utils.common.Image.open", return_value=image
        ), self.assertLogs("sglang.srt.utils.common", level="ERROR") as logs:
            with self.assertRaises(RuntimeError) as caught:
                _StubProcessor._load_single_item(b"not-a-jpeg", Modality.IMAGE)
        self.assertEqual(caught.exception.__context__.errno, errno.EMFILE)
        self.assertNotIn(
            "private-fixture", "".join(traceback.format_exception(caught.exception))
        )
        self.assertIn("OSError", "\n".join(logs.output))
        self.assertNotIn("private-fixture", "\n".join(logs.output))


class TestDecodeTimeCorruptionIsClientError(CustomTestCase):
    @staticmethod
    def _png():
        noise = random.Random(0).randbytes(512 * 512 * 3)
        image = Image.frombytes("RGB", (512, 512), noise)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _jpeg():
        pixels = bytes((index * 37) % 256 for index in range(128 * 128 * 3))
        image = Image.frombytes("RGB", (128, 128), pixels)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def test_corrupt_later_png_chunk_is_client_fault(self):
        png = bytearray(self._png())
        offset, idat_offsets = 8, []
        while offset < len(png):
            (length,) = struct.unpack(">I", png[offset : offset + 4])
            if bytes(png[offset + 4 : offset + 8]) == b"IDAT":
                idat_offsets.append(offset)
            offset += 12 + length
        self.assertGreater(len(idat_offsets), 1)
        png[idat_offsets[1] : idat_offsets[1] + 8] = b"\x00" * 8
        with self.assertRaises(ValueError):
            _StubProcessor._load_single_item(bytes(png), Modality.IMAGE)

    def test_truncated_payloads_are_client_faults(self):
        png, jpeg = self._png(), self._jpeg()
        for payload in (png[: len(png) // 2], jpeg[:-2], jpeg[: len(jpeg) // 2]):
            with self.subTest(size=len(payload)):
                with self.assertRaises(ValueError):
                    _StubProcessor._load_single_item(payload, Modality.IMAGE)

    def test_media_errors_do_not_expose_url_query_or_exception_text(self):
        marker = "SYNTHETIC_PRIVATE_MEDIA_MARKER"
        url = f"https://media.example/image?api_key={marker}&prompt={marker}"
        with patch(
            "sglang.srt.utils.common.get_mm_http_session",
            return_value=_session_raising(requests.exceptions.HTTPError(marker)),
        ):
            with self.assertRaises(ValueError) as caught:
                _StubProcessor._load_single_item(url, Modality.IMAGE)
        self.assertNotIn(marker, "".join(traceback.format_exception(caught.exception)))


class TestClientMediaExceptions(CustomTestCase):
    def test_tuple_covers_the_documented_families(self):
        for exc_type in (
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            binascii.Error,  # invalid base64
        ):
            with self.subTest(exc_type=exc_type.__name__):
                self.assertTrue(issubclass(exc_type, CLIENT_MEDIA_EXCEPTIONS))
        self.assertNotIn(OSError, CLIENT_MEDIA_EXCEPTIONS)
        self.assertNotIn(SyntaxError, CLIENT_MEDIA_EXCEPTIONS)


class _LoaderStub(_StubProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


class TestPlaceholderMismatchIsClientError(CustomTestCase):
    def test_more_placeholders_than_images(self):
        """A prompt with more image placeholders than images must be a 400, not a 500."""
        processor = object.__new__(_LoaderStub)
        tokens = MultimodalSpecialTokens(image_token="<img>").build(None)
        with ThreadPoolExecutor(max_workers=1) as processor.io_executor:
            with self.assertRaisesRegex(
                ValueError, "2 image placeholders but 1 image inputs"
            ):
                asyncio.run(
                    processor.legacy_load_mm_data(
                        prompt="a <img> b <img> c",
                        multimodal_tokens=tokens,
                        image_data=[Image.new("RGB", (8, 8))],
                    )
                )


if __name__ == "__main__":
    unittest.main()
