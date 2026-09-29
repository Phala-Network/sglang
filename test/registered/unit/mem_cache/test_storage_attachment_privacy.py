"""Runtime storage attach errors must not echo configuration input."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache


class TestStorageAttachmentPrivacy(unittest.TestCase):
    def test_parse_error_does_not_echo_sentinel_secret(self):
        secret = "SENTINEL_STORAGE_CONFIG_SECRET"
        cache = SimpleNamespace(cache_controller=object(), enable_storage=False)
        attachment = StorageAttachment(cache)
        with patch.object(attachment, "_apply_policies"), patch(
            "sglang.srt.mem_cache.unified_cache.storage_attachment.HybridCacheController.parse_storage_backend_extra_config",
            side_effect=ValueError(secret),
        ):
            ok, message = attachment.attach(
                "mooncake", storage_backend_extra_config_json=secret
            )
        self.assertFalse(ok)
        self.assertIn("ValueError", message)
        self.assertNotIn(secret, message)

    def test_hiradix_parse_error_does_not_echo_sentinel_secret(self):
        secret = "SENTINEL_HIRADIX_STORAGE_CONFIG_SECRET"
        cache = object.__new__(HiRadixCache)
        cache.enable_storage = False
        with patch.object(
            cache,
            "_parse_storage_backend_extra_config",
            side_effect=ValueError(secret),
        ):
            ok, message = cache.attach_storage_backend(
                "mooncake", storage_backend_extra_config_json=secret
            )
        self.assertFalse(ok)
        self.assertIn("ValueError", message)
        self.assertNotIn(secret, message)


if __name__ == "__main__":
    unittest.main()
