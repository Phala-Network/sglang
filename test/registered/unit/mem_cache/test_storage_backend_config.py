"""CPU tests for the shared HiCache storage config loader."""

import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.mem_cache.storage_backend_config import (
    StorageBackendConfigError,
    load_storage_backend_extra_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestStorageBackendConfig(unittest.TestCase):
    def test_empty_and_inline_json_require_mapping(self):
        self.assertEqual(load_storage_backend_extra_config(None), {})
        self.assertEqual(load_storage_backend_extra_config(""), {})
        self.assertEqual(
            load_storage_backend_extra_config('{"tenant":"fixture","tag":7}'),
            {"tenant": "fixture", "tag": 7},
        )
        for value in ("[]", "null", '"secret-value"'):
            with self.assertRaises(StorageBackendConfigError):
                load_storage_backend_extra_config(value)

    def test_json_toml_and_yaml_files_match_inline_object(self):
        expected = {"tenant": "fixture", "tag": 7, "allocator": "shm"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "config.json"
            json_path.write_text(json.dumps(expected), encoding="utf-8")
            self.assertEqual(
                load_storage_backend_extra_config(f"@{json_path}"), expected
            )

            toml_path = root / "config.toml"
            toml_path.write_text(
                'tenant = "fixture"\ntag = 7\nallocator = "shm"\n',
                encoding="utf-8",
            )
            self.assertEqual(
                load_storage_backend_extra_config(f"@{toml_path}"), expected
            )

            try:
                import yaml  # noqa: F401
            except ImportError:
                self.skipTest("PyYAML is not installed")
            yaml_path = root / "config.yaml"
            yaml_path.write_text(
                "tenant: fixture\ntag: 7\nallocator: shm\n", encoding="utf-8"
            )
            self.assertEqual(
                load_storage_backend_extra_config(f"@{yaml_path}"), expected
            )

    def test_errors_do_not_include_config_values(self):
        secret = "fixture-secret-that-must-not-leak"
        with self.assertRaises(StorageBackendConfigError) as context:
            load_storage_backend_extra_config("{" + secret)
        self.assertNotIn(secret, str(context.exception))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("{" + secret, encoding="utf-8")
            with self.assertRaises(StorageBackendConfigError) as context:
                load_storage_backend_extra_config(f"@{path}")
        self.assertNotIn(secret, str(context.exception))

    def test_at_file_preserves_allocator_for_dynamic_host_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allocator.json"
            path.write_text('{"allocator":"shm"}', encoding="utf-8")
            config = load_storage_backend_extra_config(f"@{path}")
        self.assertEqual(config["allocator"], "shm")


if __name__ == "__main__":
    unittest.main()
