"""Dynamic host allocator selection uses the shared config loader."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.pool_host import common


class TestPoolHostAllocatorConfig(unittest.TestCase):
    def test_dynamic_at_file_allocator_selects_shm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allocator.json"
            path.write_text('{"allocator":"shm"}', encoding="utf-8")
            memory = SimpleNamespace(
                hicache_storage_backend="dynamic",
                hicache_storage_backend_extra_config=f"@{path}",
            )
            with patch.object(common, "get_memory", return_value=memory):
                self.assertEqual(common.get_allocator_type(), "shm")


if __name__ == "__main__":
    unittest.main()
