"""Check removed test protocols and retained native I/O behavior without GPUs."""

import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"


def method(path, owner, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    node = next(n for n in cls.body if getattr(n, "name", None) == name)
    node.decorator_list = []
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                node,
            ],
            type_ignores=[],
        )
    )
    scope = {}
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


class Tests(unittest.TestCase):
    def test_removed_protocol_has_no_runtime_import_route_env_or_native_call(self):
        removed = re.compile(
            r"shared_cache_(?:diagnostics|seed|reader|control)|pd_diagnostic|"
            r"cold_shared_read|SGLANG_MOONCAKE_PD_TRANSFER_DIAGNOSTICS|"
            r"SGLANG_PD_BATCH_DIAGNOSTICS|batch_memory_replica_clear|"
            r"batch_transfer_sync_diagnostic|diagnostic_snapshot|"
            r"/shared-cache/(?:memory|snapshot)|SharedCacheClearMemory"
        )
        roots = [
            (SRT, "*.py"),
            (ROOT / "sgl-model-gateway/src", "*.rs"),
            (ROOT / "sgl-model-gateway/vendor/openai-protocol-1.0.0/src", "*.rs"),
        ]
        for root, glob in roots:
            for path in root.rglob(glob):
                self.assertIsNone(
                    removed.search(path.read_text(encoding="utf-8")), str(path)
                )

    def test_component_get_requires_exact_complete_bytes(self):
        path = SRT / "mem_cache/storage/mooncake_store/mooncake_store.py"
        run = method(path, "MooncakeStore", "_batch_io_v2")
        post = method(path, "MooncakeStore", "_batch_postprocess")
        for sizes, returned, expected in (
            ([8, 8], [8, 8], [True, True]),
            ([8, 8], [7, 8], [False, True]),
            ([[4, 8], [4, 8]], [12, 11], [True, False]),
            ([8, 8], [-1, 0], [False, False]),
        ):
            with self.subTest(sizes=sizes, returned=returned):
                pool = SimpleNamespace(
                    page_size=1, get_page_buffer_meta=lambda _: ([100, 200], sizes)
                )
                store = SimpleNamespace(
                    registered_pools={"component": pool},
                    _tag_keys=lambda keys: keys,
                    _get_hybrid_page_component_keys=lambda keys, _: (keys, 1),
                    _get_batch_zero_copy_impl=lambda *args: returned,
                )
                store._batch_postprocess = lambda *args, **kw: post(store, *args, **kw)
                transfer = SimpleNamespace(
                    name="component", keys=["a", "b"], host_indices=[0, 1]
                )
                self.assertEqual(run(store, [transfer], False)["component"], expected)

    def test_native_get_exception_and_transfer_error_propagate(self):
        get = method(
            SRT / "mem_cache/storage/mooncake_store/mooncake_store.py",
            "MooncakeStore",
            "_get_batch_zero_copy_impl",
        )
        backend = SimpleNamespace(batch_get_into=Mock(side_effect=RuntimeError("read")))
        store = SimpleNamespace(store=backend, _uses_multi_buffer=lambda _: False)
        with self.assertRaisesRegex(RuntimeError, "read"):
            get(store, ["key"], [100], [8])
        transfer = method(
            SRT / "disaggregation/mooncake/conn.py",
            "MooncakeKVManager",
            "_transfer_data",
        )
        engine = SimpleNamespace(batch_transfer_sync=Mock(return_value=-1))
        manager = SimpleNamespace(engine=engine)
        self.assertEqual(transfer(manager, "peer", []), 0)
        engine.batch_transfer_sync.assert_not_called()
        self.assertEqual(transfer(manager, "peer", [(100, 200, 8)]), -1)
        engine.batch_transfer_sync.assert_called_once_with("peer", [100], [200], [8])


if __name__ == "__main__":
    unittest.main()
