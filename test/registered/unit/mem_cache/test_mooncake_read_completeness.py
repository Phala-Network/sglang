"""Exercise the real Mooncake read methods with deterministic native results.

The AST harness avoids importing GPU allocators; no implementation is mirrored.
Native client compatibility and real KV restoration are separate acceptance gates.
"""

import ast
import time
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def load_store_methods():
    root = Path(__file__).resolve().parents[4]
    source = (
        root / "python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py"
    )
    tree = ast.parse(source.read_text())
    store = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MooncakeStore"
    )
    names = {
        "_normalize_read_results",
        "_batch_io_v2",
        "_batch_postprocess",
        "get",
        "batch_get",
    }
    methods = [
        n for n in store.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    node = ast.ClassDef(
        name="Store", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    ns = dict(
        Sequence=Sequence,
        Any=Any,
        List=List,
        Optional=Optional,
        PoolTransfer=Any,
        time=time,
        l3_page_span=lambda pool: 1,
    )
    exec(compile(module, str(source), "exec"), ns)
    return ns["Store"]


Store = load_store_methods()


class TestReadCompleteness(unittest.TestCase):
    def store(self, results):
        store = Store()
        store.is_mla_backend = True
        store.enable_storage_metrics = False
        store._get_batch_zero_copy_impl = lambda *args: results
        return store

    def test_single_object_requires_all_bytes(self):
        for result in (0, 1, 7, 9, -1):
            with self.subTest(result=result):
                self.assertFalse(self.store([result]).get("key", [100], [8]))
        self.assertTrue(self.store([8]).get("key", [100], [8]))

    def test_single_object_requires_one_result(self):
        for results in ([], [8, 8]):
            with self.subTest(results=results):
                self.assertFalse(self.store(results).get("key", [100], [8]))

    def test_multibuffer_object_uses_total_bytes(self):
        self.assertTrue(self.store([12]).get("key", [100, 200], [8, 4]))
        self.assertFalse(self.store([8]).get("key", [100, 200], [8, 4]))

    def test_prefix_stops_at_incomplete_object(self):
        for results in ([8, 3, 8], [8], [8, -1, 8], [8, 0, 8]):
            with self.subTest(results=results):
                self.assertEqual(
                    self.store(results).batch_get(
                        ["a", "b", "c"], [100, 200, 300], [8, 8, 8]
                    ),
                    1,
                )
        self.assertEqual(
            self.store([8, 8, 8]).batch_get(
                ["a", "b", "c"], [100, 200, 300], [8, 8, 8]
            ),
            3,
        )

    def hybrid(self, results):
        store = self.store(results)
        store.registered_pools = {"kv": [8, 8], "mamba": [[8, 4], [2], [8, 4], [2]]}
        store._tag_keys = lambda keys: keys

        def keys_for(keys, transfer):
            multiplier = 1 if transfer.name == "kv" else 2
            return [
                f"{key}:{component}" for key in keys for component in range(multiplier)
            ], multiplier

        store._get_hybrid_page_component_keys = keys_for
        store._build_pool_buffer_meta = lambda host_pool, **kwargs: (
            [100] * len(host_pool),
            host_pool,
        )
        transfers = [
            SimpleNamespace(name=name, keys=["a", "b"], host_indices=[0, 1])
            for name in store.registered_pools
        ]
        return store, transfers

    def test_hybrid_short_sidecar_cannot_restore_page(self):
        store, transfers = self.hybrid([8, 8, 12, 2, 11, 2])
        self.assertEqual(
            store._batch_io_v2(transfers, False),
            {"kv": [True, True], "mamba": [True, False]},
        )

    def test_hybrid_missing_sidecar_result_keeps_page_count(self):
        store, transfers = self.hybrid([8, 8, 12, 2, 12])
        self.assertEqual(
            store._batch_io_v2(transfers, False),
            {"kv": [True, True], "mamba": [True, False]},
        )

    def test_hybrid_extra_results_fail_closed(self):
        store, transfers = self.hybrid([8, 8, 12, 2, 12, 2, 42])
        self.assertEqual(
            store._batch_io_v2(transfers, False),
            {"kv": [False, False], "mamba": [False, False]},
        )

    def test_complete_hybrid_objects_and_put_status(self):
        store, transfers = self.hybrid([8, 8, 12, 2, 12, 2])
        self.assertEqual(
            store._batch_io_v2(transfers, False),
            {"kv": [True, True], "mamba": [True, True]},
        )
        self.assertEqual(
            store._batch_postprocess([0, 0, 0, -1], True, 2), [True, False]
        )


if __name__ == "__main__":
    unittest.main()
