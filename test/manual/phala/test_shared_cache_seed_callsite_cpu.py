"""Run real D enqueue, hybrid PUT, and whole ACK methods without a GPU."""

import ast
import hashlib
import json
import os
import queue
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"


def load_methods(path, class_name, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    owner.bases = []
    owner.body = [
        item
        for item in owner.body
        if isinstance(item, ast.FunctionDef) and item.name in names
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            owner,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[class_name]()


def load_capture():
    path = SRT / "mem_cache/shared_cache_diagnostics.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom) and node.module == "sglang.srt.environ"
        )
        and not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "shared_cache_diagnostics"
                for target in node.targets
            )
        )
    ]
    namespace = {"__name__": "seed_callsite_cpu_fixture"}
    exec(compile(tree, str(path), "exec"), namespace)
    namespace["shared_cache_diagnostics"] = namespace["SharedCacheDiagnostics"](
        enabled=False, log=Mock()
    )
    return namespace["SharedCacheSeedCapture"](), namespace


class Indices(list):
    def long(self):
        return self

    def dim(self):
        return 1

    def numel(self):
        return len(self)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return Indices(value) if isinstance(key, slice) else value


class Request:
    def __init__(self, **values):
        self.__dict__.update(values)


class StorageOperation:
    next_id = 0

    def __init__(
        self, host_indices, token_ids, *, hash_value, prefix_keys, pool_transfers
    ):
        StorageOperation.next_id += 1
        self.id = StorageOperation.next_id
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.hash_value = hash_value
        self.pool_transfers = pool_transfers
        self.completed_tokens = 0
        self.storage_start = 0
        self.pool_storage_result = NS(extra_pool_hit_pages={})
        self.pool_storage_result.update_extra_pool_hit_pages = (
            self.pool_storage_result.extra_pool_hit_pages.update
        )


def count_pool_hits(results):
    return {
        name: values.index(False) if False in values else len(values)
        for name, values in results.items()
    }


class SharedCacheSeedCallsiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.directory.chmod(0o700)
        self.path = self.directory / "capture.json"
        self.env = patch.dict(
            os.environ, {"SGLANG_SHARED_CACHE_SEED_CAPTURE_CONFIG": str(self.path)}
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.capture, self.capture_namespace = load_capture()
        self.log = Mock()
        self.diagnostics = Mock()
        self.operations = []
        self.native_calls = []
        self.results = [0, 0]
        self.exists = [0, 0]
        self.schema = {"revision": None, "pools": [["pool_a"], ["pool_b"]]}
        namespace = {
            "shared_cache_seed_capture": self.capture,
            "shared_cache_diagnostics": self.diagnostics,
            "logger": self.log,
            "hashlib": hashlib,
            "json": json,
            "os": os,
            "re": re,
            "time": time,
            "StorageOperation": StorageOperation,
            "HiCacheStorageExtraInfo": NS,
            "count_pool_hits": count_pool_hits,
            "DEFAULT_TENANT_ID": "default",
        }
        self.store = load_methods(
            SRT / "mem_cache/storage/mooncake_store/mooncake_store.py",
            "MooncakeStore",
            {"_batch_io_v2", "batch_set_v2", "_filter_group_ids"},
            namespace,
        )
        self.store.config = NS(tenant_id="default")
        self.store.shared_cache_store_instance_id = "actual-store-incarnation"
        self.store.registered_pools = {
            name: NS(page_size=2, get_page_buffer_meta=lambda _indices: ([10], [8]))
            for name in ("pool_a", "pool_b")
        }
        self.store._tag_keys = lambda keys: ["actual-tag_" + key for key in keys]
        self.store._get_hybrid_page_component_keys = lambda keys, transfer: (
            [key + "_" + transfer.name for key in keys],
            1,
        )
        self.store._can_use_group_semantics = lambda: False
        self.store._batch_exist = lambda _keys: self.exists[:1]
        self.store._batch_postprocess = lambda results, **_kwargs: [
            value == 0 for value in results
        ]

        def native_put(keys, _ptrs, _sizes, _groups):
            # This is reached from the actual store method during queue.put.
            if self.capture._config is not None and not self.capture._sealed_path:
                self.assertIsNotNone(self.capture._operation_id)
            self.native_calls.append(list(keys))
            return [self.results[len(self.native_calls) % 2 - 1]]

        self.store._put_batch_zero_copy_impl = native_put
        self.controller = load_methods(
            SRT / "mem_cache/hybrid_cache/hybrid_cache_controller.py",
            "HybridCacheController",
            {"write_storage", "_page_backup"},
            namespace,
        )
        self.controller.page_size = 2
        self.controller.storage_config = NS(
            tp_size=1,
            pp_size=1,
            tp_rank=0,
            extra_config={"extra_backend_tag": "actual-tag"},
        )
        self.controller.storage_backend = self.store
        self.controller.backup_skip = True
        self.controller.should_backup = lambda _transfer: True
        self.controller._resolve_sidecar_kv_derived_pool_transfers = lambda _op: None
        self.controller._resolve_sidecar_nonkv_derived_pool_transfers = lambda _op: None
        self.controller.ack_backup_queue = queue.Queue()

        def consume_immediately(operation):
            self.operations.append(operation)
            if operation.shared_cache_seed_selected:
                self.assertEqual(self.capture._operation_id, operation.id)
            self.controller._page_backup(operation)
            self.controller.ack_backup_queue.put(operation)

        self.controller.backup_queue = NS(put=consume_immediately)
        self.manager = load_methods(
            SRT / "disaggregation/decode_kvcache_offload_manager.py",
            "DecodeKVCacheOffloadManager",
            {
                "_arm_seed_operation",
                "_trigger_backup",
                "_check_backup_progress",
                "_dsv4_backup_complete",
                "offload_kv_cache",
                "_check_offload_progress",
                "_prefill_offloaded_len",
                "_mark_offload_started",
                "_mark_offload_finished",
                "_has_inflight_offload",
            },
            namespace,
        )
        self.manager.is_dsv4 = True
        self.manager.cache_controller = self.controller
        self.manager.decode_host_mem_pool = NS(
            storage_schema=self.schema, free=Mock(), release_transfers=Mock()
        )
        self.manager.shared_cache_d_worker_id = "actual-D-incarnation"
        self.manager.page_size = 2
        self.manager.offload_stride = 2
        self.manager.ongoing_backup = {}
        self.manager.backup_extra_pools = {}
        self.manager._compute_prefix_hash = lambda _req, _tokens, prior=None: [
            "actual-page-hash" + str(prior)
        ]
        self.req = Request(
            rid="random-internal-rid", pd_diagnostic_request_ref="a" * 64
        )
        self.config = {
            "schema": "phala.shared-cache.seed-config.v1",
            "run_id": "run",
            "seed_id": "seed",
            "case_id": "case",
            "epoch": "epoch",
            "request_ref": self.req.pd_diagnostic_request_ref,
            "tenant_id": "default",
            "backend_tag": "actual-tag",
            "model_revision": None,
            "kv_schema": hashlib.sha256(
                json.dumps(self.schema, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "page_range": {"start": 8, "end": 9},
            "rank": 0,
            "key_salt": "offline-fixture-salt",
            "output_dir": str(self.directory),
            "required_components": ["pool_a:0", "pool_b:0"],
            "max_keys": 8,
            "max_logical_bytes": 128,
            "max_events": 8,
            "max_duration_ms": 30000,
            "max_artifact_bytes": 65536,
        }
        self.write_config()

    def write_config(self):
        self.path.write_text(json.dumps(self.config), encoding="utf-8")
        self.path.chmod(0o600)

    def transfers(self):
        return [
            NS(name=name, host_indices=Indices([0, 1]), keys=None)
            for name in ("pool_a", "pool_b")
        ]

    def submit(self, page_start=8, prior=None):
        self.manager._trigger_backup(
            self.req,
            Indices([0, 1]),
            [1, 2],
            time.time(),
            prior,
            self.transfers(),
            page_start=page_start,
        )

    def ack(self):
        self.manager._check_backup_progress(1)

    def test_selects_exact_second_operation_and_binds_before_immediate_put(self):
        self.submit(page_start=7, prior="first")
        self.assertIsNone(self.capture._config)
        self.assertFalse(self.operations[0].shared_cache_seed_selected)
        self.ack()
        self.submit(page_start=8, prior="second")
        self.assertTrue(self.operations[1].shared_cache_seed_selected)
        self.assertEqual(self.capture._operation_id, self.operations[1].id)
        self.assertEqual({x["page_index"] for x in self.capture._entries.values()}, {8})
        self.ack()
        sealed = self.directory / "seed.seed.json"
        self.assertTrue(sealed.is_file())
        document = json.loads(sealed.read_text())
        self.assertEqual(document["request_id"], self.req.rid)
        self.assertEqual(
            document["store_instance_id"], self.store.shared_cache_store_instance_id
        )
        self.assertEqual(document["d_worker_id"], self.manager.shared_cache_d_worker_id)
        original = sealed.read_bytes()
        entries = dict(self.capture._entries)
        self.submit(page_start=9, prior="third")
        self.ack()
        self.assertFalse(self.operations[2].shared_cache_seed_selected)
        self.assertEqual(sealed.read_bytes(), original)
        self.assertEqual(self.capture._entries, entries)
        self.assertIsNone(self.capture._failed)
        self.assertEqual(len(self.native_calls), 6)

    def test_missing_or_wrong_server_ref_cannot_arm(self):
        for value in (None, "forged", "b" * 64):
            self.req.pd_diagnostic_request_ref = value
            self.submit()
            self.ack()
            self.assertIsNone(self.capture._config)
        self.assertEqual(len(self.native_calls), 6)

    def test_actual_runtime_selectors_must_match_private_config(self):
        for field, wrong in (
            ("backend_tag", "other-tag"),
            ("model_revision", "other-revision"),
            ("kv_schema", "b" * 64),
            ("tenant_id", "other-tenant"),
            ("required_components", ["pool_a:0"]),
        ):
            with self.subTest(field=field):
                original = self.config[field]
                self.config[field] = wrong
                self.write_config()
                self.submit()
                self.ack()
                self.assertIsNone(self.capture._config)
                self.config[field] = original

    def test_incomplete_pool_put_cannot_seal_whole_group_ack(self):
        self.results = [0, -1]
        self.submit()
        self.assertFalse(self.manager._dsv4_backup_complete(self.operations[0]))
        self.ack()
        self.assertFalse((self.directory / "seed.seed.json").exists())

    def test_bind_failure_does_not_retry_or_skip_real_enqueue(self):
        with patch.object(self.capture, "bind_operation", side_effect=RuntimeError):
            self.submit()
        self.assertFalse(self.operations[0].shared_cache_seed_selected)
        self.assertEqual(len(self.operations), 1)
        self.assertEqual(len(self.native_calls), 2)
        self.ack()
        self.assertFalse((self.directory / "seed.seed.json").exists())

    def test_arm_failure_does_not_retry_real_put(self):
        with patch.object(self.capture, "arm", side_effect=RuntimeError):
            self.submit()
        self.assertEqual(len(self.operations), 1)
        self.assertEqual(len(self.native_calls), 2)
        self.assertFalse(self.operations[0].shared_cache_seed_selected)

    def test_unregistered_pool_or_multirank_cannot_make_partial_manifest(self):
        self.store.registered_pools["missing"] = NS()
        self.submit()
        self.ack()
        self.assertIsNone(self.capture._config)
        del self.store.registered_pools["missing"]
        self.controller.storage_config.tp_size = 2
        self.submit()
        self.ack()
        self.assertIsNone(self.capture._config)

    def test_offload_submission_saves_page_range_before_request_frontier_moves(self):
        self.manager.req_to_token_pool = NS(req_to_token={0: Indices(range(64))})
        self.req.kv = NS(req_pool_idx=0)
        self.req.origin_input_ids = list(range(16))
        self.req.output_ids = [20, 21, 22]
        self.req.finished = lambda: False
        self.manager.request_counter = 0
        self.manager.offload_inflight = {}
        self.manager.offloaded_state = {self.req: NS(last_hash="prior", inc_len=0)}
        self.manager.ongoing_offload = {}
        self.manager.offload_page_starts = {}
        self.manager.offload_extra_pools = {}
        self.manager._dsv4_device_transfers = lambda _indices: self.transfers()
        self.manager._all_ranks_ready = lambda ready: ready
        event = NS(synchronize=Mock())
        self.controller.ack_write_queue = []

        def d2h_write(*, node_id, **_kwargs):
            self.controller.ack_write_queue.append(
                NS(node_ids=[node_id], finish_event=event)
            )
            return Indices([0, 1])

        self.controller.write = d2h_write
        self.assertTrue(self.manager.offload_kv_cache(self.req))
        self.assertEqual(self.manager.offload_page_starts, {1: 8})
        self.req.output_ids.extend(range(32))
        self.manager._check_offload_progress(1)
        self.assertTrue(self.operations[0].shared_cache_seed_selected)
        self.assertEqual(self.operations[0].shared_cache_diag_page_start, 8)
        self.assertEqual(self.manager.offload_page_starts, {})
        self.ack()
        self.assertTrue((self.directory / "seed.seed.json").exists())


if __name__ == "__main__":
    unittest.main()
