"""Real producer -> queued controller -> component GET -> diagnostic CPU checks.

Windows lacks POSIX modes and directory fsync: only that filesystem adapter is
simulated there. Linux runs the actual private-file and seal implementation.
"""

import dataclasses
import hashlib
import json
import os
import queue
import stat
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from test_shared_cache_seed_callsite_cpu import (
    SRT,
    Indices,
    SeedCallsiteFixture,
    StorageOperation,
    count_pool_hits,
    load_methods,
)


class WindowsPosixFixture:
    """Metadata/fsync adapter only; all content, descriptors and inode checks real."""

    def __getattr__(self, name):
        return getattr(os, name)

    def lstat(self, path):
        value = os.lstat(path)
        return self._metadata(value)

    def fstat(self, fd):
        return self._metadata(os.fstat(fd))

    @staticmethod
    def _metadata(value):
        fields = {k: getattr(value, k) for k in dir(value) if k.startswith("st_")}
        fields["st_mode"] = stat.S_IFMT(value.st_mode) | (
            0o700 if stat.S_ISDIR(value.st_mode) else 0o600
        )
        return NS(**fields)

    def open(self, path, flags, *args):
        if os.path.isdir(path) and flags == os.O_RDONLY:
            return os.open(os.devnull, os.O_RDONLY)
        return os.open(path, flags, *args)

    def fsync(self, fd):
        pass


@dataclasses.dataclass(frozen=True)
class Handle:
    rid: str
    attempt_id: int
    bootstrap_room: int
    pd_diagnostic_request_ref: str = "b" * 64


class ReaderLateArmTests(SeedCallsiteFixture):
    def setUp(self):
        super().setUp()
        if os.name == "nt":
            self.capture_namespace["os"] = WindowsPosixFixture()
        names = getattr(self, "fixture_pool_names", ("pool_a", "pool_b"))
        self.store.registered_pools = {
            name: NS(page_size=2, get_page_buffer_meta=lambda indices: ([10], [8]))
            for name in names
        }
        self.transfers = lambda: [
            NS(name=name, host_indices=Indices([0, 1]), keys=None) for name in names
        ]
        self.config["required_components"] = sorted(f"{name}:0" for name in names)
        self.config["max_keys"] = len(names) * 2
        self.config["max_events"] = 128
        self.schema["pools"] = [
            [name, 2, "page_first", 1, "torch.float32", 8, None] for name in names
        ]
        self.config["kv_schema"] = hashlib.sha256(
            json.dumps(self.schema, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.config.update(
            schema="phala.shared-cache.seed-config.v2",
            page_range={"start": 2, "end": 4},
            operation_ranges=[{"start": 2, "end": 3}, {"start": 3, "end": 4}],
        )
        self.write_config()
        StorageOperation.next_id = 0
        self.submit(page_start=2, prior="first")
        self.submit(page_start=3, prior="second")
        self.ack()
        self.ack()
        self.seed = self.directory / "seed.seed.json"
        self.assertTrue(self.seed.exists())
        self.document = json.loads(self.seed.read_text())
        self.assertEqual(
            [op["operation_id"] for op in self.document["operations"]], [0, 1]
        )
        self.reader_log = Mock()
        self.reader = self.capture_namespace["SharedCacheDiagnostics"](
            reader_manifest=str(self.seed), log=self.reader_log
        )
        self.namespace = {
            "shared_cache_diagnostics": self.reader,
            "shared_cache_seed_capture": self.capture,
            "HiCacheStorageExtraInfo": NS,
            "PoolTransfer": lambda **kw: NS(indices_from_pool=None, **kw),
            "PoolName": NS(KV="kv"),
            "PrefetchAck": NS,
            "STORAGE_BATCH_SIZE": 1,
            "count_pool_hits": count_pool_hits,
            "DEFAULT_TENANT_ID": "default",
            "logger": Mock(),
        }
        store = load_methods(
            SRT / "mem_cache/storage/mooncake_store/mooncake_store.py",
            "MooncakeStore",
            {"_batch_io_v2", "batch_get_v2"},
            self.namespace,
        )
        store.__dict__.update(self.store.__dict__)
        self.store = store
        self.store._batch_postprocess = lambda values, **kwargs: [
            x == 1 for x in values
        ]
        self.store._get_batch_zero_copy_impl = lambda keys, ptrs, sizes: list(sizes)
        base = load_methods(
            SRT / "managers/cache_controller.py",
            "HiCacheController",
            {"_page_transfer", "_page_transfer_kv_batch"},
            self.namespace,
        )
        hybrid = load_methods(
            SRT / "mem_cache/hybrid_cache/hybrid_cache_controller.py",
            "HybridCacheController",
            {"_prefetch_extra_info", "_page_transfer_sidecar"},
            self.namespace,
        )
        self.controller = type("ReaderController", (type(hybrid), type(base)), {})()
        self.controller.page_size = 2
        self.controller.prefetch_sync_queue = queue.Queue()
        self.controller.storage_backend = store
        self.controller.storage_config = NS(
            tp_size=1,
            pp_size=1,
            tp_rank=0,
            extra_config={"extra_backend_tag": "actual-tag"},
        )
        self.controller.mem_pool_host = NS(storage_schema=self.schema)
        # DSV4 has a logical KV anchor; actual payload lives in side pools.
        self.controller.page_get_func = lambda op, keys, indices, extra: len(keys)
        self.controller._sync_trailing_keys = lambda *args: None
        self.controller._resolve_sidecar_nonkv_derived_pool_transfers = lambda op: None

    def operation(self, rid="reader-a", room=71, ident=7, page="first", attempt=0):
        operation = NS(
            handle=Handle(
                rid, attempt, room, hashlib.sha256(f"{rid}:{room}".encode()).hexdigest()
            ),
            request_id=rid,
            id=ident,
            hash_value=["actual-page-hash" + page],
            prefix_keys=None,
            host_indices=Indices([0, 1]),
            terminated=False,
            sidecar_hash_values=None,
            sidecar_hit_pages=1,
        )
        operation.is_terminated = lambda: operation.terminated
        operation.pool_transfers = [
            NS(
                name=name,
                indices_from_pool="kv",
                host_indices=Indices([0, 1]),
                keys=operation.hash_value,
            )
            for name in self.store.registered_pools
        ]
        return operation

    def events(self, event=None):
        records = [json.loads(c.args[1]) for c in self.reader_log.info.call_args_list]
        return [r for r in records if event is None or r["event"] == event]

    def rewrite(self, **changes):
        document = dict(self.document, **changes)
        document.pop("manifest_sha256", None)
        document["manifest_sha256"] = hashlib.sha256(
            json.dumps(
                document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        self.seed.write_text(json.dumps(document))

    def test_long_startup_then_arm_full_union_and_one_shot_expiry(self):
        self.reader._started_at -= 3600
        self.assertEqual(self.controller._page_transfer(self.operation()), 1)
        self.assertTrue(self.reader.enabled)
        self.assertEqual(len(self.reader._key_ids), len(self.document["keys"]))
        self.assertEqual(self.reader._max_duration_s, 120)
        started = self.reader._started_at
        self.seed.write_text("invalid replacement")
        self.controller._page_transfer(self.operation("reader-b", 72, 8, "second"))
        self.assertEqual(len(self.events("get_components")), 4)
        self.assertEqual(len(self.events("reader_capture_armed")), 1)
        self.reader._started_at -= 121
        self.controller._page_transfer(self.operation("reader-c", 73, 9))
        self.assertEqual(len(self.events("get_components")), 4)
        self.assertEqual(self.reader._started_at, started - 121)
        self.assertTrue(self.reader._truncation_logged)

    def test_no_path_is_off_without_open_or_schema_access(self):
        self.reader._reader_manifest = None
        with patch.object(
            self.capture_namespace["os"], "lstat", side_effect=AssertionError
        ):
            self.controller._page_transfer(self.operation())
        self.assertFalse(self.reader.enabled)
        self.assertFalse(self.reader._reader_attempted)
        self.assertEqual(self.events(), [])

    def test_invalid_file_one_attempt_and_business_reads_continue(self):
        self.seed.write_text("{")
        self.assertEqual(self.controller._page_transfer(self.operation()), 1)
        self.rewrite()
        self.assertEqual(self.controller._page_transfer(self.operation("b", 3, 8)), 1)
        self.assertFalse(self.reader.enabled)
        self.assertEqual(self.events(), [])

    def test_donor_before_seed_then_published_seed_arms_only_new_reader(self):
        sealed_bytes = self.seed.read_bytes()
        self.seed.unlink()
        donor = self.operation("donor", 70, 6)
        self.assertEqual(self.controller._page_transfer(donor), 1)
        self.assertFalse(self.reader._reader_attempted)
        self.assertFalse(self.reader.enabled)
        self.seed.write_bytes(sealed_bytes)
        self.seed.chmod(0o600)
        self.controller._page_transfer(self.operation())
        self.assertTrue(self.reader.enabled)
        self.assertEqual(len(self.events("reader_capture_armed")), 1)
        self.assertEqual(len(self.events("get_components")), 2)
        # An old donor's later batch must not inherit the new reader's context.
        self.controller._page_transfer(donor)
        self.assertIsNone(donor.shared_cache_reader_context)
        self.assertEqual(len(self.events("get_components")), 2)

    def test_expired_manifest_rejected(self):
        self.rewrite(expires_at=time.time() - 1)
        self.controller._page_transfer(self.operation())
        self.assertFalse(self.reader.enabled)

    def test_foreign_schema_rejected(self):
        self.rewrite(kv_schema="a" * 64)
        self.controller._page_transfer(self.operation())
        self.assertFalse(self.reader.enabled)

    def test_foreign_component_union_rejected(self):
        self.controller.storage_backend.registered_pools["extra_pool"] = NS(
            page_size=2, get_page_buffer_meta=lambda indices: ([10], [8])
        )
        self.controller._page_transfer(self.operation())
        self.assertFalse(self.reader.enabled)

    def test_symlink_or_wrong_mode_rejected(self):
        real_lstat = self.capture_namespace["os"].lstat
        metadata = real_lstat(self.seed)
        fake = NS(st_mode=stat.S_IFLNK | 0o600, st_size=metadata.st_size)
        with patch.object(self.capture_namespace["os"], "lstat", return_value=fake):
            self.controller._page_transfer(self.operation())
        self.assertFalse(self.reader.enabled)

    def test_two_async_operations_keep_rooms_refs_keys_and_actual_schema(self):
        barrier = threading.Barrier(2)

        def native(keys, ptrs, sizes):
            barrier.wait(timeout=5)
            return sizes

        self.store._get_batch_zero_copy_impl = native
        first = self.operation()
        second = self.operation("reader-b", 72, 8, "second")
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(self.controller._page_transfer, [first, second]))
        self.assertEqual(results, [1, 1])
        events = self.events("get_components")
        self.assertEqual(len(events), 4)
        for op in (first, second):
            selected = [r for r in events if r["room"] == op.handle.bootstrap_room]
            self.assertEqual(len(selected), 2)
            expected = {
                self.reader.key_id("actual-tag_" + op.hash_value[0] + "_" + pool)
                for pool in self.store.registered_pools
            }
            self.assertEqual(
                {c["key_id"] for r in selected for c in r["components"]}, expected
            )
            self.assertEqual(
                {r["request_id"] for r in selected},
                {self.reader._request_id(op.handle.rid)},
            )
            self.assertEqual(
                {r["reader_request_ref"] for r in selected},
                {op.handle.pd_diagnostic_request_ref},
            )
            self.assertEqual(
                {r["kv_schema"] for r in selected}, {self.document["kv_schema"]}
            )
            self.assertEqual(
                {r["seed_source_request_ref"] for r in selected},
                {self.document["request_ref"]},
            )
        raw = json.dumps(events)
        for secret in (
            "reader-a",
            "reader-b",
            "actual-page-hash",
            self.config["key_salt"],
        ):
            self.assertNotIn(secret, raw)

    def test_short_failure_cancel_and_reused_rid_remain_distinct(self):
        op = self.operation()

        def native(keys, ptrs, sizes):
            op.terminated = True
            return [3] if "pool_a" in keys[0] else [-1]

        self.store._get_batch_zero_copy_impl = native
        self.assertEqual(self.controller._page_transfer(op), 0)
        old = self.events("get_components")
        self.assertTrue(all(r["reader_cancelled"] for r in old))
        self.assertTrue(
            all(not c["read_complete"] for r in old for c in r["components"])
        )
        self.store._get_batch_zero_copy_impl = lambda keys, ptrs, sizes: sizes
        self.controller._page_transfer(self.operation(room=72, ident=8, attempt=1))
        new = self.events("get_components")[2:]
        self.assertTrue(all(not r["reader_cancelled"] for r in new))
        self.assertNotEqual(old[0]["attempt_id"], new[0]["attempt_id"])
        self.assertNotEqual(old[0]["operation_id"], new[0]["operation_id"])

    def test_cancel_before_io_never_arms_and_sidecar_uses_same_context(self):
        cancelled = self.operation()
        cancelled.terminated = True
        self.assertEqual(self.controller._page_transfer(cancelled), 0)
        self.assertFalse(self.reader._reader_attempted)
        op = self.operation()
        self.controller._page_transfer(op)
        for transfer in op.pool_transfers:
            transfer.indices_from_pool = None
        self.controller._page_transfer_sidecar(op, 1)
        events = self.events("get_components")
        self.assertEqual(len(events), 4)
        self.assertEqual(len({r["operation_id"] for r in events}), 1)

    def test_full_six_component_twelve_key_union_and_bounded_fourteen_readers(self):
        fixture = ReaderLateArmTests()
        fixture.fixture_pool_names = (
            "deepseek_v4_c128",
            "deepseek_v4_c4",
            "deepseek_v4_c4_indexer",
            "deepseek_v4_c4_indexer_state",
            "deepseek_v4_c4_state",
            "swa",
        )
        fixture.setUp()
        try:
            for index in range(14):
                for page in ("first", "second"):
                    fixture.controller._page_transfer(
                        fixture.operation(
                            f"reader-{index}",
                            70 + index,
                            7 + index * 2 + (page == "second"),
                            page,
                        )
                    )
            events = fixture.events("get_components")
            self.assertEqual(len(events), 14 * 12)
            self.assertEqual(len(fixture.reader._key_ids), 12)
            for room in range(70, 84):
                self.assertEqual(
                    {
                        c["key_id"]
                        for r in events
                        if r["room"] == room
                        for c in r["components"]
                    },
                    fixture.reader._key_ids,
                )
            armed = fixture.events("reader_capture_armed")[0]
            self.assertEqual(armed["storage_schema"], fixture.schema)
            self.assertEqual(armed["max_key_observations"], 12 * 32 * 4)
            self.assertFalse(fixture.reader._truncation_logged)
        finally:
            fixture.doCleanups()

    def test_request_budget_stops_once_without_rearm(self):
        for index in range(33):
            self.controller._page_transfer(
                self.operation(f"reader-{index}", index, index)
            )
        self.assertEqual(len(self.reader._requests), 32)
        self.assertTrue(self.reader._truncation_logged)
        self.assertEqual(len(self.events("get_components")), 64)
        self.assertEqual(len(self.events("reader_capture_armed")), 1)

    def test_bad_digest_and_component_hmac_rejected(self):
        document = dict(self.document)
        document["keys"][0]["key"] = "tampered"
        self.seed.write_text(json.dumps(document))
        self.controller._page_transfer(self.operation())
        self.assertFalse(self.reader.enabled)

    def test_prefetch_c128_compatibility_and_trusted_ref_join(self):
        operation = self.operation()
        self.controller._page_transfer(operation)
        self.reader.record_prefetch(
            request_id=operation.handle.rid,
            requested_tokens=2,
            completed_tokens=2,
            accepted=True,
            reader_context=operation.shared_cache_reader_context,
        )
        self.reader.record_c128_transfer(
            request_id=operation.handle.rid,
            room=operation.handle.bootstrap_room,
            index_count=1,
            online=True,
            sender_mode="fake",
            completed=True,
            reader_request_ref=operation.handle.pd_diagnostic_request_ref,
        )
        selected = (
            self.events("get_components")
            + self.events("prefetch_boundary")
            + self.events("c128_transfer")
        )
        self.assertEqual(len({r["request_id"] for r in selected}), 1)
        self.assertEqual(len({r["reader_request_ref"] for r in selected}), 1)


if __name__ == "__main__":
    unittest.main()
