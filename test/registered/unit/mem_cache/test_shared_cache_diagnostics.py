import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    deepseek_v4_storage_schema,
)
from sglang.srt.mem_cache.shared_cache_diagnostics import SharedCacheDiagnostics
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class CaptureLog:
    def __init__(self):
        self.records = []

    def info(self, _, message):
        self.records.append(message)

    def warning(self, _, message):
        self.records.append(message)


class TestSharedCacheDiagnostics(unittest.TestCase):
    def make_capture(self, **kwargs):
        log = CaptureLog()
        key_salt = kwargs.pop("key_salt", "test-only-shared-salt")
        if key_salt is None:
            return (
                SharedCacheDiagnostics(
                    enabled=True,
                    key_salt=None,
                    case_id="unit-case",
                    epoch="unit-epoch",
                    key_ids="0" * 64,
                    log=log,
                    **kwargs,
                ),
                log,
            )
        case_id = kwargs.pop("case_id", "unit-case")
        epoch = kwargs.pop("epoch", "unit-epoch")
        allowed_keys = kwargs.pop(
            "allowed_keys",
            ["private-put-key-1", "private-put-key-2", "key-1", "key-2", "key-3"],
        )
        bootstrap = SharedCacheDiagnostics(
            enabled=True,
            key_salt=key_salt,
            case_id=case_id,
            epoch=epoch,
            key_ids="bootstrap",
        )
        key_ids = ",".join(bootstrap.key_id(key) for key in allowed_keys) or "0" * 64
        capture = SharedCacheDiagnostics(
            enabled=True,
            key_salt=key_salt,
            case_id=case_id,
            epoch=epoch,
            key_ids=key_ids,
            log=log,
            **kwargs,
        )
        return capture, log

    def events(self, log):
        return [json.loads(record) for record in log.records]

    def test_default_off_does_not_hash_or_emit(self):
        capture = SharedCacheDiagnostics(enabled=False, key_salt="unused")
        with patch(
            "sglang.srt.mem_cache.shared_cache_diagnostics.hmac.new",
            side_effect=AssertionError("disabled capture hashed a key"),
        ):
            capture.record_io(
                is_set=False,
                pool="KV",
                keys=["private-object-key"],
                sizes=[8],
                results=[8],
            )

    def test_missing_salt_keeps_capture_off(self):
        capture, log = self.make_capture(key_salt=None)
        self.assertFalse(capture.enabled)
        capture.record_schema({"pools": [["KV", 1, "layout", 1, "uint8", 4]]})
        self.assertEqual(log.records, [])

    def test_schema_reports_registered_pool_geometry_only(self):
        capture, log = self.make_capture()
        capture.record_schema(
            {
                "revision": "private-model-revision",
                "pools": [
                    ["KV", 256, "page_first", 8, "torch.uint8", 4096],
                    ["C128", 128, "state", 3, "torch.float16", 8192],
                ],
            }
        )
        event = self.events(log)[0]
        self.assertEqual(event["event"], "registered_pools")
        self.assertEqual([p["name"] for p in event["pools"]], ["KV", "C128"])
        self.assertNotIn("private-model-revision", log.records[0])

    def test_schema_hook_reads_only_registered_entries(self):
        capture, log = self.make_capture()
        module = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
        cache = SimpleNamespace(
            kv_layout="page_first",
            _unified_kv=True,
            uniform_fp8=False,
            layer_mapping=[SimpleNamespace(compress_ratio=4, compress_layer_id=1)],
            start_layer=0,
            end_layer=1,
        )
        pool = SimpleNamespace(
            page_size=256,
            layout="page_first",
            layer_num=8,
            dtype="torch.uint8",
            item_bytes=4,
            state_page_bytes=None,
        )
        group = SimpleNamespace(
            entries=[SimpleNamespace(name=PoolName.KV, host_pool=pool)]
        )
        parallel = SimpleNamespace(
            attn_tp_size=1,
            attn_cp_size=1,
            pp_size=1,
            attn_cp_rank=0,
        )
        with (
            patch(module + "get_parallel", return_value=parallel),
            patch(module + "get_model", return_value=SimpleNamespace(revision="rev")),
            patch(module + "shared_cache_diagnostics", capture),
        ):
            schema = deepseek_v4_storage_schema(cache, group)

        self.assertEqual([pool[0] for pool in schema["pools"]], [str(PoolName.KV)])
        self.assertEqual(self.events(log)[0]["pools"][0]["page_size"], 256)

    def test_hmac_identity_is_tenant_case_epoch_and_domain_bound(self):
        capture, _ = self.make_capture()
        same = capture.key_id("exact-component-key", tenant_id="tenant-a")
        self.assertEqual(
            same, capture.key_id("exact-component-key", tenant_id="tenant-a")
        )
        self.assertNotEqual(
            same, capture.key_id("exact-component-key", tenant_id="tenant-b")
        )
        self.assertNotEqual(
            same,
            capture._request_id("exact-component-key", tenant_id="tenant-a"),
        )

    def test_key_manifest_miss_is_ignored_without_truncating_capture(self):
        capture, log = self.make_capture(allowed_keys=[])
        capture.record_io(
            is_set=False,
            pool="KV",
            keys=["unlisted-sensitive-key"],
            sizes=[8],
            results=[8],
        )
        events = self.events(log)
        self.assertEqual(events, [])
        self.assertFalse(capture._truncated)
        self.assertNotIn("unlisted-sensitive-key", "\n".join(log.records))

    def test_request_limit_counts_unique_hashed_requests(self):
        capture, log = self.make_capture(max_requests=1)
        capture.record_backup(
            phase="submitted",
            request_id="request-a",
            operation_id=4,
            complete=False,
            tokens=10,
        )
        capture.record_backup(
            phase="ack",
            request_id="request-a",
            operation_id=4,
            complete=True,
            tokens=10,
        )
        capture.record_backup(
            phase="submitted",
            request_id="request-b",
            operation_id=5,
            complete=False,
            tokens=4,
        )
        events = self.events(log)
        self.assertEqual(
            [event["event"] for event in events[:2]], ["backup_submitted", "backup_ack"]
        )
        self.assertEqual(events[-1]["event"], "capture_truncated")
        self.assertEqual(events[-1]["dropped_requests"], 1)
        self.assertNotIn("request-a", "\n".join(log.records))

    def test_duration_limit_stops_before_key_hashing(self):
        with patch(
            "sglang.srt.mem_cache.shared_cache_diagnostics.time.monotonic"
        ) as now:
            now.return_value = 1.0
            capture, log = self.make_capture(max_duration_ms=1000)
            now.return_value = 3.0
            with patch(
                "sglang.srt.mem_cache.shared_cache_diagnostics.hmac.new",
                side_effect=AssertionError("expired capture hashed a key"),
            ):
                capture.record_io(
                    is_set=False,
                    pool="KV",
                    keys=["private-object-key"],
                    sizes=[8],
                    results=[8],
                )
        self.assertEqual(self.events(log)[0]["event"], "capture_truncated")

    def test_mooncake_batch_io_preserves_component_results_without_raw_keys(self):
        component_keys = [
            f"{pool}:private-put-key-{index}"
            for pool in (PoolName.KV, PoolName.DEEPSEEK_V4_C4)
            for index in (1, 2)
        ]
        capture, log = self.make_capture(allowed_keys=component_keys)
        store = object.__new__(MooncakeStore)
        pool = SimpleNamespace(
            page_size=1,
            get_page_buffer_meta=lambda indices: ([100, 200], [[2, 2], [2, 2]]),
        )
        store.registered_pools = {PoolName.KV: pool, PoolName.DEEPSEEK_V4_C4: pool}
        store._tag_keys = lambda keys: keys
        store._get_hybrid_page_component_keys = lambda keys, transfer: (
            [f"{transfer.name}:{key}" for key in keys],
            1,
        )
        store._can_use_group_semantics = lambda: False
        store._filter_group_ids = lambda groups, indices: None
        store._batch_exist = MagicMock(return_value=[1, 0])
        store._put_batch_zero_copy_impl = MagicMock(return_value=[0])
        store._get_batch_zero_copy_impl = MagicMock(return_value=[4, 2])

        transfer = PoolTransfer(
            PoolName.KV,
            host_indices=torch.arange(2),
            keys=["private-put-key-1", "private-put-key-2"],
        )
        sidecar = PoolTransfer(
            PoolName.DEEPSEEK_V4_C4,
            host_indices=torch.arange(2),
            keys=["private-put-key-1", "private-put-key-2"],
        )
        with patch(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store.shared_cache_diagnostics",
            capture,
        ):
            self.assertEqual(
                store._batch_io_v2([transfer, sidecar], is_set=True)[PoolName.KV],
                [True, True],
            )
            self.assertEqual(
                store._batch_io_v2([transfer, sidecar], is_set=False)[PoolName.KV],
                [True, False],
            )

        put_kv, put_sidecar, get_kv, get_sidecar = self.events(log)
        self.assertEqual(put_kv["event"], "put_components")
        self.assertEqual(put_sidecar["pool"], str(PoolName.DEEPSEEK_V4_C4))
        self.assertEqual(
            [c["already_present"] for c in put_kv["components"]], [True, False]
        )
        self.assertEqual(
            [c["written"] for c in put_sidecar["components"]], [False, True]
        )
        self.assertEqual([c["result"] for c in get_kv["components"]], [4, 2])
        self.assertEqual([c["short_read"] for c in get_kv["components"]], [False, True])
        self.assertNotEqual(
            [c["key_id"] for c in get_kv["components"]],
            [c["key_id"] for c in get_sidecar["components"]],
        )
        for raw_key in transfer.keys:
            self.assertNotIn(raw_key, "\n".join(log.records))

    def test_capture_limits_emit_one_truncation_summary(self):
        capture, log = self.make_capture(
            max_events=2,
            max_keys=1,
            keys_per_event=1,
            allowed_keys=["key-1", "key-2", "key-3"],
        )
        capture.record_io(
            is_set=False,
            pool="KV",
            keys=["key-1", "key-2"],
            sizes=[1, 1],
            results=[1, 1],
        )
        capture.record_io(
            is_set=False,
            pool="KV",
            keys=["key-3"],
            sizes=[1],
            results=[1],
        )
        events = self.events(log)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event"], "get_components")
        self.assertTrue(events[0]["truncated"])
        self.assertEqual(events[1]["event"], "capture_truncated")
        self.assertGreaterEqual(events[1]["dropped_keys"], 1)

    def test_non_allowlisted_keys_do_not_truncate_capture(self):
        capture, log = self.make_capture(allowed_keys=["key-1"])
        capture.record_io(
            is_set=False,
            pool="KV",
            keys=["unknown-private-key"],
            sizes=[],
            results=[],
        )
        self.assertFalse(capture._truncated)
        self.assertEqual(log.records, [])

        capture.record_io(
            is_set=False,
            pool="KV",
            keys=["key-1"],
            sizes=[],
            results=[],
        )
        events = self.events(log)
        self.assertEqual(events[0]["event"], "get_components")
        self.assertTrue(events[0]["truncated"])
        self.assertEqual(events[0]["filtered_keys"], 1)

    def test_capture_logging_failure_is_counted_and_never_propagates(self):
        capture, _ = self.make_capture()
        capture._logger.info = MagicMock(side_effect=RuntimeError("sink failed"))
        capture.record_io(
            is_set=False,
            pool="KV",
            keys=["key-1"],
            sizes=[1],
            results=[1],
        )
        self.assertEqual(capture.capture_failures, 1)

    def test_malformed_capture_value_is_counted_and_never_propagates(self):
        capture, _ = self.make_capture()
        with patch.object(capture, "_key_id", side_effect=ValueError("bad key")):
            capture.record_io(
                is_set=False,
                pool="KV",
                keys=["key-1"],
                sizes=[1],
                results=[1],
            )
        self.assertEqual(capture.capture_failures, 1)

    def test_prefill_forward_and_c128_completion_are_correlated(self):
        capture, log = self.make_capture()
        capture.record_prefill_forward(
            request_id="private-prefill-request",
            h_tokens=128,
            n_tokens=385,
            forward_start=128,
            forward_end=385,
            tail_complete=True,
        )
        capture.record_c128_transfer(
            request_id="private-prefill-request",
            room=77,
            index_count=4,
            online=False,
            sender_mode="MooncakeKVSender",
            completed=True,
        )
        forward, c128 = self.events(log)
        self.assertEqual(forward["event"], "prefill_forward_complete")
        self.assertEqual(forward["h_tokens"], 128)
        self.assertEqual(forward["n_tokens"], 385)
        self.assertEqual(forward["suffix_tokens"], 257)
        self.assertEqual(forward["forward_tokens"], 257)
        self.assertTrue(forward["tail_complete"])
        self.assertEqual(forward["request_id"], c128["request_id"])
        self.assertEqual(c128["index_count"], 4)
        self.assertFalse(c128["online"])
        self.assertTrue(c128["completed"])
        self.assertNotIn("private-prefill-request", "\n".join(log.records))

    def test_real_prefill_result_hook_records_total_n_after_copy(self):
        capture, log = self.make_capture()
        order = []
        request = SimpleNamespace(
            rid="private-prefill-request",
            extend_range=SimpleNamespace(start=128, end=385),
            prefix_indices=list(range(100)),
            host_hit_length=28,
            origin_input_ids=list(range(385)),
            inflight_middle_chunks=0,
        )

        class BadDiagnosticRequest:
            rid = "bad-request"

            @property
            def extend_range(self):
                raise RuntimeError("bad diagnostic field")

        bad_request = BadDiagnosticRequest()
        result = SimpleNamespace(
            logits_output=None,
            next_token_ids=None,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            copy_done=SimpleNamespace(synchronize=lambda: order.append("copy")),
        )
        batch = SimpleNamespace(reqs=[request, bad_request])
        scheduler = SimpleNamespace(
            batch_result_processor=SimpleNamespace(
                snapshot_auxiliary_output_starts=lambda _batch, _result: (
                    order.append("snapshot"),
                    (_ for _ in ()).throw(StopIteration()),
                )[1]
            )
        )

        with patch(
            "sglang.srt.disaggregation.prefill.shared_cache_diagnostics", capture
        ):
            with self.assertRaises(StopIteration):
                SchedulerDisaggregationPrefillMixin.process_batch_result_disagg_prefill(
                    scheduler, batch, result
                )

        event = self.events(log)[0]
        self.assertEqual(order, ["copy", "snapshot"])
        self.assertEqual(event["h_tokens"], 128)
        self.assertEqual(event["n_tokens"], 385)
        self.assertEqual(event["suffix_tokens"], 257)
        self.assertEqual((event["forward_start"], event["forward_end"]), (128, 385))
        self.assertEqual(capture.capture_failures, 1)

    def test_real_prefill_transfer_handler_records_terminal_c128_and_cleans_up(self):
        for poll, expected_completed in (
            (KVPoll.Success, True),
            (KVPoll.Failed, False),
            (KVPoll.Success, None),
        ):
            with self.subTest(poll=poll):
                capture, log = self.make_capture()
                request_type = SimpleNamespace
                if expected_completed is None:

                    class CleanupErrorRequest(SimpleNamespace):
                        def __delattr__(self, name):
                            super().__delattr__(name)
                            if name == "_shared_cache_diag_c128":
                                raise RuntimeError("diagnostic cleanup failed")

                    request_type = CleanupErrorRequest
                diag_metadata = {
                    "index_count": 4,
                    "online": False,
                    "sender_mode": "MooncakeKVSender",
                }
                if expected_completed is not None:
                    diag_metadata["room"] = 77
                request = request_type(
                    rid="private-transfer-request",
                    pending_bootstrap=False,
                    finished_reason=None,
                    cache_request_handle=object(),
                    bootstrap_host=FAKE_BOOTSTRAP_HOST,
                    return_logprob=False,
                    disagg_kv_sender=MagicMock(),
                    time_stats=MagicMock(),
                    _shared_cache_diag_c128=diag_metadata,
                )
                scheduler = MagicMock()
                scheduler.disagg_prefill_inflight_queue = [request]
                scheduler._record_c128_transfer_completion = (
                    SchedulerDisaggregationPrefillMixin._record_c128_transfer_completion
                )

                with (
                    patch(
                        "sglang.srt.disaggregation.prefill.shared_cache_diagnostics",
                        capture,
                    ),
                    patch(
                        "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                        return_value=[poll],
                    ),
                    patch("sglang.srt.disaggregation.prefill.release_kv_cache"),
                    patch(
                        "sglang.srt.disaggregation.prefill.maybe_release_metadata_buffer"
                    ),
                ):
                    done = SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                        scheduler
                    )

                events = self.events(log)
                self.assertEqual(done, [request])
                if expected_completed is None:
                    self.assertEqual(events, [])
                    self.assertEqual(capture.capture_failures, 2)
                else:
                    event = events[0]
                    self.assertEqual(event["event"], "c128_transfer")
                    self.assertEqual(event["index_count"], 4)
                    self.assertEqual(event["completed"], expected_completed)
                self.assertFalse(hasattr(request, "_shared_cache_diag_c128"))
                self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])

    def test_p_recovery_boundary_uses_hashed_request_identity(self):
        capture, log = self.make_capture()
        request = CacheRequestHandle(rid="private-prefill-request", attempt_id=1)
        cache = object.__new__(UnifiedRadixCache)
        cache.ongoing_prefetch = {request: (None, [1, 2], None, None, None, None)}
        cache._check_hybrid_prefetch_result = MagicMock(return_value=False)
        operation = SimpleNamespace(
            handle=request, completed_tokens=1, hash_value=["private-page-key"]
        )
        with patch(
            "sglang.srt.mem_cache.unified_radix_cache.shared_cache_diagnostics",
            capture,
        ):
            cache._handle_prefetch_result(operation)

        event = self.events(log)[0]
        self.assertEqual(event["event"], "prefetch_boundary")
        self.assertFalse(event["accepted"])
        self.assertEqual(event["completed_tokens"], 1)
        self.assertNotIn("private-prefill-request", log.records[0])


if __name__ == "__main__":
    unittest.main()
