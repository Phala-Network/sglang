"""Execute the bounded capture state and the real P/D submission path on CPU."""

import copy
import hashlib
import hmac
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.mooncake import pd_transfer_diagnostics as diagnostics
from sglang.srt.managers.io_struct import GenerateReqInput
from test_mooncake_pd_diagnostics_cpu import load_manager


def request_ref(value):
    fields = ("case", "epoch", "default", value)
    message = b"phala.shared-cache-request.v1\0"
    for field in fields:
        encoded = field.encode()
        message += struct.pack(">I", len(encoded)) + encoded
    return hmac.new(b"offline-fixture-salt", message, hashlib.sha256).hexdigest()


def native_record(*, transport="nvlink_intraNode", status="completed", size=17):
    return {
        "result": 0 if status == "completed" else -1,
        "batch_sequence": 7, "diagnostics_truncated": False,
        "attempts": [{"attempt": 0, "task_count": 1, "missing_transports": 0,
                      "selected_transports": {transport: 1},
                      "terminal_status": status, "transferred_bytes": size}],
    }


class PDBatchCorrelationTests(unittest.TestCase):
    def setUp(self):
        self.log = Mock()
        self.capture = diagnostics.PDBatchDiagnostics(
            enabled=True, key_salt="offline-fixture-salt", case_id="case",
            epoch="epoch", request_ids=request_ref("external-id"), log=self.log,
        )
        self.ref = request_ref("external-id")

    def records(self):
        return [json.loads(call.args[1]) for call in self.log.info.call_args_list]

    def arm_worker(self, room=1234, internal="random-worker-rid"):
        self.capture.bind_worker(internal, room, "prefill", 0, self.ref)

    def test_header_gate_overwrites_forgery_and_joins_distinct_n2_rooms(self):
        obj = NS(bootstrap_room=[1234, 5678], _pd_diagnostic_request_ref="forged")
        with patch.object(diagnostics, "pd_batch_diagnostics", self.capture):
            for headers in ({}, {"x-request-id": "wrong"}):
                diagnostics.bind_pd_ingress(obj, NS(headers=headers), "prefill")
                self.assertIsNone(obj._pd_diagnostic_request_ref)
                self.assertIsNone(self.capture._started_at)
            diagnostics.bind_pd_ingress(
                obj, NS(headers={"x-request-id": "external-id"}), "prefill"
            )
        self.assertEqual(obj._pd_diagnostic_request_ref, self.ref)
        self.arm_worker(1234, "choice-a")
        self.arm_worker(5678, "choice-b")
        self.capture.record_native(1234, "kv", [17], native_record(), 0)
        records = self.records()
        ingress = [row for row in records if row["event"] == "pd_ingress_bind"]
        worker = [row for row in records if row["event"] == "pd_worker_bind"]
        self.assertEqual(len(ingress), 2)
        self.assertNotEqual(ingress[0]["room_id"], ingress[1]["room_id"])
        self.assertEqual({row["room_id"] for row in ingress}, {row["room_id"] for row in worker})
        self.assertEqual(records[-1]["request_id"], self.ref)
        encoded = json.dumps(records)
        for raw in ("external-id", "random-worker-rid", "choice-a", "choice-b", "offline-fixture-salt"):
            self.assertNotIn(raw, encoded)

    def test_real_n2_normalization_keeps_server_ref_after_regenerated_rids(self):
        obj = GenerateReqInput(input_ids=[1, 2], sampling_params={"n": 2}, bootstrap_room=[1234, 5678])
        obj._pd_diagnostic_request_ref = self.ref
        obj.normalize_batch_and_arguments()
        choices = [copy.copy(obj[i]) for i in range(2)]
        self.assertEqual([item.bootstrap_room for item in choices], [1234, 5678])
        old = [item.rid for item in choices]
        for item in choices:
            item.regenerate_rid()
            self.assertEqual(item._pd_diagnostic_request_ref, self.ref)
        self.assertNotEqual([item.rid for item in choices], old)
        obj._pd_diagnostic_request_ref = None
        self.assertIsNone(obj[0]._pd_diagnostic_request_ref)
        self.assertNotIn("pd_diagnostic_request_ref", GenerateReqInput.__dataclass_fields__)

    def test_native_truth_rejects_tcp_partial_bytes_retry_fallback_and_missing_tasks(self):
        self.arm_worker()
        for record in (native_record(transport="tcp"), native_record(size=16), native_record(status="failed")):
            self.capture.record_native(1234, "kv", [17], record, 0)
            self.assertFalse(self.records()[-1]["all_attempts_nvlink_intra"])
        mixed = native_record()
        mixed["attempts"].insert(0, native_record(transport="tcp", status="failed")["attempts"][0])
        self.capture.record_native(1234, "kv", [17], mixed, 0)
        self.assertTrue(self.records()[-1]["completed_with_exact_bytes"])
        self.assertFalse(self.records()[-1]["all_attempts_nvlink_intra"])
        self.capture.record_native(1234, "kv", [17], native_record(), 0)
        self.assertTrue(self.records()[-1]["all_attempts_nvlink_intra"])
        missing = native_record()
        missing["attempts"][0]["missing_transports"] = 1
        self.capture.record_native(1234, "kv", [17], missing, 0)
        self.assertFalse(self.records()[-1]["all_attempts_nvlink_intra"])

    def test_native_submission_is_once_and_capture_failure_never_resubmits(self):
        manager = load_manager({"envs": NS(SGLANG_MOONCAKE_PD_TRANSFER_DIAGNOSTICS=NS(get=lambda: False))})
        manager.kv_args = NS(engine_rank=0)
        manager.engine = NS(
            batch_transfer_sync=Mock(return_value=0),
            batch_transfer_sync_diagnostic=Mock(return_value=native_record()),
        )
        self.arm_worker()
        with patch.object(diagnostics, "pd_batch_diagnostics", self.capture):
            result = manager._transfer_data("private-endpoint", [(999, 777, 17)], diagnostic_room=1234)
            self.assertEqual(result, 0)
            manager.engine.batch_transfer_sync.assert_not_called()
            manager.engine.batch_transfer_sync_diagnostic.assert_called_once()
            self.log.info.side_effect = RuntimeError("private logger failure")
            self.assertEqual(manager._transfer_data("private-endpoint", [(999, 777, 17)], diagnostic_room=1234), 0)
            self.assertEqual(manager.engine.batch_transfer_sync_diagnostic.call_count, 2)
            manager.engine.batch_transfer_sync.assert_not_called()
        self.assertGreater(self.capture.capture_failures, 0)

    def test_once_file_arm_is_private_and_does_not_start_for_mismatch_or_rearm_after_expiry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "capture.json"
            capture = diagnostics.PDBatchDiagnostics(config_path=str(path), log=self.log)
            self.assertIsNone(capture.external_request_ref("external-id"))
            config = {"schema": "phala.pd-batch-capture.v1", "key_salt": "offline-fixture-salt",
                      "case_id": "case", "epoch": "epoch", "tenant_id": "default",
                      "request_ids": [self.ref], "max_duration_ms": 1}
            path.write_text(json.dumps(config), encoding="utf-8")
            os.chmod(path, 0o600)
            self.assertIsNone(capture.external_request_ref("wrong"))
            self.assertIsNone(capture._started_at)
            self.assertEqual(capture.external_request_ref("external-id"), self.ref)
            self.assertIsNone(capture._started_at)
            capture.bind_worker("internal", 1, "prefill", 0, self.ref)
            capture._started_at -= 1
            self.assertFalse(capture.active_room(1))
            started = capture._started_at
            config["max_duration_ms"] = 300000
            path.write_text(json.dumps(config), encoding="utf-8")
            capture.bind_worker("internal", 1, "prefill", 0, self.ref)
            self.assertEqual(capture._started_at, started)
            self.assertFalse(capture.active_room(1))

    def test_symlink_oversize_or_permissions_are_permanent_rejections(self):
        with tempfile.TemporaryDirectory() as folder:
            actual = Path(folder) / "actual"
            actual.write_text("{}", encoding="utf-8")
            os.chmod(actual, 0o600)
            link = Path(folder) / "link"
            link.symlink_to(actual)
            oversized = Path(folder) / "oversized"
            oversized.write_bytes(b" " * 65537)
            os.chmod(oversized, 0o600)
            public = Path(folder) / "public"
            public.write_text("{}", encoding="utf-8")
            os.chmod(public, 0o644)
            for path in (link, oversized, public):
                capture = diagnostics.PDBatchDiagnostics(config_path=str(path), log=self.log)
                self.assertIsNone(capture.external_request_ref("external-id"))
                self.assertTrue(capture._load_attempted)
                self.assertFalse(capture.enabled)

    def test_bounds_and_reused_room_fail_closed(self):
        self.arm_worker()
        self.capture.bind_worker("other-internal", 1234, "prefill", 0, self.ref)
        self.assertFalse(self.capture.active_room(1234))
        self.arm_worker()
        self.assertFalse(self.capture.active_room(1234))
        limited = diagnostics.PDBatchDiagnostics(
            enabled=True, key_salt="offline-fixture-salt", case_id="case", epoch="epoch",
            request_ids=self.ref, max_events=2, log=self.log,
        )
        limited.bind_worker("internal", 1, "prefill", 0, self.ref)
        self.assertFalse(limited.active_room(1))


if __name__ == "__main__":
    unittest.main()
