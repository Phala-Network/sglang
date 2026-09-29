import asyncio
from queue import Queue
import hashlib
import json
from types import SimpleNamespace

import pytest

from sglang.srt.managers.shared_cache_control import (
    SharedCacheControlError,
    _PENDING_FIELDS,
    _private_jsonl,
    execute_bounded_clear,
    expected_identities,
    load_manifest,
    clear_from_configured_artifacts,
    _stable_id,
    parse_clear_selector_body,
    validate_clear_selectors,
    validate_owner_drain,
    validate_single_decode_writer,
    validate_snapshot,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _clear_context(tenant="tenant-1", multi_tenant_enabled=True):
    store = object()
    backend = SimpleNamespace(
        store=store,
        shared_cache_store_instance_id="store-1",
    )
    scheduler = SimpleNamespace(
        decode_offload_manager=SimpleNamespace(
            cache_controller=SimpleNamespace(storage_backend=backend)
        ),
        ps=SimpleNamespace(tp_size=1, dp_size=1, pp_size=1, attn_cp_size=1),
        server_args=SimpleNamespace(
            tp_size=1,
            dp_size=1,
            pp_size=1,
            dcp_size=1,
            attn_cp_size=1,
            nnodes=1,
        ),
        disaggregation_mode="decode",
    )
    manifest = {
        "case_id": "case-1",
        "epoch": "epoch-1",
        "tenant": tenant,
        "backend_tag": "backend-1",
        "original_D_worker_id": "writer-1",
        "original_store_instance_id": "store-1",
        "manifest_sha256": "a" * 64,
    }
    master_evidence = {
        "case_id": "case-1",
        "epoch": "epoch-1",
        "tenant_id": tenant,
        "requested_tenant_id": tenant,
        "effective_tenant_id": tenant,
        "master_multi_tenant_enabled": multi_tenant_enabled,
        "backend_tag": "backend-1",
        "d_worker_id": "writer-1",
        "store_instance_id": "store-1",
        "master_pid": 100,
        "init_scan_completed_unix_ms": 900,
        "bucket_eviction_policy": "none",
        "disk_watermark_eviction": False,
        "sample_time_unix_ms": 1000,
    }
    return scheduler, store, manifest, master_evidence


def _owner_drain_fixture():
    manifest = {
        "case_id": "case-1",
        "epoch": "epoch-1",
        "tenant": "default",
        "keys": [
            {
                "ssd_owner_uuid": "2-2",
                "ssd_scope": "endpoint-1",
            }
        ],
    }
    owner_id = _stable_id(
        b"test-salt",
        b"phala.shared-cache-owner.v1\0",
        "case-1",
        "epoch-1",
        "default",
        manifest["keys"][0]["ssd_owner_uuid"],
    )
    scope_id = _stable_id(
        b"test-salt",
        b"phala.shared-cache-owner.v1\0",
        "case-1",
        "epoch-1",
        "default",
        "endpoint-1",
    )
    manifest["_expected_ssd_owner_ids"] = {
        manifest["keys"][0]["ssd_owner_uuid"]: owner_id
    }
    manifest["_expected_ssd_scope_ids"] = {"endpoint-1": scope_id}
    mapping = {
        "owner_id": owner_id,
        "store_instance_id": "c" * 64,
        "scope_id": scope_id,
        "backend_path_id": "d" * 64,
        "pid": 200,
        "init_completed_unix_ms": 600,
        "scan_completed_unix_ms": 700,
    }
    evidence = {"ssd_owner_mappings": [mapping]}
    sample = {
        "schema": "phala.shared-cache.owner-drain.v1",
        "kind": "owner_drain",
        "case_id": "case-1",
        "epoch": "epoch-1",
        "tenant_id": "default",
        "pid": 200,
        "owner_id": owner_id,
        "store_instance_id": mapping["store_instance_id"],
        "scope_id": scope_id,
        "backend_path_id": mapping["backend_path_id"],
        "owner_client_requested_tenant_id": "default",
        "coverage": "owner_global_bucket",
        "sample_time_unix_ms": 750,
        "init_completed_unix_ms": 600,
        "scan_completed_unix_ms": 700,
        "owner_sample_sequence": 1,
        "activity_sequence": 8,
        "active": {
            field: 0
            for field in (
                "loads",
                "offloads",
                "promotions",
                "removes",
                "heartbeats",
                "rescans",
                "leased_read_buffers",
            )
        },
        "pending": {
            field: 0
            for field in (
                "backend_writes",
                "backend_evictions",
                "ungrouped_offloads",
                "read_guards",
            )
        },
        "bucket_count": 1,
        "covered_buckets": 1,
        "available": True,
        "consistent": True,
        "backend_initialized": True,
        "bucket_eviction_policy": "NONE",
        "disk_watermark_eviction": False,
        "metadata_resync_pending": False,
        "draining": False,
        "heartbeat_ok": True,
    }
    records = [
        {**sample, "sample_time_unix_ms": t, "owner_sample_sequence": seq}
        for seq, t in ((10, 750), (11, 875), (12, 1000))
    ]
    return manifest, evidence, mapping, records


def _write_owner_jsonl(path, records, *, summary_overrides=None):
    lines = [
        json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
        for item in records
    ]
    summary = {
        "kind": "capture_summary",
        "emitted": len(lines),
        "dropped": 0,
        "rejected": 0,
        "complete": True,
        "bytes": sum(len(line) + 1 for line in lines),
        **(summary_overrides or {}),
    }
    path.write_bytes(
        b"".join(line + b"\n" for line in lines)
        + json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    path.chmod(0o600)
    return summary


def test_owner_drain_jsonl_requires_complete_final_summary(tmp_path):
    _, _, _, records = _owner_drain_fixture()
    path = tmp_path / "owner.jsonl"
    _write_owner_jsonl(path, records)
    parsed, summary = _private_jsonl(str(path))
    assert len(parsed) == 3
    assert summary["complete"] is True

    valid = path.read_bytes()
    cases = (
        (valid.rsplit(b"{\"bytes\"", 1)[0], "owner_drain_summary_missing"),
        (valid[:-1], "owner_drain_file_incomplete"),
    )
    for payload, reason in cases:
        path.write_bytes(payload)
        path.chmod(0o600)
        with pytest.raises(SharedCacheControlError, match=reason):
            _private_jsonl(str(path))

    _write_owner_jsonl(path, records, summary_overrides={"dropped": 1})
    with pytest.raises(SharedCacheControlError, match="owner_drain_capture_incomplete"):
        _private_jsonl(str(path))

    _write_owner_jsonl(path, records)
    lines = path.read_bytes().splitlines()
    path.write_bytes(lines[-1] + b"\n" + b"\n".join(lines[:-1]) + b"\n")
    path.chmod(0o600)
    with pytest.raises(SharedCacheControlError, match="owner_drain_summary_not_final"):
        _private_jsonl(str(path))


def test_owner_drain_jsonl_counts_mixed_event_bytes_including_lf(tmp_path):
    _, _, _, owner_records = _owner_drain_fixture()
    native_record = {
        "schema": "phala.shared-cache.native.v1",
        "case_id": "case-1",
        "epoch": "epoch-1",
        "kind": "cache_event",
    }
    records = [*owner_records, native_record]
    path = tmp_path / "mixed-owner.jsonl"
    summary = _write_owner_jsonl(path, records)
    event_lines = path.read_bytes().splitlines(keepends=True)[:-1]
    assert summary["emitted"] == len(records)
    assert summary["bytes"] == sum(map(len, event_lines))
    parsed, parsed_summary = _private_jsonl(str(path))
    assert parsed == records
    assert parsed_summary == summary

    _write_owner_jsonl(
        path,
        records,
        summary_overrides={"bytes": sum(len(line) - 1 for line in event_lines)},
    )
    with pytest.raises(SharedCacheControlError, match="owner_drain_capture_incomplete"):
        _private_jsonl(str(path))


def test_expected_identities_match_native_uint64_pair_format():
    manifest = {
        "key_salt": "test-salt",
        "case_id": "case-1",
        "epoch": "epoch-1",
        "tenant": "default",
        "original_writer_client_id": "1102750741119211504-12071701333474074511",
        "exact_nonempty_MEMORY_segment": "MEMORY-cold",
        "keys": [
            {
                "ssd_owner_uuid": "00018446744073709551615-0000000000000000007",
                "ssd_scope": "endpoint-1",
            }
        ],
    }
    writer_hash, segment_hash = expected_identities(manifest)
    context = ("case-1", "epoch-1", "default")
    assert writer_hash == _stable_id(
        b"test-salt",
        b"phala.shared-cache-owner.v1\0",
        *context,
        "1102750741119211504-12071701333474074511",
    )
    assert segment_hash == _stable_id(
        b"test-salt",
        b"phala.shared-cache-segment.v1\0",
        *context,
        "MEMORY-cold",
    )
    assert manifest["_expected_ssd_owner_ids"][
        "00018446744073709551615-0000000000000000007"
    ] == _stable_id(
        b"test-salt",
        b"phala.shared-cache-owner.v1\0",
        *context,
        "18446744073709551615-7",
    )

    manifest["original_writer_client_id"] = "18446744073709551616-1"
    with pytest.raises(SharedCacheControlError, match="invalid_writer_identity"):
        expected_identities(manifest)


def test_owner_drain_uses_time_coverage_not_fixed_sample_count():
    manifest, evidence, _, records = _owner_drain_fixture()
    two_samples = [
        {**records[0], "owner_sample_sequence": 20, "sample_time_unix_ms": 750},
        {**records[1], "owner_sample_sequence": 21, "sample_time_unix_ms": 1000},
    ]
    summary = {
        "emitted": 2,
        "dropped": 0,
        "rejected": 0,
        "bytes": 0,
        "complete": True,
    }
    receipt = validate_owner_drain(
        two_samples,
        manifest,
        evidence,
        summary=summary,
        quiet_ms=250,
        now_ms=1000,
    )
    assert receipt["quiet_start_unix_ms"] == 750
    assert receipt["quiet_end_unix_ms"] == 1000


def test_owner_drain_accepts_scan_before_init():
    manifest, evidence, mapping, records = _owner_drain_fixture()
    mapping["scan_completed_unix_ms"] = 550
    for sample in records:
        sample["scan_completed_unix_ms"] = 550
    receipt = validate_owner_drain(
        records,
        manifest,
        evidence,
        summary={"emitted": len(records), "dropped": 0, "rejected": 0, "bytes": 0, "complete": True},
        quiet_ms=250,
        now_ms=1000,
    )
    assert receipt["quiet_end_unix_ms"] == 1000


def test_owner_drain_accepts_same_millisecond_samples_with_increasing_sequence():
    manifest, evidence, _, records = _owner_drain_fixture()
    records[2]["sample_time_unix_ms"] = records[1]["sample_time_unix_ms"]
    records.append({**records[2], "owner_sample_sequence": 13, "sample_time_unix_ms": 1000})
    receipt = validate_owner_drain(
        records,
        manifest,
        evidence,
        summary={"emitted": len(records), "dropped": 0, "rejected": 0, "bytes": 0, "complete": True},
        quiet_ms=250,
        now_ms=1000,
    )
    assert receipt["quiet_start_unix_ms"] == 750
    assert receipt["quiet_end_unix_ms"] == 1000


def test_owner_drain_fails_on_sequence_activity_or_freshness_gap():
    manifest, evidence, _, records = _owner_drain_fixture()
    summary = {
        "emitted": 3,
        "dropped": 0,
        "rejected": 0,
        "bytes": 0,
        "complete": True,
    }
    cases = []
    gap = [dict(item) for item in records]
    gap[1]["owner_sample_sequence"] += 1
    cases.append((gap, 1000, "owner_drain_sequence_discontinuous"))
    activity = [dict(item) for item in records]
    activity[-1]["activity_sequence"] += 1
    cases.append((activity, 1000, "owner_drain_quiet_window_missing"))
    stale = [dict(item) for item in records]
    cases.append((stale, 40_000, "owner_drain_sample_stale"))
    changed_owner = [dict(item) for item in records]
    changed_owner[-1]["backend_path_id"] = "e" * 64
    cases.append((changed_owner, 1000, "owner_drain_sample_invalid"))
    before_init = [dict(item) for item in records]
    before_init[-1]["sample_time_unix_ms"] = 599
    cases.append((before_init, 1000, "owner_drain_sample_invalid"))
    for samples, now_ms, reason in cases:
        with pytest.raises(SharedCacheControlError, match=reason):
            validate_owner_drain(
                samples,
                manifest,
                evidence,
                summary=summary,
                quiet_ms=250,
                now_ms=now_ms,
            )


def test_clear_requires_actual_multitenant_master_evidence():
    assert "dynamic_replication" in _PENDING_FIELDS
    invalid_evidence = [
        {"master_multi_tenant_enabled": False},
        {"requested_tenant_id": "default"},
        {"effective_tenant_id": "default"},
        {"requested_tenant_id": None},
    ]
    for overrides in invalid_evidence:
        scheduler, store, manifest, evidence = _clear_context()
        evidence.update(overrides)
        with pytest.raises(SharedCacheControlError, match="master_evidence_invalid"):
            execute_bounded_clear(
                scheduler=scheduler,
                native_store=store,
                manifest=manifest,
                snapshot_reader=lambda _sample_id: pytest.fail(
                    "invalid master evidence must be rejected before snapshots"
                ),
                master_evidence=evidence,
                writer_id="writer-hash",
                segment_id="segment-hash",
                now_ms=lambda: 1000,
            )


def test_master_evidence_timestamps_fail_closed_before_snapshot_or_clear():
    invalid_fields = (
        ({}, "sample_time_unix_ms"),
        ({"sample_time_unix_ms": "1000"}, None),
        ({"sample_time_unix_ms": True}, None),
        ({}, "init_scan_completed_unix_ms"),
        ({"init_scan_completed_unix_ms": "900"}, None),
        ({"init_scan_completed_unix_ms": True}, None),
    )

    class NativeStore:
        def __init__(self):
            self.calls = []

        def batch_replica_clear(self, *_args):
            self.calls.append(True)
            return []

    for overrides, missing_field in invalid_fields:
        scheduler, _, manifest, evidence = _clear_context()
        native = NativeStore()
        scheduler.decode_offload_manager.cache_controller.storage_backend.store = native
        evidence.update(overrides)
        if missing_field:
            evidence.pop(missing_field)
        with pytest.raises(SharedCacheControlError, match="master_evidence_invalid"):
            execute_bounded_clear(
                scheduler=scheduler,
                native_store=native,
                manifest=manifest,
                snapshot_reader=lambda _sample_id: pytest.fail(
                    "invalid timestamps must fail before native snapshots"
                ),
                master_evidence=evidence,
                writer_id="writer-hash",
                segment_id="segment-hash",
                now_ms=lambda: 1000,
            )
        assert native.calls == []


def test_fully_idle_scheduler_is_a_drained_state():
    controller = SimpleNamespace(
        ack_backup_queue=Queue(),
        backup_queue=Queue(),
        ack_write_queue=[],
        write_queue=[],
        ack_load_queue=[],
        ack_prefetch_queue=Queue(),
    )
    manager = SimpleNamespace(
        cache_controller=controller,
        ongoing_offload={},
        ongoing_backup={},
        offload_inflight={},
    )
    scheduler = SimpleNamespace(
        decode_offload_manager=manager,
        is_fully_idle=lambda: True,
    )
    from sglang.srt.managers.shared_cache_control import require_stable_drain

    assert require_stable_drain(scheduler, quiet_ms=50) == {
        "scheduler_busy": 0,
        "ongoing_offload": 0,
        "ongoing_backup": 0,
        "offload_inflight": 0,
        "ack_write": 0,
        "ack_backup": 0,
        "backup_queue": 0,
        "write_queue": 0,
        "ack_load": 0,
        "ack_prefetch": 0,
    }


def test_clear_does_not_trust_configured_tenant_as_effective_tenant():
    scheduler, store, manifest, evidence = _clear_context()
    scheduler.decode_offload_manager.cache_controller.storage_backend.config = (
        SimpleNamespace(tenant_id="tenant-1")
    )
    evidence["effective_tenant_id"] = "default"
    with pytest.raises(SharedCacheControlError, match="master_evidence_invalid"):
        execute_bounded_clear(
            scheduler=scheduler,
            native_store=store,
            manifest=manifest,
            snapshot_reader=lambda _sample_id: pytest.fail(
                "effective tenant mismatch must be rejected before snapshots"
            ),
            master_evidence=evidence,
            writer_id="writer-hash",
            segment_id="segment-hash",
            now_ms=lambda: 1000,
        )


def test_default_tenant_is_allowed_when_master_multitenancy_is_disabled():
    scheduler, store, manifest, evidence = _clear_context(
        tenant="default", multi_tenant_enabled=False
    )
    with pytest.raises(SharedCacheControlError, match="snapshot_identity_mismatch"):
        execute_bounded_clear(
            scheduler=scheduler,
            native_store=store,
            manifest=manifest,
            snapshot_reader=lambda _sample_id: {},
            master_evidence=evidence,
            writer_id="writer-hash",
            segment_id="segment-hash",
            now_ms=lambda: 1000,
        )


def test_snapshot_checks_effective_tenant_and_mode():
    manifest = {
        "tenant": "default",
        "case_id": "c",
        "epoch": "e",
        "keys": [{"key_id": "k", "logical_bytes": 1, "ssd_replica_ids": []}],
        "_expected_ssd_owner_ids": {},
        "_expected_ssd_scope_ids": {},
    }
    snapshot = {
        "schema": "phala.shared-cache.snapshot.v1",
        "success": True,
        "case_id": "c",
        "epoch": "e",
        "tenant_id": "default",
        "requested_tenant_id": "default",
        "effective_tenant_id": "default",
        "master_multi_tenant_enabled": False,
        "sample_id": "sample",
        "master_pid": 1,
        "sample_time_unix_ms": 1,
        "backend_drain_available": False,
        "objects": [
            {
                "key_id": "k",
                "found": True,
                "logical_bytes": 1,
                "writer_id": "writer",
                "lease_expired": True,
                "pending": {field: 0 for field in _PENDING_FIELDS},
                "replicas": [
                    {
                        "type": "MEMORY",
                        "segment_ids": ["segment"],
                        "status": "COMPLETE",
                        "readable": True,
                        "refcount": 0,
                    }
                ],
            }
        ],
    }
    ack = validate_snapshot(
        snapshot,
        manifest,
        "sample",
        expected_writer_id="writer",
        expected_segment_id="segment",
    )
    assert ack["master_pid"] == 1
    snapshot["effective_tenant_id"] = "default-fallback"
    with pytest.raises(SharedCacheControlError, match="snapshot_identity_mismatch"):
        validate_snapshot(
            snapshot,
            manifest,
            "sample",
            expected_writer_id="writer",
            expected_segment_id="segment",
        )


def test_clear_selector_bounds_and_exact_decode_topology():
    validate_clear_selectors("manifest-1", "a" * 64, "request-1")
    for values in (
        ("x" * 65, "a" * 64, "request-1"),
        ("manifest-1", "a" * 63, "request-1"),
        ("manifest-1", "a" * 64, "x" * 65),
    ):
        with pytest.raises(SharedCacheControlError):
            validate_clear_selectors(*values)

    topology = SimpleNamespace(
        tp_size=1,
        dp_size=1,
        pp_size=1,
        dcp_size=1,
        attn_cp_size=1,
        nnodes=1,
    )
    validate_single_decode_writer(topology, "decode")
    with pytest.raises(SharedCacheControlError, match="unsupported_topology"):
        validate_single_decode_writer(SimpleNamespace(**{**topology.__dict__, "attn_cp_size": 2}), "decode")
    with pytest.raises(SharedCacheControlError, match="decode_writer_required"):
        validate_single_decode_writer(topology, "prefill")


def test_clear_http_body_accepts_only_bounded_selectors():
    body = json.dumps(
        {
            "manifest_id": "manifest-1",
            "manifest_sha256": "a" * 64,
            "request_id": "request-1",
        }
    ).encode()
    assert parse_clear_selector_body(body)["manifest_id"] == "manifest-1"
    for raw, reason in (
        (body + b" " * 512, "request_body_too_large"),
        (
            json.dumps(
                {
                    "manifest_id": "manifest-1",
                    "manifest_sha256": "a" * 64,
                    "request_id": "request-1",
                    "path": "/tmp/anything",
                }
            ).encode(),
            "invalid_selector_fields",
        ),
        (b"not-json", "invalid_json"),
    ):
        with pytest.raises(SharedCacheControlError, match=reason):
            parse_clear_selector_body(raw)


def test_clear_requires_control_artifacts_before_native_access(monkeypatch):
    for name in (
        "SGLANG_SHARED_CACHE_CLEAR_MANIFEST",
        "SGLANG_SHARED_CACHE_MASTER_EVIDENCE",
        "SGLANG_SHARED_CACHE_NATIVE_ADMIN_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SharedCacheControlError, match="clear_control_not_configured"):
        clear_from_configured_artifacts(
            object(), manifest_id="manifest-1", manifest_sha256="a" * 64
        )
    with pytest.raises(SharedCacheControlError, match="invalid_manifest_selector"):
        clear_from_configured_artifacts(
            object(), manifest_id="/tmp/manifest.json", manifest_sha256="a" * 64
        )


def test_tokenizer_rejects_invalid_clear_before_ipc():
    from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin

    class Manager:
        server_args = SimpleNamespace()
        disaggregation_mode = "decode"

        def auto_create_handle_loop(self):
            pytest.fail("invalid selectors must fail before IPC setup")

        async def clear_shared_cache_memory_communicator(self, _request):
            pytest.fail("invalid selectors must fail before IPC")

    result = asyncio.run(
        TokenizerControlMixin.clear_shared_cache_memory(
            Manager(), "manifest-1", "bad-sha", "request-1"
        )
    )
    assert not result.success
    assert result.reason == "invalid_manifest_selector"


def test_clear_uses_store_instance_identity_from_live_wrapper():
    scheduler, store, manifest, evidence = _clear_context()
    scheduler.decode_offload_manager.cache_controller.storage_backend.shared_cache_store_instance_id = (
        "other-store"
    )
    with pytest.raises(SharedCacheControlError, match="original_store_instance_mismatch"):
        execute_bounded_clear(
            scheduler=scheduler,
            native_store=store,
            manifest=manifest,
            snapshot_reader=lambda _sample_id: pytest.fail(
                "different store instance must be rejected before snapshots"
            ),
            master_evidence=evidence,
            writer_id="writer-hash",
            segment_id="segment-hash",
            now_ms=lambda: 1000,
        )


def test_original_writer_clear_handler_flow_and_stable_drain(tmp_path):
    salt = "test-salt"
    writer_client_id = "1102750741119211504-12071701333474074511"
    owner_uuid = "123-456"
    manifest = {
        "manifest_version": 1,
        "manifest_id": "manifest-1",
        "run_id": "run-1",
        "case_id": "case-1",
        "epoch": "epoch-1",
        "model_revision": "revision-1",
        "kv_schema": "schema-1",
        "tenant": "default",
        "backend_tag": "backend-1",
        "original_D_worker_id": "writer-1",
        "original_writer_client_id": writer_client_id,
        "original_store_instance_id": "store-1",
        "exact_nonempty_MEMORY_segment": "MEMORY-cold",
        "key_salt": salt,
        "required_components": ["kv"],
        "keys": [
            {
                "key": "exact-key",
                "logical_bytes": 4096,
                "rank": 0,
                "component": "kv",
                "page_range": "0-0",
                "ssd_replica_ids": [2],
                "ssd_owner_uuid": owner_uuid,
                "ssd_scope": "endpoint-1",
            }
        ],
    }
    key = manifest["keys"][0]
    key["key_id"] = _stable_id(
        salt.encode(),
        b"phala.shared-cache-key.v1\0",
        "case-1",
        "epoch-1",
        "default",
        key["key"],
    )
    unsigned = dict(manifest)
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o600)
    manifest = load_manifest(
        str(manifest_path), "manifest-1", manifest["manifest_sha256"]
    )
    writer_hash, segment_hash = expected_identities(manifest)
    ssd_owner_hash = _stable_id(
        salt.encode(), b"phala.shared-cache-owner.v1\0", "case-1", "epoch-1", "default", owner_uuid
    )
    ssd_scope_hash = _stable_id(
        salt.encode(),
        b"phala.shared-cache-owner.v1\0",
        "case-1",
        "epoch-1",
        "default",
        "endpoint-1",
    )
    ssd = {
        "id": 2,
        "type": "LOCAL_DISK",
        "status": "COMPLETE",
        "readable": True,
        "refcount": 0,
        "segment_ids": [],
        "owner_id": ssd_owner_hash,
        "scope_id": ssd_scope_hash,
    }
    memory = {
        "id": 1,
        "type": "MEMORY",
        "status": "COMPLETE",
        "readable": True,
        "refcount": 0,
        "segment_ids": [segment_hash],
        "owner_id": None,
        "scope_id": None,
    }
    pending = {
        field: 0
        for field in (
            "processing",
            "offload",
            "promotion",
            "promotion_candidate",
            "replication",
            "dynamic_replication",
        )
    }

    def snapshot(sample_id, with_memory):
        return {
            "schema": "phala.shared-cache.snapshot.v1",
            "success": True,
            "case_id": "case-1",
            "epoch": "epoch-1",
            "tenant_id": "default",
            "requested_tenant_id": "default",
            "effective_tenant_id": "default",
            "master_multi_tenant_enabled": False,
            "sample_id": sample_id,
            "master_pid": 100,
            "sample_time_unix_ms": 1000,
            "backend_drain_available": False,
            "objects": [
                {
                    "key_id": key["key_id"],
                    "found": True,
                    "logical_bytes": 4096,
                    "writer_id": writer_hash,
                    "lease_expired": True,
                    "pending": pending,
                    "replicas": ([memory] if with_memory else []) + [ssd],
                }
            ],
        }

    class NativeStore:
        def __init__(self):
            self.calls = []

        def batch_replica_clear(self, keys, segment):
            self.calls.append((list(keys), segment))
            return list(keys)

    native = NativeStore()
    controller = SimpleNamespace(
        ack_backup_queue=Queue(),
        backup_queue=Queue(),
        ack_write_queue=[],
        write_queue=[],
        ack_load_queue=[],
        ack_prefetch_queue=Queue(),
        storage_backend=SimpleNamespace(
            store=native,
            shared_cache_store_instance_id="store-1",
        ),
    )
    scheduler = SimpleNamespace(
        decode_offload_manager=SimpleNamespace(
            cache_controller=controller,
            ongoing_offload={},
            ongoing_backup={},
            offload_inflight={},
        ),
        ps=SimpleNamespace(tp_size=1, dp_size=1, pp_size=1, attn_cp_size=1),
        server_args=SimpleNamespace(
            tp_size=1,
            dp_size=1,
            pp_size=1,
            dcp_size=1,
            attn_cp_size=1,
            nnodes=1,
        ),
        disaggregation_mode="decode",
        is_fully_idle=lambda: True,
    )
    evidence = {
        "case_id": "case-1",
        "epoch": "epoch-1",
        "tenant_id": "default",
        "requested_tenant_id": "default",
        "effective_tenant_id": "default",
        "master_multi_tenant_enabled": False,
        "backend_tag": "backend-1",
        "d_worker_id": "writer-1",
        "store_instance_id": "store-1",
        "master_pid": 100,
        "init_scan_completed_unix_ms": 900,
        "bucket_eviction_policy": "none",
        "disk_watermark_eviction": False,
        "sample_time_unix_ms": 1000,
        "ssd_owner_mappings": [
            {
                "owner_id": ssd_owner_hash,
                "store_instance_id": "c" * 64,
                "scope_id": ssd_scope_hash,
                "backend_path_id": "d" * 64,
                "pid": 200,
                "init_completed_unix_ms": 600,
                "scan_completed_unix_ms": 700,
            }
        ],
    }

    owner_sample = {
        "schema": "phala.shared-cache.owner-drain.v1",
        "kind": "owner_drain",
        "case_id": "case-1",
        "epoch": "epoch-1",
        "tenant_id": "default",
        "pid": 200,
        "owner_id": ssd_owner_hash,
        "store_instance_id": "c" * 64,
        "scope_id": ssd_scope_hash,
        "backend_path_id": "d" * 64,
        "owner_client_requested_tenant_id": "default",
        "coverage": "owner_global_bucket",
        "init_completed_unix_ms": 600,
        "scan_completed_unix_ms": 700,
        "activity_sequence": 8,
        "active": {
            field: 0
            for field in (
                "loads",
                "offloads",
                "promotions",
                "removes",
                "heartbeats",
                "rescans",
                "leased_read_buffers",
            )
        },
        "pending": {
            field: 0
            for field in (
                "backend_writes",
                "backend_evictions",
                "ungrouped_offloads",
                "read_guards",
            )
        },
        "bucket_count": 1,
        "covered_buckets": 1,
        "available": True,
        "consistent": True,
        "backend_initialized": True,
        "bucket_eviction_policy": "NONE",
        "disk_watermark_eviction": False,
        "metadata_resync_pending": False,
        "draining": False,
        "heartbeat_ok": True,
    }
    owner_records = [
        {
            **owner_sample,
            "owner_sample_sequence": sequence,
            "sample_time_unix_ms": sample_time,
        }
        for sequence, sample_time in ((10, 750), (11, 875), (12, 1000))
    ]
    owner_lines = [
        json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
        for item in owner_records
    ]
    owner_summary = {
        "kind": "capture_summary",
        "emitted": len(owner_records),
        "dropped": 0,
        "rejected": 0,
        "complete": True,
        "bytes": sum(len(line) + 1 for line in owner_lines),
    }
    owner_drain_path = tmp_path / "owner-drain.jsonl"
    owner_drain_path.write_bytes(
        b"".join(line + b"\n" for line in owner_lines)
        + json.dumps(owner_summary, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    owner_drain_path.chmod(0o600)

    def read_snapshot(sample_id):
        return snapshot(sample_id, with_memory=len(native.calls) == 0)

    receipt = execute_bounded_clear(
        scheduler=scheduler,
        native_store=native,
        manifest=manifest,
        snapshot_reader=read_snapshot,
        master_evidence=evidence,
        writer_id=writer_hash,
        segment_id=segment_hash,
        owner_drain_evidence_path=str(owner_drain_path),
        quiet_ms=250,
        duration_ms=1000,
        now_ms=lambda: 1000,
    )
    assert receipt["success"] is True
    assert native.calls == [(["exact-key"], "MEMORY-cold")]
    assert receipt["cleared_key_ids"] == [key["key_id"]]
