from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import struct
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

MAX_MANIFEST_BYTES = 64 * 1024
MAX_KEYS = 256
MAX_LOGICAL_BYTES = 16 * 1024 * 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_OWNER_DRAIN_BYTES = 1024 * 1024
MAX_OWNER_DRAIN_EVENTS = 4096
MAX_CONTROL_DURATION_MS = 120_000
_CLEAR_QUIET_MS = 250
_LABEL = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_HEX = re.compile(r"^[0-9a-f]{64}$")
_NATIVE_UUID_PAIR = re.compile(r"^([0-9]+)-([0-9]+)$")
_FORBIDDEN_KEY = re.compile(r"[*?\[\]\x00-\x1f\x7f,]")
_UINT64_MAX = (1 << 64) - 1
_PENDING_FIELDS = (
    "processing",
    "offload",
    "promotion",
    "promotion_candidate",
    "replication",
    "dynamic_replication",
)


class SharedCacheControlError(RuntimeError):
    def __init__(self, reason: str, *, unknown: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.unknown = unknown


def validate_clear_selectors(manifest_id, manifest_sha256, request_id=None) -> None:
    if not isinstance(manifest_id, str) or not _LABEL.fullmatch(manifest_id):
        raise SharedCacheControlError("invalid_manifest_selector")
    if not isinstance(manifest_sha256, str) or not _HEX.fullmatch(manifest_sha256):
        raise SharedCacheControlError("invalid_manifest_selector")
    if request_id is not None and (
        not isinstance(request_id, str) or not _LABEL.fullmatch(request_id)
    ):
        raise SharedCacheControlError("invalid_request_id")


def validate_single_decode_writer(topology: Any, mode: Any) -> None:
    mode_value = getattr(mode, "value", mode)
    if mode_value != "decode":
        raise SharedCacheControlError("decode_writer_required")
    required_one = (
        "tp_size",
        "dp_size",
        "pp_size",
        "dcp_size",
        "attn_cp_size",
        "nnodes",
    )
    if any(getattr(topology, name, None) != 1 for name in required_one):
        raise SharedCacheControlError("unsupported_topology")


def parse_clear_selector_body(body: bytes, *, maximum: int = 512) -> Dict[str, str]:
    if not isinstance(body, bytes) or len(body) > maximum:
        raise SharedCacheControlError("request_body_too_large")
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SharedCacheControlError("invalid_json") from None
    fields = {"manifest_id", "manifest_sha256", "request_id"}
    if not isinstance(value, dict) or set(value) != fields:
        raise SharedCacheControlError("invalid_selector_fields")
    validate_clear_selectors(
        value["manifest_id"], value["manifest_sha256"], value["request_id"]
    )
    return value


def _stable_id(
    salt: bytes, domain: bytes, case_id: str, epoch: str, tenant: str, value: str
) -> str:
    message = bytearray(domain)
    for field in (case_id, epoch, tenant, value):
        encoded = field.encode("utf-8")
        message.extend(struct.pack(">I", len(encoded)))
        message.extend(encoded)
    return hmac.new(salt, message, hashlib.sha256).hexdigest()


def _canonical_native_uuid_pair(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("native UUID pair must be a string")
    match = _NATIVE_UUID_PAIR.fullmatch(value)
    if match is None:
        raise ValueError("invalid native UUID pair")
    first, second = (int(part) for part in match.groups())
    if first > _UINT64_MAX or second > _UINT64_MAX:
        raise ValueError("native UUID pair component is out of range")
    return f"{first}-{second}"


def _private_json(path: str, *, maximum: int) -> Dict[str, Any]:
    file_path = Path(path)
    try:
        metadata = file_path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise SharedCacheControlError("private_file_required")
        if metadata.st_size > maximum:
            raise SharedCacheControlError("file_too_large")
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        fd = os.open(file_path, flags)
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise SharedCacheControlError("file_changed")
            if hasattr(os, "getuid") and opened.st_uid != os.getuid():
                raise SharedCacheControlError("wrong_file_owner")
            with os.fdopen(fd, "rb", closefd=False) as source:
                payload = source.read(maximum + 1)
        finally:
            os.close(fd)
    except SharedCacheControlError:
        raise
    except (OSError, ValueError):
        raise SharedCacheControlError("file_unavailable") from None
    if len(payload) > maximum:
        raise SharedCacheControlError("file_too_large")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SharedCacheControlError("invalid_json") from None
    if not isinstance(value, dict):
        raise SharedCacheControlError("invalid_document")
    return value


def _private_jsonl(path: str) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    file_path = Path(path)
    try:
        metadata = file_path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise SharedCacheControlError("owner_drain_private_file_required")
        if metadata.st_size > MAX_OWNER_DRAIN_BYTES:
            raise SharedCacheControlError("owner_drain_file_too_large")
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        fd = os.open(file_path, flags)
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise SharedCacheControlError("owner_drain_file_changed")
            if hasattr(os, "getuid") and opened.st_uid != os.getuid():
                raise SharedCacheControlError("owner_drain_wrong_file_owner")
            with os.fdopen(fd, "rb", closefd=False) as source:
                payload = source.read(MAX_OWNER_DRAIN_BYTES + 1)
        finally:
            os.close(fd)
    except SharedCacheControlError:
        raise
    except (OSError, ValueError):
        raise SharedCacheControlError("owner_drain_file_unavailable") from None
    if len(payload) > MAX_OWNER_DRAIN_BYTES or not payload.endswith(b"\n"):
        raise SharedCacheControlError("owner_drain_file_incomplete")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    lines = payload.splitlines(keepends=True)
    if not lines or len(lines) > MAX_OWNER_DRAIN_EVENTS + 1:
        raise SharedCacheControlError("owner_drain_event_count_invalid")
    records = []
    summary = None
    byte_count = 0
    for index, raw_line in enumerate(lines):
        if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n"):
            raise SharedCacheControlError("owner_drain_line_invalid")
        encoded = raw_line[:-1]
        if not encoded or len(encoded) > 4096:
            raise SharedCacheControlError("owner_drain_line_invalid")
        try:
            value = json.loads(encoded, object_pairs_hook=reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise SharedCacheControlError("owner_drain_invalid_jsonl") from None
        if not isinstance(value, dict):
            raise SharedCacheControlError("owner_drain_invalid_record")
        if value.get("kind") == "capture_summary":
            if index != len(lines) - 1 or summary is not None:
                raise SharedCacheControlError("owner_drain_summary_not_final")
            summary = value
            continue
        if summary is not None:
            raise SharedCacheControlError("owner_drain_record_after_summary")
        if value.get("schema") not in (
            "phala.shared-cache.owner-drain.v1",
            "phala.shared-cache.native.v1",
        ):
            raise SharedCacheControlError("owner_drain_schema_mismatch")
        records.append(value)
        byte_count += len(raw_line)
    if summary is None:
        raise SharedCacheControlError("owner_drain_summary_missing")
    for field in ("emitted", "dropped", "rejected", "bytes"):
        value = summary.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SharedCacheControlError("owner_drain_summary_invalid")
    if (
        summary.get("complete") is not True
        or summary["dropped"] != 0
        or summary["emitted"] != len(records)
        or summary["emitted"] > MAX_OWNER_DRAIN_EVENTS
        or summary["bytes"] != byte_count
    ):
        raise SharedCacheControlError("owner_drain_capture_incomplete")
    return records, summary


def validate_owner_drain(
    records: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    master_evidence: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    quiet_ms: int,
    now_ms: int,
    freshness_ms: int = 30_000,
) -> Dict[str, Any]:
    records = list(records)
    for field in ("emitted", "dropped", "rejected", "bytes"):
        value = summary.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SharedCacheControlError("owner_drain_summary_invalid")
    if (
        summary.get("complete") is not True
        or summary["dropped"] != 0
        or summary["emitted"] != len(records)
    ):
        raise SharedCacheControlError("owner_drain_capture_incomplete")
    mappings = master_evidence.get("ssd_owner_mappings")
    if not isinstance(mappings, list) or not mappings:
        raise SharedCacheControlError("ssd_owner_mappings_missing")

    expected_owners = {}
    expected_owner_ids = manifest.get("_expected_ssd_owner_ids", {})
    expected_scope_ids = manifest.get("_expected_ssd_scope_ids", {})
    for entry in manifest["keys"]:
        owner_id = expected_owner_ids.get(entry["ssd_owner_uuid"])
        scope_id = expected_scope_ids.get(entry["ssd_scope"])
        if not owner_id or not scope_id:
            raise SharedCacheControlError("manifest_ssd_identity_invalid")
        if owner_id in expected_owners and expected_owners[owner_id] != scope_id:
            raise SharedCacheControlError("manifest_ssd_owner_scope_conflict")
        expected_owners[owner_id] = scope_id

    mapping_by_owner = {}
    instance_ids = set()
    mapping_fields = {
        "owner_id",
        "store_instance_id",
        "scope_id",
        "backend_path_id",
        "pid",
        "init_completed_unix_ms",
        "scan_completed_unix_ms",
    }
    for item in mappings:
        if not isinstance(item, Mapping) or set(item) != mapping_fields:
            raise SharedCacheControlError("ssd_owner_mapping_invalid")
        owner_id = item["owner_id"]
        if (
            not isinstance(owner_id, str)
            or not _HEX.fullmatch(owner_id)
            or owner_id in mapping_by_owner
            or expected_owners.get(owner_id) != item["scope_id"]
        ):
            raise SharedCacheControlError("ssd_owner_mapping_identity_mismatch")
        for field in ("store_instance_id", "scope_id", "backend_path_id"):
            value = item[field]
            if not isinstance(value, str) or not _HEX.fullmatch(value):
                raise SharedCacheControlError("ssd_owner_mapping_identity_invalid")
        pid = item["pid"]
        initialized = item["init_completed_unix_ms"]
        scanned = item["scan_completed_unix_ms"]
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            or not isinstance(initialized, int)
            or isinstance(initialized, bool)
            or initialized <= 0
            or not isinstance(scanned, int)
            or isinstance(scanned, bool)
            or scanned <= 0
        ):
            raise SharedCacheControlError("ssd_owner_mapping_lifecycle_invalid")
        instance_id = item["store_instance_id"]
        if instance_id in instance_ids:
            raise SharedCacheControlError("ssd_owner_instance_not_unique")
        instance_ids.add(instance_id)
        mapping_by_owner[owner_id] = item
    if set(mapping_by_owner) != set(expected_owners):
        raise SharedCacheControlError("ssd_owner_mapping_set_mismatch")

    active_fields = (
        "loads",
        "offloads",
        "promotions",
        "removes",
        "heartbeats",
        "rescans",
        "leased_read_buffers",
    )
    pending_fields = (
        "backend_writes",
        "backend_evictions",
        "ungrouped_offloads",
        "read_guards",
    )
    by_owner = {owner_id: [] for owner_id in expected_owners}
    for sample in records:
        if sample.get("schema") == "phala.shared-cache.native.v1":
            if (
                sample.get("case_id") != manifest["case_id"]
                or sample.get("epoch") != manifest["epoch"]
            ):
                raise SharedCacheControlError("owner_drain_event_identity_mismatch")
            continue
        owner_id = sample.get("owner_id")
        mapping = mapping_by_owner.get(owner_id)
        active = sample.get("active")
        pending = sample.get("pending")
        sequence = sample.get("owner_sample_sequence")
        activity_sequence = sample.get("activity_sequence")
        sample_time = sample.get("sample_time_unix_ms")
        initialized = sample.get("init_completed_unix_ms")
        scanned = sample.get("scan_completed_unix_ms")
        bucket_count = sample.get("bucket_count")
        covered = sample.get("covered_buckets")
        if (
            sample.get("schema") != "phala.shared-cache.owner-drain.v1"
            or sample.get("kind") != "owner_drain"
            or sample.get("case_id") != manifest["case_id"]
            or sample.get("epoch") != manifest["epoch"]
            or sample.get("tenant_id") != manifest["tenant"]
            or sample.get("owner_client_requested_tenant_id") != manifest["tenant"]
            or sample.get("coverage") != "owner_global_bucket"
            or mapping is None
            or not isinstance(sample.get("pid"), int)
            or isinstance(sample.get("pid"), bool)
            or sample.get("scope_id") != mapping["scope_id"]
            or sample.get("store_instance_id") != mapping["store_instance_id"]
            or sample.get("backend_path_id") != mapping["backend_path_id"]
            or sample.get("pid") != mapping["pid"]
            or initialized != mapping["init_completed_unix_ms"]
            or scanned != mapping["scan_completed_unix_ms"]
            or not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence < 0
            or not isinstance(activity_sequence, int)
            or isinstance(activity_sequence, bool)
            or activity_sequence < 0
            or not isinstance(sample_time, int)
            or isinstance(sample_time, bool)
            or sample_time < scanned
            or sample_time < initialized
            or sample_time > now_ms + 1000
            or not isinstance(bucket_count, int)
            or isinstance(bucket_count, bool)
            or not 1 <= bucket_count <= 256
            or not isinstance(covered, int)
            or isinstance(covered, bool)
            or covered != bucket_count
            or not isinstance(active, Mapping)
            or not isinstance(pending, Mapping)
            or any(
                not isinstance(group.get(field), int)
                or isinstance(group.get(field), bool)
                or group[field] < 0
                for group, fields in (
                    (active, active_fields),
                    (pending, pending_fields),
                )
                for field in fields
            )
            or sample.get("available") is not True
            or sample.get("consistent") is not True
            or sample.get("backend_initialized") is not True
            or str(sample.get("bucket_eviction_policy", "")).upper() != "NONE"
            or sample.get("disk_watermark_eviction") is not False
            or sample.get("metadata_resync_pending") is not False
            or sample.get("draining") is not False
            or sample.get("heartbeat_ok") is not True
        ):
            raise SharedCacheControlError("owner_drain_sample_invalid")
        by_owner[owner_id].append(sample)

    if any(not samples for samples in by_owner.values()):
        raise SharedCacheControlError("owner_drain_sample_missing")
    quiet_intervals = []
    sample_counts = {}
    for owner_id, samples in by_owner.items():
        ordered = sorted(samples, key=lambda item: item["owner_sample_sequence"])
        if any(
            current["owner_sample_sequence"] != previous["owner_sample_sequence"] + 1
            or current["sample_time_unix_ms"] < previous["sample_time_unix_ms"]
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise SharedCacheControlError("owner_drain_sequence_discontinuous")
        last_time = ordered[-1]["sample_time_unix_ms"]
        if last_time > now_ms + 1000 or max(0, now_ms - last_time) > freshness_ms:
            raise SharedCacheControlError("owner_drain_sample_stale")
        zero_fields = (*active_fields, *pending_fields)
        run_start = None
        prior = None
        for sample in ordered:
            counts = sample["active"] | sample["pending"]
            is_zero = all(counts[field] == 0 for field in zero_fields)
            if not is_zero:
                run_start = None
            elif (
                prior is None
                or run_start is None
                or not all(
                    prior[group][field] == 0
                    for group, fields in (
                        ("active", active_fields),
                        ("pending", pending_fields),
                    )
                    for field in fields
                )
                or prior["activity_sequence"] != sample["activity_sequence"]
            ):
                run_start = sample["sample_time_unix_ms"]
            prior = sample
        if (
            run_start is None
            or ordered[-1]["sample_time_unix_ms"] - run_start < quiet_ms
        ):
            raise SharedCacheControlError("owner_drain_quiet_window_missing")
        quiet_intervals.append((run_start, last_time))
        sample_counts[owner_id] = len(ordered)

    common_start = max(start for start, _ in quiet_intervals)
    common_end = min(end for _, end in quiet_intervals)
    if common_end - common_start < quiet_ms or now_ms - common_end > freshness_ms:
        raise SharedCacheControlError("owner_drain_global_quiet_window_missing")
    return {
        "owner_count": len(mapping_by_owner),
        "sample_counts": sample_counts,
        "quiet_start_unix_ms": common_start,
        "quiet_end_unix_ms": common_end,
        "capture_summary": {
            "emitted": summary["emitted"],
            "rejected": summary["rejected"],
            "bytes": summary["bytes"],
        },
    }


def load_manifest(path: str, manifest_id: str, manifest_sha256: str) -> Dict[str, Any]:
    if not _LABEL.fullmatch(manifest_id or "") or not _HEX.fullmatch(
        manifest_sha256 or ""
    ):
        raise SharedCacheControlError("invalid_manifest_selector")
    manifest = _private_json(path, maximum=MAX_MANIFEST_BYTES)
    expected = manifest.get("manifest_sha256")
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    calculated = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    if not hmac.compare_digest(str(expected or ""), calculated):
        raise SharedCacheControlError("manifest_hash_mismatch")
    if not hmac.compare_digest(calculated, manifest_sha256):
        raise SharedCacheControlError("manifest_hash_mismatch")
    if (
        manifest.get("manifest_id") != manifest_id
        or manifest.get("manifest_version") != 1
    ):
        raise SharedCacheControlError("manifest_selector_mismatch")
    for field in (
        "run_id",
        "case_id",
        "epoch",
        "tenant",
        "backend_tag",
        "model_revision",
        "kv_schema",
    ):
        if not _LABEL.fullmatch(str(manifest.get(field) or "")):
            raise SharedCacheControlError("invalid_manifest_metadata")
    if manifest.get("exact_nonempty_MEMORY_segment") in (None, ""):
        raise SharedCacheControlError("empty_memory_segment")
    raw_segment = manifest["exact_nonempty_MEMORY_segment"]
    if not isinstance(raw_segment, str) or _FORBIDDEN_KEY.search(raw_segment):
        raise SharedCacheControlError("invalid_memory_segment")
    salt = manifest.get("key_salt")
    if not isinstance(salt, str) or not salt or len(salt.encode()) > 256:
        raise SharedCacheControlError("invalid_manifest_salt")
    keys = manifest.get("keys")
    components = manifest.get("required_components")
    if not isinstance(keys, list) or not 1 <= len(keys) <= MAX_KEYS:
        raise SharedCacheControlError("invalid_key_count")
    if not isinstance(components, list) or not components:
        raise SharedCacheControlError("invalid_component_set")
    if any(not _LABEL.fullmatch(str(item)) for item in components):
        raise SharedCacheControlError("invalid_component_set")
    total = 0
    seen = set()
    seen_components = set()
    context = (manifest["case_id"], manifest["epoch"], manifest["tenant"])
    for item in keys:
        if not isinstance(item, dict):
            raise SharedCacheControlError("invalid_key_entry")
        key = item.get("key")
        size = item.get("logical_bytes")
        component = item.get("component")
        if (
            not isinstance(key, str)
            or not key
            or len(key.encode()) > 4096
            or _FORBIDDEN_KEY.search(key)
        ):
            raise SharedCacheControlError("invalid_key")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise SharedCacheControlError("invalid_logical_bytes")
        if not _LABEL.fullmatch(str(component or "")) or component not in components:
            raise SharedCacheControlError("invalid_component")
        if item.get("rank") != 0:
            raise SharedCacheControlError("unsupported_topology")
        page_range = item.get("page_range")
        if not isinstance(page_range, str) or not _LABEL.fullmatch(page_range):
            raise SharedCacheControlError("invalid_page_range")
        key_id = _stable_id(
            salt.encode(), b"phala.shared-cache-key.v1\0", *context, key
        )
        if item.get("key_id") != key_id or key_id in seen:
            raise SharedCacheControlError("key_id_mismatch_or_duplicate")
        seen.add(key_id)
        seen_components.add(component)
        item["key_id"] = key_id
        replica_ids = item.get("ssd_replica_ids")
        if (
            not isinstance(replica_ids, list)
            or not replica_ids
            or any(
                not isinstance(replica_id, int) or isinstance(replica_id, bool)
                for replica_id in replica_ids
            )
            or len(set(replica_ids)) != len(replica_ids)
        ):
            raise SharedCacheControlError("invalid_ssd_replica_ids")
        for identity_field in ("ssd_owner_uuid", "ssd_scope"):
            if (
                not isinstance(item.get(identity_field), str)
                or not item[identity_field]
            ):
                raise SharedCacheControlError("missing_ssd_identity")
        total += size
        if total > MAX_LOGICAL_BYTES:
            raise SharedCacheControlError("logical_bytes_too_large")
    if seen_components != set(components):
        raise SharedCacheControlError("incomplete_components")
    for field in (
        "original_D_worker_id",
        "original_writer_client_id",
        "original_store_instance_id",
    ):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise SharedCacheControlError("missing_writer_identity")
    return manifest


def expected_identities(manifest: Mapping[str, Any]) -> tuple[str, str]:
    salt = manifest["key_salt"].encode()
    context = (manifest["case_id"], manifest["epoch"], manifest["tenant"])
    try:
        writer_uuid = _canonical_native_uuid_pair(manifest["original_writer_client_id"])
        owner_uuids = {
            item["ssd_owner_uuid"]: _canonical_native_uuid_pair(item["ssd_owner_uuid"])
            for item in manifest["keys"]
        }
    except (ValueError, TypeError, KeyError, AttributeError):
        raise SharedCacheControlError("invalid_writer_identity") from None
    writer_id = _stable_id(
        salt, b"phala.shared-cache-owner.v1\0", *context, writer_uuid
    )
    segment_id = _stable_id(
        salt,
        b"phala.shared-cache-segment.v1\0",
        *context,
        manifest["exact_nonempty_MEMORY_segment"],
    )
    manifest["_expected_writer_id"] = writer_id
    manifest["_expected_segment_id"] = segment_id
    manifest["_expected_ssd_owner_ids"] = {
        raw: _stable_id(salt, b"phala.shared-cache-owner.v1\0", *context, normalized)
        for raw, normalized in owner_uuids.items()
    }
    manifest["_expected_ssd_scope_ids"] = {
        item["ssd_scope"]: _stable_id(
            salt,
            b"phala.shared-cache-owner.v1\0",
            *context,
            item["ssd_scope"],
        )
        for item in manifest["keys"]
    }
    return writer_id, segment_id


def burn_once(manifest_path: str, manifest_sha256: str) -> None:
    marker = f"{manifest_path}.{manifest_sha256}.consumed"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(marker, flags, 0o600)
        with os.fdopen(fd, "w", encoding="ascii") as output:
            output.write("consumed\n")
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError:
        raise SharedCacheControlError("manifest_already_consumed") from None
    except OSError:
        raise SharedCacheControlError("cannot_consume_manifest") from None


def clear_from_configured_artifacts(
    scheduler: Any,
    *,
    manifest_id: str,
    manifest_sha256: str,
    snapshot_timeout_s: float = 3.0,
) -> Dict[str, Any]:
    validate_clear_selectors(manifest_id, manifest_sha256)
    manifest_path = os.getenv("SGLANG_SHARED_CACHE_CLEAR_MANIFEST")
    evidence_path = os.getenv("SGLANG_SHARED_CACHE_MASTER_EVIDENCE")
    owner_drain_path = os.getenv("SGLANG_SHARED_CACHE_OWNER_DRAIN_EVIDENCE")
    admin_url = os.getenv("SGLANG_SHARED_CACHE_NATIVE_ADMIN_URL")
    if not manifest_path or not evidence_path or not owner_drain_path or not admin_url:
        raise SharedCacheControlError("clear_control_not_configured")
    manifest = load_manifest(manifest_path, manifest_id, manifest_sha256)
    evidence = _private_json(evidence_path, maximum=MAX_MANIFEST_BYTES)
    if evidence.get("schema") != "phala.shared-cache.master-evidence.v1":
        raise SharedCacheControlError("master_evidence_schema_mismatch")
    writer_id, segment_id = expected_identities(manifest)
    burn_once(manifest_path, manifest_sha256)

    def read_snapshot(sample_id: str) -> Mapping[str, Any]:
        return read_native_snapshot(
            admin_url,
            manifest,
            sample_id,
            timeout=snapshot_timeout_s,
            max_response_bytes=MAX_RESPONSE_BYTES,
        )

    storage_backend = getattr(
        getattr(
            getattr(scheduler, "decode_offload_manager", None),
            "cache_controller",
            None,
        ),
        "storage_backend",
        None,
    )
    native_store = getattr(storage_backend, "store", None)
    if native_store is None:
        raise SharedCacheControlError("storage_backend_missing")
    return execute_bounded_clear(
        scheduler=scheduler,
        native_store=native_store,
        manifest=manifest,
        snapshot_reader=read_snapshot,
        master_evidence=evidence,
        writer_id=writer_id,
        segment_id=segment_id,
        owner_drain_evidence_path=owner_drain_path,
        quiet_ms=_CLEAR_QUIET_MS,
        duration_ms=30_000,
    )


def _safe_admin_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError:
        raise SharedCacheControlError("invalid_admin_url") from None
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or port is None
    ):
        raise SharedCacheControlError("invalid_admin_url")
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, "/batch_query_keys", "", "")
    )


def read_native_snapshot(
    admin_url: str,
    manifest: Mapping[str, Any],
    sample_id: str,
    *,
    timeout: float,
    max_response_bytes: int,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Dict[str, Any]:
    key_ids = [item["key_id"] for item in manifest["keys"]]
    query = urllib.parse.urlencode(
        {
            "finite_snapshot": "1",
            "tenant_id": manifest["tenant"],
            "case_id": manifest["case_id"],
            "epoch": manifest["epoch"],
            "sample_id": sample_id,
            "key_ids": ",".join(key_ids),
        }
    )
    request = urllib.request.Request(
        f"{_safe_admin_url(admin_url)}?{query}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with opener(request, timeout=timeout) as response:
            body = response.read(max_response_bytes + 1)
    except Exception:
        raise SharedCacheControlError("snapshot_unavailable") from None
    if len(body) > max_response_bytes or len(body) > MAX_RESPONSE_BYTES:
        raise SharedCacheControlError("snapshot_too_large")
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SharedCacheControlError("invalid_snapshot") from None
    if not isinstance(value, dict):
        raise SharedCacheControlError("invalid_snapshot")
    return value


def validate_snapshot(
    snapshot: Mapping[str, Any],
    manifest: Mapping[str, Any],
    sample_id: str,
    *,
    expected_writer_id: str,
    expected_segment_id: str,
    expect_memory: bool = True,
) -> Dict[str, Any]:
    tenant = manifest["tenant"]
    multi_tenant_enabled = snapshot.get("master_multi_tenant_enabled")
    if (
        snapshot.get("schema") != "phala.shared-cache.snapshot.v1"
        or snapshot.get("success") is not True
        or snapshot.get("case_id") != manifest["case_id"]
        or snapshot.get("epoch") != manifest["epoch"]
        or snapshot.get("tenant_id") != tenant
        or snapshot.get("requested_tenant_id") != tenant
        or snapshot.get("effective_tenant_id") != tenant
        or not isinstance(multi_tenant_enabled, bool)
        or (tenant != "default" and multi_tenant_enabled is not True)
        or snapshot.get("sample_id") != sample_id
        or not isinstance(snapshot.get("master_pid"), int)
        or not isinstance(snapshot.get("sample_time_unix_ms"), int)
        or snapshot.get("backend_drain_available") is not False
    ):
        raise SharedCacheControlError("snapshot_identity_mismatch")
    objects = snapshot.get("objects")
    if not isinstance(objects, list) or len(objects) != len(manifest["keys"]):
        raise SharedCacheControlError("snapshot_object_count_mismatch")
    by_id = {item.get("key_id"): item for item in objects if isinstance(item, dict)}
    expected_ids = {item["key_id"] for item in manifest["keys"]}
    if len(by_id) != len(objects) or set(by_id) != expected_ids:
        raise SharedCacheControlError("snapshot_key_set_mismatch")
    ack_objects = []
    for entry in manifest["keys"]:
        obj = by_id[entry["key_id"]]
        if (
            obj.get("found") is not True
            or obj.get("logical_bytes") != entry["logical_bytes"]
            or obj.get("writer_id") != expected_writer_id
            or obj.get("lease_expired") is not True
        ):
            raise SharedCacheControlError("object_precondition_failed")
        pending = obj.get("pending")
        if not isinstance(pending, dict) or any(
            pending.get(field) != 0 or isinstance(pending.get(field), bool)
            for field in _PENDING_FIELDS
        ):
            raise SharedCacheControlError("native_work_pending")
        replicas = obj.get("replicas")
        if not isinstance(replicas, list):
            raise SharedCacheControlError("replica_state_missing")
        memory_targets = []
        disk_targets = []
        for replica in replicas:
            if not isinstance(replica, dict):
                raise SharedCacheControlError("invalid_replica_state")
            kind = replica.get("type")
            if kind == "MEMORY" and expected_segment_id in replica.get(
                "segment_ids", []
            ):
                memory_targets.append(replica)
            if kind == "LOCAL_DISK" and replica.get("id") in entry["ssd_replica_ids"]:
                disk_targets.append(replica)
            if kind == "NoF":
                raise SharedCacheControlError("nof_replica_present")
        if expect_memory:
            if len(memory_targets) != 1:
                raise SharedCacheControlError("memory_target_mismatch")
            memory = memory_targets[0]
            if (
                memory.get("status") != "COMPLETE"
                or memory.get("readable") is not True
                or memory.get("refcount") != 0
            ):
                raise SharedCacheControlError("memory_target_not_clearable")
        elif memory_targets:
            raise SharedCacheControlError("memory_target_survived_clear")
        if {item.get("id") for item in disk_targets} != set(entry["ssd_replica_ids"]):
            raise SharedCacheControlError("ssd_replica_set_mismatch")
        for replica in disk_targets:
            if (
                replica.get("status") != "COMPLETE"
                or replica.get("readable") is not True
                or replica.get("owner_id")
                != manifest["_expected_ssd_owner_ids"].get(entry["ssd_owner_uuid"])
                or replica.get("scope_id")
                != manifest["_expected_ssd_scope_ids"].get(entry["ssd_scope"])
            ):
                raise SharedCacheControlError("ssd_replica_not_preserved")
        ack_objects.append(
            {
                "key_id": entry["key_id"],
                "logical_bytes": entry["logical_bytes"],
                "ssd_replica_ids": sorted(entry["ssd_replica_ids"]),
            }
        )
    return {
        "master_pid": snapshot["master_pid"],
        "sample_time_unix_ms": snapshot["sample_time_unix_ms"],
        "objects": ack_objects,
    }


def scheduler_drain_snapshot(scheduler: Any) -> Dict[str, int]:
    manager = getattr(scheduler, "decode_offload_manager", None)
    if manager is None:
        raise SharedCacheControlError("decode_offload_disabled")
    controller = getattr(manager, "cache_controller", None)
    if controller is None:
        raise SharedCacheControlError("cache_controller_missing")
    ack_backup = getattr(controller, "ack_backup_queue", None)
    backup = getattr(controller, "backup_queue", None)
    if ack_backup is None or backup is None:
        raise SharedCacheControlError("backup_queue_missing")
    queues = {
        "scheduler_busy": int(not bool(scheduler.is_fully_idle())),
        "ongoing_offload": len(getattr(manager, "ongoing_offload", {})),
        "ongoing_backup": len(getattr(manager, "ongoing_backup", {})),
        "offload_inflight": len(getattr(manager, "offload_inflight", {})),
        "ack_write": len(getattr(controller, "ack_write_queue", [])),
        "ack_backup": int(ack_backup.qsize()),
        "backup_queue": int(backup.qsize()),
        "write_queue": len(getattr(controller, "write_queue", [])),
        "ack_load": len(getattr(controller, "ack_load_queue", [])),
        "ack_prefetch": int(
            getattr(controller, "ack_prefetch_queue", None).qsize()
            if getattr(controller, "ack_prefetch_queue", None) is not None
            else -1
        ),
    }
    return queues


def require_stable_drain(
    scheduler: Any, *, quiet_ms: int, sample_ms: int = 50
) -> Dict[str, int]:
    if not 50 <= quiet_ms <= 5000:
        raise SharedCacheControlError("invalid_quiet_interval")
    deadline = time.monotonic() + quiet_ms / 1000
    previous = scheduler_drain_snapshot(scheduler)
    if any(value != 0 for value in previous.values()):
        raise SharedCacheControlError("scheduler_not_drained")
    while time.monotonic() < deadline:
        time.sleep(min(sample_ms / 1000, max(0.0, deadline - time.monotonic())))
        current = scheduler_drain_snapshot(scheduler)
        if current != previous or any(value != 0 for value in current.values()):
            raise SharedCacheControlError("scheduler_drain_changed")
    return previous


def execute_bounded_clear(
    *,
    scheduler: Any,
    native_store: Any,
    manifest: Mapping[str, Any],
    snapshot_reader: Callable[[str], Mapping[str, Any]],
    master_evidence: Mapping[str, Any],
    writer_id: str,
    segment_id: str,
    owner_drain_evidence_path: Optional[str] = None,
    quiet_ms: int = _CLEAR_QUIET_MS,
    duration_ms: int = 30_000,
    now_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
) -> Dict[str, Any]:
    if not 1000 <= duration_ms <= MAX_CONTROL_DURATION_MS:
        raise SharedCacheControlError("invalid_control_duration")
    if quiet_ms != _CLEAR_QUIET_MS:
        raise SharedCacheControlError("invalid_quiet_interval")
    validate_single_decode_writer(
        getattr(scheduler, "server_args", None),
        getattr(scheduler, "disaggregation_mode", None),
    )
    started = time.monotonic()
    case_id = manifest["case_id"]
    epoch = manifest["epoch"]
    storage_backend = getattr(
        getattr(
            getattr(scheduler, "decode_offload_manager", None),
            "cache_controller",
            None,
        ),
        "storage_backend",
        None,
    )
    if storage_backend is None:
        raise SharedCacheControlError("storage_backend_missing")
    if native_store is not getattr(storage_backend, "store", None):
        raise SharedCacheControlError("original_store_mismatch")
    if (
        getattr(storage_backend, "shared_cache_store_instance_id", None)
        != manifest["original_store_instance_id"]
    ):
        raise SharedCacheControlError("original_store_instance_mismatch")
    if (
        getattr(scheduler.ps, "tp_size", None) != 1
        or getattr(scheduler.ps, "dp_size", None) != 1
        or getattr(scheduler.ps, "pp_size", None) != 1
        or getattr(scheduler.ps, "attn_cp_size", None) != 1
    ):
        raise SharedCacheControlError("unsupported_topology")
    if not isinstance(master_evidence, Mapping):
        raise SharedCacheControlError("master_evidence_missing")
    init_scan_ms = master_evidence.get("init_scan_completed_unix_ms")
    master_sample_ms = master_evidence.get("sample_time_unix_ms")
    if (
        master_evidence.get("case_id") != case_id
        or master_evidence.get("epoch") != epoch
        or master_evidence.get("tenant_id") != manifest["tenant"]
        or master_evidence.get("requested_tenant_id") != manifest["tenant"]
        or master_evidence.get("effective_tenant_id") != manifest["tenant"]
        or not isinstance(master_evidence.get("master_multi_tenant_enabled"), bool)
        or (
            manifest["tenant"] != "default"
            and master_evidence.get("master_multi_tenant_enabled") is not True
        )
        or master_evidence.get("backend_tag") != manifest["backend_tag"]
        or master_evidence.get("d_worker_id") != manifest["original_D_worker_id"]
        or master_evidence.get("store_instance_id")
        != manifest["original_store_instance_id"]
        or not isinstance(master_evidence.get("master_pid"), int)
        or isinstance(master_evidence.get("master_pid"), bool)
        or not isinstance(init_scan_ms, int)
        or isinstance(init_scan_ms, bool)
        or init_scan_ms <= 0
        or not isinstance(master_sample_ms, int)
        or isinstance(master_sample_ms, bool)
        or master_sample_ms < init_scan_ms
        or master_evidence.get("bucket_eviction_policy") != "none"
        or master_evidence.get("disk_watermark_eviction") is not False
        or now_ms() - master_sample_ms > 30_000
    ):
        raise SharedCacheControlError("master_evidence_invalid")
    before_sample_id = uuid.uuid4().hex
    before = snapshot_reader(before_sample_id)
    before_ack = validate_snapshot(
        before,
        manifest,
        before_sample_id,
        expected_writer_id=writer_id,
        expected_segment_id=segment_id,
    )
    if before_ack["master_pid"] != master_evidence["master_pid"]:
        raise SharedCacheControlError("master_instance_mismatch")
    drain = require_stable_drain(scheduler, quiet_ms=quiet_ms)
    if not owner_drain_evidence_path:
        raise SharedCacheControlError("owner_drain_evidence_missing")
    owner_records, owner_summary = _private_jsonl(owner_drain_evidence_path)
    owner_drain = validate_owner_drain(
        owner_records,
        manifest,
        master_evidence,
        summary=owner_summary,
        quiet_ms=quiet_ms,
        now_ms=now_ms(),
    )
    keys = [item["key"] for item in manifest["keys"]]
    memory_segment = manifest["exact_nonempty_MEMORY_segment"]
    remaining = duration_ms - int((time.monotonic() - started) * 1000)
    if remaining <= 0:
        raise SharedCacheControlError("control_deadline_before_clear")
    result_holder: Dict[str, Any] = {}

    def clear_once():
        try:
            result_holder["result"] = native_store.batch_replica_clear(
                keys, memory_segment
            )
        except Exception:
            result_holder["error"] = True

    clear_thread = threading.Thread(
        target=clear_once, name="shared-cache-clear", daemon=True
    )
    clear_thread.start()
    clear_thread.join(remaining / 1000)
    if clear_thread.is_alive():
        scheduler._shared_cache_clear_unknown = True
        raise SharedCacheControlError("clear_result_unknown", unknown=True)
    if "error" in result_holder:
        scheduler._shared_cache_clear_unknown = True
        raise SharedCacheControlError("clear_result_unknown", unknown=True)
    cleared = result_holder.get("result")
    if not isinstance(cleared, Iterable) or isinstance(cleared, (str, bytes)):
        scheduler._shared_cache_clear_unknown = True
        raise SharedCacheControlError("clear_result_unknown", unknown=True)
    cleared_list = list(cleared)
    if len(cleared_list) != len(set(cleared_list)) or set(cleared_list) != set(keys):
        scheduler._shared_cache_clear_unknown = True
        raise SharedCacheControlError("partial_clear_result", unknown=True)
    if (time.monotonic() - started) * 1000 > duration_ms:
        scheduler._shared_cache_clear_unknown = True
        raise SharedCacheControlError("clear_deadline_exceeded", unknown=True)
    after_sample_id = uuid.uuid4().hex
    after = snapshot_reader(after_sample_id)
    after_ack = validate_snapshot(
        after,
        manifest,
        after_sample_id,
        expected_writer_id=writer_id,
        expected_segment_id=segment_id,
        expect_memory=False,
    )
    if (
        after_ack["master_pid"] != before_ack["master_pid"]
        or after_ack["objects"] != before_ack["objects"]
        or scheduler_drain_snapshot(scheduler) != drain
    ):
        scheduler._shared_cache_clear_unknown = True
        raise SharedCacheControlError("postcondition_failed", unknown=True)
    return {
        "success": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "requested_key_ids": [item["key_id"] for item in manifest["keys"]],
        "cleared_key_ids": [item["key_id"] for item in manifest["keys"]],
        "key_count": len(keys),
        "logical_bytes": sum(item["logical_bytes"] for item in manifest["keys"]),
        "memory_segment_id": segment_id,
        "master_pid": before_ack["master_pid"],
        "sample_before_unix_ms": before_ack["sample_time_unix_ms"],
        "sample_after_unix_ms": after_ack["sample_time_unix_ms"],
        "drain": drain,
        "owner_drain": owner_drain,
        "completed_at_unix_ms": now_ms(),
    }
