import hashlib
import hmac
import json
import logging
import math
import os
import re
import stat
import struct
import threading
import time
from functools import wraps

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_TRUNCATION_RESERVE_BYTES = 256
_MAX_EVENTS = 4096
_MAX_KEYS = 65536
_MAX_BYTES = 64 * 1024 * 1024
_MAX_KEYS_PER_EVENT = 256
_MAX_REQUESTS = 1024
_MAX_DURATION_MS = 900000
_MAX_CAPTURE_FAILURES = 1024
_SEED_MAX_KEYS = 256
_SEED_MAX_BYTES = 16 * 1024 * 1024 * 1024
_SEED_MAX_EVENTS = 1024
_SEED_MAX_DURATION_MS = 120000


def _swallow_capture_errors(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if not self.enabled:
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception:
            with self._lock:
                self._capture_failures = min(
                    self._capture_failures + 1, _MAX_CAPTURE_FAILURES
                )
            return None

    return wrapped


class SharedCacheDiagnostics:
    """Default-off, bounded event capture for shared-cache correctness tests."""

    def __init__(
        self,
        *,
        enabled=False,
        key_salt=None,
        case_id=None,
        epoch=None,
        key_ids=None,
        max_events=256,
        max_keys=2048,
        max_bytes=1048576,
        keys_per_event=32,
        max_requests=32,
        max_duration_ms=60000,
        reader_manifest=None,
        log=logger,
    ):
        safe_label = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
        self.enabled = bool(
            enabled
            and key_salt
            and len(key_salt.encode()) <= 256
            and case_id
            and safe_label.fullmatch(case_id)
            and epoch
            and safe_label.fullmatch(epoch)
            and key_ids
        )
        self._salt = key_salt.encode() if self.enabled else None
        self._case_id = case_id if self.enabled else None
        self._epoch = epoch if self.enabled else None
        self._max_events = min(_MAX_EVENTS, max(1, int(max_events)))
        self._max_keys = min(_MAX_KEYS, max(0, int(max_keys)))
        self._max_bytes = min(
            _MAX_BYTES,
            max(_TRUNCATION_RESERVE_BYTES, int(max_bytes)),
        )
        self._keys_per_event = min(_MAX_KEYS_PER_EVENT, max(1, int(keys_per_event)))
        self._max_requests = min(_MAX_REQUESTS, max(0, int(max_requests)))
        self._max_duration_s = (
            min(_MAX_DURATION_MS, max(1, int(max_duration_ms))) / 1000
        )
        self._key_ids = {
            value.strip().lower()
            for value in (key_ids or "")[: _MAX_KEYS * 65].split(",")[: self._max_keys]
            if re.fullmatch(r"[0-9a-fA-F]{64}", value.strip())
        }
        self._logger = log
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._events = 0
        self._keys = 0
        self._keys_hashed = 0
        self._requests = set()
        self._bytes = 0
        self._truncated = False
        self._truncation_logged = False
        self._dropped_events = 0
        self._dropped_keys = 0
        self._dropped_bytes = 0
        self._dropped_requests = 0
        self._capture_failures = 0
        # Fixed at process startup; never taken from a request or reread from env.
        self._reader_manifest = reader_manifest
        self._reader_attempted = False
        self._reader_lock = threading.Lock()
        self._reader_identity = None
        self._seed_source_request_ref = None

    @property
    def capture_failures(self):
        return self._capture_failures

    def record_capture_failure(self):
        try:
            with self._lock:
                self._capture_failures = min(
                    self._capture_failures + 1, _MAX_CAPTURE_FAILURES
                )
        except Exception:
            pass

    def _capture_open(self):
        if not self.enabled:
            return False
        with self._lock:
            if self._truncation_logged:
                return False
        if time.monotonic() - self._started_at <= self._max_duration_s:
            return True
        with self._lock:
            self._truncated = True
            self._dropped_events += 1
        self.flush_truncation()
        return False

    def _register_request(self, request_id, tenant_id):
        if len(str(request_id).encode()) > 256 or len(str(tenant_id).encode()) > 128:
            with self._lock:
                self._truncated = True
                self._dropped_requests += 1
            return None
        request_ref = self._request_id(request_id, tenant_id)
        with self._lock:
            if request_ref in self._requests:
                return request_ref
            if len(self._requests) >= self._max_requests:
                self._truncated = True
                self._dropped_requests += 1
                return None
            self._requests.add(request_ref)
        return request_ref

    def _stable_id(self, domain, tenant_id, value):
        fields = (
            str(self._case_id),
            str(self._epoch),
            str(tenant_id or "default"),
            str(value),
        )
        message = bytearray(domain)
        for field in fields:
            encoded = field.encode()
            message.extend(struct.pack(">I", len(encoded)))
            message.extend(encoded)
        return hmac.new(self._salt, message, hashlib.sha256).hexdigest()

    def _key_id(self, value, tenant_id):
        return self._stable_id(b"phala.shared-cache-key.v1\0", tenant_id, value)

    def key_id(self, value, tenant_id="default"):
        if not self.enabled:
            return None
        return self._key_id(value, tenant_id)

    def arm_from_manifest(self, path, *, reader_identity=None):
        """Load the sealed seed key allowlist once into the default-off collector."""
        try:
            document = _read_private_json(path, maximum=64 * 1024)
            if reader_identity is not None:
                if document.get("schema") != "phala.shared-cache.seed-manifest.v2":
                    return False
                if not re.fullmatch(
                    r"[0-9a-f]{64}", str(document.get("request_ref", ""))
                ):
                    return False
                if any(
                    not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", str(document.get(k, "")))
                    for k in ("case_id", "epoch")
                ):
                    return False
                if (
                    not isinstance(document.get("key_salt"), str)
                    or not 1 <= len(document["key_salt"].encode()) <= 256
                ):
                    return False
                if any(document.get(k) != v for k, v in reader_identity.items()):
                    return False
                duration = document.get("max_duration_ms")
                if (
                    type(duration) is not int
                    or not 1 <= duration <= _SEED_MAX_DURATION_MS
                ):
                    return False
                # Older sealed v2 producers have no absolute expiry. When present,
                # an expiry is an additional bound, never a replacement window.
                expiry = document.get("expires_at")
                if expiry is not None and (
                    type(expiry) not in (int, float)
                    or not math.isfinite(expiry)
                    or not time.time() < expiry
                ):
                    return False
            if document.get("schema") not in (
                "phala.shared-cache.seed-manifest.v1",
                "phala.shared-cache.seed-manifest.v2",
            ):
                return False
            if document["schema"].endswith(".v2") and not _multi_seed_evidence_valid(
                document
            ):
                return False
            keys = document.get("keys")
            if not isinstance(keys, list) or not keys or len(keys) > _SEED_MAX_KEYS:
                return False
            salt = document.get("key_salt")
            case_id = document.get("case_id")
            epoch = document.get("epoch")
            tenant_id = document.get("tenant_id")
            if not salt or not case_id or not epoch or not tenant_id:
                return False
            unsigned = {
                key: value
                for key, value in document.items()
                if key != "manifest_sha256"
            }
            calculated_digest = hashlib.sha256(
                json.dumps(
                    unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode()
            ).hexdigest()
            if document.get("manifest_sha256") != calculated_digest:
                return False
            calculated = {
                _shared_cache_id(
                    salt.encode(),
                    b"phala.shared-cache-key.v1\0",
                    case_id,
                    epoch,
                    tenant_id,
                    key.get("key"),
                )
                for key in keys
                if isinstance(key, dict) and isinstance(key.get("key"), str)
            }
            declared = {item.get("key_id") for item in keys if isinstance(item, dict)}
            if len(calculated) != len(keys) or declared != calculated:
                return False
            with self._lock:
                if self.enabled:
                    return False
                self._salt = salt.encode()
                self._case_id = case_id
                self._epoch = epoch
                self._key_ids = declared
                self._max_events = min(
                    _MAX_EVENTS, max(1, int(document.get("max_events", 256)))
                )
                self._max_keys = min(_MAX_KEYS, max(1, len(keys)))
                self._max_bytes = min(
                    _MAX_BYTES,
                    max(
                        _TRUNCATION_RESERVE_BYTES,
                        int(document.get("max_total_logical_bytes", _MAX_BYTES)),
                    ),
                )
                self._max_requests = 1
                if reader_identity is not None:
                    self._max_requests = 32
                    self._max_keys = min(_MAX_KEYS, len(keys) * 32 * 4)
                    self._max_bytes = min(_MAX_BYTES, 1048576)
                    self._max_events = min(_MAX_EVENTS, 1024)
                    self._reader_identity = dict(reader_identity)
                    self._seed_source_request_ref = document["request_ref"]
                self._max_duration_s = (
                    min(
                        _MAX_DURATION_MS,
                        max(1, int(document.get("max_duration_ms", 60000))),
                    )
                    / 1000
                )
                self._started_at = time.monotonic()
                if reader_identity is not None and expiry is not None:
                    self._max_duration_s = min(
                        self._max_duration_s, max(0, expiry - time.time())
                    )
                self.enabled = True
            return True
        except Exception:
            return False

    def reader_context(self, operation, controller):
        """One-shot late arm at actual queued hybrid prefetch, before any GET.

        The returned context belongs to this operation, including after cancellation;
        no process/thread-local current-request state crosses the IO queues.
        """
        if not self._reader_manifest:
            return None
        try:
            handle = operation.handle
            if type(getattr(handle, "bootstrap_room", None)) is not int:
                return None
            trusted_ref = _trusted_reader_ref(
                getattr(handle, "pd_diagnostic_request_ref", None)
            )
            if not trusted_ref:
                return None
            with self._reader_lock:
                if not self._reader_attempted:
                    # The donor may prefetch before D publishes the seed. Only
                    # absence leaves the fixed-path arm opportunity pending.
                    try:
                        os.lstat(self._reader_manifest)
                    except FileNotFoundError:
                        return None
                    except OSError:
                        self._reader_attempted = True
                        return None
                    self._reader_attempted = True
                    store = controller.storage_backend
                    config = controller.storage_config
                    schema = controller.mem_pool_host.storage_schema
                    if config.tp_size != 1 or config.pp_size != 1:
                        return None
                    transfers = operation.pool_transfers or []
                    if {str(x.name) for x in transfers} != {
                        str(name) for name in store.registered_pools
                    }:
                        return None
                    components = []
                    for transfer in transfers:
                        _, multiplier = store._get_hybrid_page_component_keys(
                            operation.hash_value[:1], transfer
                        )
                        components.extend(
                            f"{transfer.name}:{i}" for i in range(multiplier)
                        )
                    identity = {
                        "backend_tag": config.extra_config["extra_backend_tag"],
                        "model_revision": schema["revision"],
                        "kv_schema": hashlib.sha256(
                            json.dumps(
                                schema, sort_keys=True, separators=(",", ":")
                            ).encode()
                        ).hexdigest(),
                        "required_components": sorted(components),
                        "tenant_id": store.config.tenant_id,
                        "rank": config.tp_rank,
                    }
                    if not self.arm_from_manifest(
                        self._reader_manifest, reader_identity=identity
                    ):
                        return None
                    self.record_schema(schema)
                    self._emit(
                        {
                            "event": "reader_capture_armed",
                            "kv_schema": identity["kv_schema"],
                            "storage_schema": schema,
                            "seed_source_request_ref": self._seed_source_request_ref,
                            "max_duration_ms": int(self._max_duration_s * 1000),
                            "max_requests": self._max_requests,
                            "allowlist_keys": len(self._key_ids),
                            "max_key_observations": self._max_keys,
                            "max_events": self._max_events,
                            "max_log_bytes": self._max_bytes,
                        }
                    )
                if self._reader_identity is None or not self._capture_open():
                    return None
                tenant = controller.storage_backend.config.tenant_id
                request_ref = self._register_request(handle.rid, tenant)
                if request_ref is None:
                    self.flush_truncation()
                    return None
                return {
                    "request_id": request_ref,
                    **trusted_ref,
                    "attempt_id": self._operation_id(
                        f"{handle.rid}:{handle.attempt_id}", tenant
                    ),
                    "operation_id": self._operation_id(operation.id, tenant),
                    "room": handle.bootstrap_room,
                    "kv_schema": self._reader_identity["kv_schema"],
                    "seed_source_request_ref": self._seed_source_request_ref,
                }
        except Exception:
            self.record_capture_failure()
            return None

    def _request_id(self, value, tenant_id="default"):
        return self._stable_id(b"phala.shared-cache-request.v1\0", tenant_id, value)

    def _operation_id(self, value, tenant_id="default"):
        return self._stable_id(b"phala.shared-cache-operation.v1\0", tenant_id, value)

    def _emit(self, event, *, key_count=0):
        if not self._capture_open():
            return
        event.setdefault("case_id", self._case_id)
        event.setdefault("epoch", self._epoch)
        encoded = json.dumps(event, separators=(",", ":"), sort_keys=True)
        encoded_size = len(encoded.encode())
        with self._lock:
            if self._events >= self._max_events - 1:
                self._truncated = True
                self._dropped_events += 1
                self._dropped_keys += key_count
                return
            if self._keys + key_count > self._max_keys:
                self._truncated = True
                self._dropped_keys += key_count
                return
            if self._bytes + encoded_size > self._max_bytes - _TRUNCATION_RESERVE_BYTES:
                self._truncated = True
                self._dropped_bytes += encoded_size
                return
            self._logger.info("shared_cache_diag %s", encoded)
            self._events += 1
            self._keys += key_count
            self._bytes += encoded_size

    @_swallow_capture_errors
    def flush_truncation(self):
        if not self.enabled:
            return
        with self._lock:
            if not self._truncated or self._truncation_logged:
                return
            event = {
                "event": "capture_truncated",
                "dropped_bytes": self._dropped_bytes,
                "dropped_events": self._dropped_events,
                "dropped_keys": self._dropped_keys,
                "dropped_requests": self._dropped_requests,
            }
            encoded = json.dumps(event, separators=(",", ":"), sort_keys=True)
            if self._events >= self._max_events:
                return
            self._logger.warning("shared_cache_diag %s", encoded)
            self._events += 1
            self._bytes += len(encoded.encode())
            self._truncation_logged = True

    @_swallow_capture_errors
    def record_schema(self, schema):
        if not self._capture_open():
            return
        pools = []
        for item in schema.get("pools", ())[: self._keys_per_event]:
            if len(item) >= 6:
                pools.append(
                    {
                        "name": str(item[0]),
                        "page_size": int(item[1]),
                        "layout": str(item[2]),
                        "layer_num": int(item[3]),
                        "dtype": str(item[4]),
                    }
                )
        self._emit({"event": "registered_pools", "pools": pools}, key_count=len(pools))
        if len(schema.get("pools", ())) > len(pools):
            with self._lock:
                self._truncated = True
                self._dropped_keys += len(schema["pools"]) - len(pools)
        self.flush_truncation()

    @staticmethod
    def _size_bytes(size):
        if isinstance(size, (list, tuple)):
            return sum(int(part) for part in size)
        return int(size)

    @_swallow_capture_errors
    def record_io(
        self,
        *,
        is_set,
        pool,
        keys,
        sizes,
        results,
        exist_results=None,
        tenant_id="default",
        reader_context=None,
        reader_cancelled=False,
    ):
        if not self._capture_open():
            return
        if not is_set and self._reader_identity is not None and not reader_context:
            return
        limit = min(len(keys), self._keys_per_event)
        components = []
        filtered_keys = 0
        for index in range(limit):
            if len(keys[index].encode()) > 4096 or len(str(tenant_id).encode()) > 128:
                filtered_keys += 1
                continue
            with self._lock:
                if self._keys_hashed >= self._max_keys:
                    self._truncated = True
                    self._dropped_keys += len(keys) - index
                    break
                self._keys_hashed += 1
            key_id = self._key_id(keys[index], tenant_id)
            if key_id not in self._key_ids:
                continue
            if index >= len(sizes) or index >= len(results):
                filtered_keys += 1
                continue
            size = self._size_bytes(sizes[index])
            result = int(results[index])
            component = {
                "key_id": key_id,
                "requested_bytes": size,
                "result": result,
            }
            if is_set:
                existed = (
                    exist_results is not None
                    and index < len(exist_results)
                    and exist_results[index] == 1
                )
                component["already_present"] = bool(existed)
                component["written"] = bool(not existed and result == 0)
            else:
                component["read_bytes"] = max(0, result)
                component["short_read"] = 0 < result < size
                component["read_complete"] = result == size and size > 0
            components.append(component)
        truncated = len(keys) > limit or self._keys_hashed >= self._max_keys
        truncated = truncated or filtered_keys > 0
        if components or filtered_keys:
            self._emit(
                {
                    "event": "put_components" if is_set else "get_components",
                    "pool": str(pool),
                    "components": components,
                    "filtered_keys": filtered_keys,
                    "truncated": truncated,
                    **(
                        dict(reader_context, reader_cancelled=bool(reader_cancelled))
                        if not is_set and reader_context
                        else {}
                    ),
                },
                key_count=len(components),
            )
        if truncated:
            with self._lock:
                self._truncated = True
                self._dropped_keys += max(0, len(keys) - limit)
                self._dropped_keys += filtered_keys
        self.flush_truncation()

    @_swallow_capture_errors
    def record_backup(
        self,
        *,
        phase,
        request_id,
        operation_id,
        complete,
        tokens,
        tenant_id="default",
    ):
        if not self._capture_open():
            return
        request_ref = self._register_request(request_id, tenant_id)
        if request_ref is None:
            self.flush_truncation()
            return
        self._emit(
            {
                "event": "backup_" + phase,
                "request_id": request_ref,
                "operation_id": self._operation_id(operation_id, tenant_id),
                "complete": bool(complete),
                "tokens": int(tokens),
            }
        )
        self.flush_truncation()

    @_swallow_capture_errors
    def record_prefetch(
        self,
        *,
        request_id,
        requested_tokens,
        completed_tokens,
        accepted,
        tenant_id="default",
        reader_context=None,
    ):
        if not self._capture_open():
            return
        request_ref = self._register_request(request_id, tenant_id)
        if request_ref is None:
            self.flush_truncation()
            return
        self._emit(
            {
                "event": "prefetch_boundary",
                "request_id": request_ref,
                "requested_tokens": int(requested_tokens),
                "completed_tokens": int(completed_tokens),
                "accepted": bool(accepted),
                **(reader_context or {}),
            }
        )
        self.flush_truncation()

    @_swallow_capture_errors
    def record_prefill_forward(
        self,
        *,
        request_id,
        h_tokens,
        n_tokens,
        forward_start,
        forward_end,
        tail_complete,
        tenant_id="default",
    ):
        if not self._capture_open():
            return
        request_ref = self._register_request(request_id, tenant_id)
        if request_ref is None:
            self.flush_truncation()
            return
        self._emit(
            {
                "event": "prefill_forward_complete",
                "request_id": request_ref,
                "h_tokens": int(h_tokens),
                "n_tokens": int(n_tokens),
                "suffix_tokens": max(0, int(n_tokens) - int(h_tokens)),
                "forward_start": int(forward_start),
                "forward_end": int(forward_end),
                "forward_tokens": int(forward_end) - int(forward_start),
                "tail_complete": bool(tail_complete),
            }
        )
        self.flush_truncation()

    @_swallow_capture_errors
    def record_c128_transfer(
        self,
        *,
        request_id,
        room,
        index_count,
        online,
        sender_mode,
        completed,
        tenant_id="default",
        reader_request_ref=None,
    ):
        if not self._capture_open():
            return
        request_ref = self._register_request(request_id, tenant_id)
        if request_ref is None:
            self.flush_truncation()
            return
        self._emit(
            {
                "event": "c128_transfer",
                "request_id": request_ref,
                **_trusted_reader_ref(reader_request_ref),
                "room": int(room),
                "index_count": int(index_count),
                "online": bool(online),
                "sender_mode": str(sender_mode),
                "completed": bool(completed),
            }
        )
        self.flush_truncation()


def _trusted_reader_ref(value):
    # The scheduler only copies this from the existing authenticated PD field.
    return (
        {"reader_request_ref": value}
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
        else {}
    )


def _read_private_json(path, *, maximum):
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ValueError("private regular file required")
    if metadata.st_size > maximum:
        raise ValueError("private file too large")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (metadata.st_dev, metadata.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("private file changed")
        if hasattr(os, "getuid") and opened.st_uid != os.getuid():
            raise ValueError("private file owner mismatch")
        with os.fdopen(fd, "rb", closefd=False) as source:
            raw = source.read(maximum + 1)
    finally:
        os.close(fd)
    if len(raw) > maximum:
        raise ValueError("private file too large")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("private object required")
    return value


def _shared_cache_id(salt, domain, case_id, epoch, tenant_id, value):
    message = bytearray(domain)
    for field in (case_id, epoch, tenant_id, value):
        encoded = str(field).encode("utf-8")
        message.extend(struct.pack(">I", len(encoded)))
        message.extend(encoded)
    return hmac.new(salt, message, hashlib.sha256).hexdigest()


def _write_all(fd, payload):
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _seed_two_range_plan_valid(document):
    ranges = document.get("operation_ranges")
    union = document.get("page_range")
    if not isinstance(ranges, list) or len(ranges) != 2:
        return False
    for value in [union, *ranges]:
        if (
            not isinstance(value, dict)
            or set(value) != {"start", "end"}
            or type(value["start"]) is not int
            or type(value["end"]) is not int
            or not 0 <= value["start"] < value["end"]
        ):
            return False
    return (
        union["start"] == ranges[0]["start"]
        and ranges[0]["end"] == ranges[1]["start"]
        and ranges[1]["end"] == union["end"]
    )


def _multi_seed_evidence_valid(document):
    """Verify each real ACK and its exact key product before consuming v2."""
    if not _seed_two_range_plan_valid(document):
        return False
    components = document.get("required_components")
    keys = document.get("keys")
    puts = document.get("put_results")
    operations = document.get("operations")
    if (
        not isinstance(components, list)
        or not components
        or len(components) > 64
        or not all(isinstance(value, str) for value in components)
        or components != sorted(set(components))
        or not isinstance(keys, list)
        or not 1 <= len(keys) <= _SEED_MAX_KEYS
        or not isinstance(puts, list)
        or len(puts) != len(keys)
        or not isinstance(operations, list)
        or len(operations) != 2
        or "operation_id" in document
        or "backup_ack" in document
        or "model_revision" not in document
        or document["model_revision"] is not None
        and not isinstance(document["model_revision"], str)
    ):
        return False
    union = document["page_range"]
    cardinality = (union["end"] - union["start"]) * len(components)
    if cardinality > _SEED_MAX_KEYS or cardinality != len(keys):
        return False
    ids, key_ids, page_components = set(), set(), set()
    for operation, expected_range in zip(operations, document["operation_ranges"]):
        if not isinstance(operation, dict):
            return False
        expected_cardinality = (expected_range["end"] - expected_range["start"]) * len(
            components
        )
        if expected_cardinality > len(keys):
            return False
        operation_id = operation.get("operation_id")
        expected_tokens = operation.get("expected_tokens")
        declared_keys = operation.get("key_ids")
        if (
            type(operation_id) is not int
            or operation_id in ids
            or operation.get("page_range") != expected_range
            or type(expected_tokens) is not int
            or expected_tokens <= 0
            or type(operation.get("tokens")) is not int
            or operation["tokens"] != expected_tokens
            or operation.get("complete") is not True
            or not isinstance(declared_keys, list)
            or not all(isinstance(value, str) for value in declared_keys)
            or declared_keys != sorted(set(declared_keys))
        ):
            return False
        ids.add(operation_id)
        owned_keys = []
        owned_product = set()
        for key in keys:
            if not isinstance(key, dict):
                return False
            if type(key.get("operation_id")) is not int:
                return False
            if key.get("operation_id") != operation_id:
                continue
            key_id = key.get("key_id")
            page, component = key.get("page_index"), key.get("component")
            size = key.get("logical_bytes")
            if (
                not isinstance(key_id, str)
                or key_id in key_ids
                or type(page) is not int
                or not expected_range["start"] <= page < expected_range["end"]
                or component not in components
                or (page, component) in page_components
                or type(size) is not int
                or size <= 0
                or type(key.get("rank")) is not int
                or key["rank"] != 0
            ):
                return False
            key_ids.add(key_id)
            page_components.add((page, component))
            owned_keys.append(key_id)
            owned_product.add((page, component))
        if len(owned_keys) != expected_cardinality:
            return False
        if sorted(owned_keys) != declared_keys or owned_product != {
            (page, component)
            for page in range(expected_range["start"], expected_range["end"])
            for component in components
        }:
            return False
    if len(key_ids) != len(keys):
        return False
    put_ids = set()
    for put in puts:
        if not isinstance(put, dict):
            return False
        key_id = put.get("key_id")
        if (
            not isinstance(key_id, str)
            or key_id not in key_ids
            or key_id in put_ids
            or type(put.get("native_result")) is not int
            or put["native_result"] != 0
            or put.get("already_present") is not False
        ):
            return False
        put_ids.add(key_id)
    return put_ids == key_ids


class SharedCacheSeedCapture:
    """One-request producer for exact D-writer keys and successful backup ACK."""

    def __init__(self, *, log=logger):
        self._log = log
        self._lock = threading.Lock()
        self._config = None
        self._operation_id = None
        self._operations = {}
        self._entries = {}
        self._events = 0
        self._total_bytes = 0
        self._failed = None
        self._fd = None
        self._provisional_path = None
        self._sealed_path = None
        self._started = None
        self._sealed = False

    def arm(
        self,
        selectors,
        *,
        actual_request_id,
        actual_store_instance_id,
        actual_d_worker_id,
        actual_tenant_id,
    ):
        path = os.getenv("SGLANG_SHARED_CACHE_SEED_CAPTURE_CONFIG")
        if not path:
            return False
        with self._lock:
            if self._config is not None:
                if (
                    self._sealed
                    or self._failed
                    or self._config["schema"] != "phala.shared-cache.seed-config.v2"
                    or actual_request_id != self._config["request_id"]
                ):
                    return False
                if self._deadline_expired_locked():
                    return False
                if (
                    actual_store_instance_id != self._config["store_instance_id"]
                    or actual_d_worker_id != self._config["d_worker_id"]
                    or actual_tenant_id != self._config["tenant_id"]
                    or any(
                        self._config.get(key) != value
                        for key, value in selectors.items()
                        if key != "page_range"
                    )
                ):
                    self._failed = "operation_identity_mismatch"
                    self._write_provisional_locked()
                    return False
                selected_range = selectors.get("page_range")
                return selected_range in self._config["operation_ranges"]
            try:
                config = _read_private_json(path, maximum=64 * 1024)
                if config.get("schema") not in (
                    "phala.shared-cache.seed-config.v1",
                    "phala.shared-cache.seed-config.v2",
                ):
                    return False
                multi = config["schema"].endswith(".v2")
                if any(
                    config.get(key) != value
                    for key, value in selectors.items()
                    if not multi or key != "page_range"
                ):
                    return False
                if multi and (
                    not _seed_two_range_plan_valid(config)
                    or selectors.get("page_range") != config["operation_ranges"][0]
                    or "model_revision" not in config
                ):
                    return False
                if (
                    not isinstance(actual_request_id, str)
                    or not 1 <= len(actual_request_id) <= 256
                ):
                    return False
                if config.get("tenant_id") != actual_tenant_id:
                    return False
                request_ref = selectors.get("request_ref")
                page_range = selectors.get("page_range")
                rank = selectors.get("rank")
                components = selectors.get("required_components")
                if (
                    not isinstance(request_ref, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", request_ref)
                    or not isinstance(page_range, dict)
                    or type(page_range.get("start")) is not int
                    or type(page_range.get("end")) is not int
                    or page_range["start"] < 0
                    or page_range["end"] <= page_range["start"]
                    or type(rank) is not int
                    or rank != 0
                    or not isinstance(components, list)
                    or not components
                    or components != sorted(set(components))
                ):
                    return False
                if config.get("model_revision") is not None and not isinstance(
                    config.get("model_revision"), str
                ):
                    return False
                config["store_instance_id"] = actual_store_instance_id
                config["d_worker_id"] = actual_d_worker_id
                config["request_id"] = actual_request_id
                if not multi:
                    config["page_range"] = page_range
                config["rank"] = rank
                if not all(
                    isinstance(config.get(key), str) and config[key]
                    for key in (
                        "run_id",
                        "case_id",
                        "epoch",
                        "request_id",
                        "tenant_id",
                        "backend_tag",
                        "kv_schema",
                        "d_worker_id",
                        "store_instance_id",
                        "key_salt",
                        "output_dir",
                    )
                ):
                    return False
                max_keys = int(config.get("max_keys", 0))
                max_bytes = int(config.get("max_logical_bytes", 0))
                max_events = int(config.get("max_events", 0))
                max_duration = int(config.get("max_duration_ms", 0))
                max_artifact = int(config.get("max_artifact_bytes", 0))
                required = config.get("required_components")
                if not (
                    1 <= max_keys <= _SEED_MAX_KEYS
                    and 1 <= max_bytes <= _SEED_MAX_BYTES
                    and 1 <= max_events <= _SEED_MAX_EVENTS
                    and 1 <= max_duration <= _SEED_MAX_DURATION_MS
                    and 1024 <= max_artifact <= 64 * 1024
                    and isinstance(required, list)
                    and required
                    and len(required) <= 64
                    and all(
                        isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9._:-]{1,96}", x)
                        for x in required
                    )
                ):
                    return False
                if multi and (
                    (config["page_range"]["end"] - config["page_range"]["start"])
                    * len(required)
                    > max_keys
                ):
                    return False
                directory = config["output_dir"]
                dir_stat = os.lstat(directory)
                if (
                    not stat.S_ISDIR(dir_stat.st_mode)
                    or stat.S_IMODE(dir_stat.st_mode) != 0o700
                ):
                    return False
                if hasattr(os, "getuid") and dir_stat.st_uid != os.getuid():
                    return False
                seed_id = config.get("seed_id")
                if not isinstance(seed_id, str) or not re.fullmatch(
                    r"[A-Za-z0-9._-]{1,64}", seed_id
                ):
                    return False
                provisional = os.path.join(directory, f"{seed_id}.provisional.json")
                sealed = os.path.join(directory, f"{seed_id}.seed.json")
                if os.path.lexists(provisional) or os.path.lexists(sealed):
                    return False
                flags = (
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                )
                self._fd = os.open(provisional, flags, 0o600)
                self._config = config
                self._provisional_path = provisional
                self._sealed_path = sealed
                self._started = time.monotonic()
                self._write_provisional_locked()
                if self._failed:
                    os.close(self._fd)
                    self._fd = None
                    self._config = None
                    return False
                return True
            except Exception:
                if self._fd is not None:
                    os.close(self._fd)
                    self._fd = None
                self._config = None
                return False

    def _deadline_expired_locked(self):
        if (
            time.monotonic() - self._started
            > int(self._config["max_duration_ms"]) / 1000
        ):
            self._failed = self._failed or "duration_exceeded"
            self._write_provisional_locked()
            return True
        return False

    def bind_operation(
        self, request_id, operation_id, *, page_range=None, expected_tokens=None
    ):
        with self._lock:
            if (
                self._sealed
                or self._failed
                or self._config is None
                or request_id != self._config["request_id"]
            ):
                return False
            if self._deadline_expired_locked():
                return False
            if self._config["schema"].endswith(".v2"):
                if (
                    type(operation_id) is not int
                    or operation_id in self._operations
                    or page_range not in self._config["operation_ranges"]
                    or any(
                        operation["page_range"] == page_range
                        for operation in self._operations.values()
                    )
                    or len(self._operations) >= 2
                    or type(expected_tokens) is not int
                    or expected_tokens <= 0
                ):
                    self._failed = "invalid_or_duplicate_operation"
                    self._write_provisional_locked()
                    return False
                self._operations[operation_id] = dict(
                    operation_id=operation_id,
                    page_range=dict(page_range),
                    expected_tokens=expected_tokens,
                    tokens=None,
                    complete=False,
                )
                self._operation_id = operation_id
                self._write_provisional_locked()
                return not self._failed
            if self._operation_id is not None:
                self._failed = "multiple_backup_operations"
                return False
            self._operation_id = int(operation_id)
            self._write_provisional_locked()
            return True

    def fail_closed(self, reason="capture_hook_error"):
        try:
            with self._lock:
                if (
                    not self._sealed
                    and self._config is not None
                    and self._failed is None
                ):
                    self._failed = reason
                    self._write_provisional_locked()
        except Exception:
            return

    def prepare_batch(self, context, *, pool, component_names, keys, sizes):
        if not isinstance(context, dict):
            return None
        with self._lock:
            if self._sealed or self._config is None or self._failed:
                return None
            request_id = context.get(
                "request_id", context.get("shared_cache_diag_request_id")
            )
            operation_id = context.get(
                "operation_id", context.get("shared_cache_diag_operation_id")
            )
            multi = self._config["schema"].endswith(".v2")
            operation = self._operations.get(operation_id) if multi else None
            if request_id != self._config["request_id"] or (
                operation is None if multi else operation_id != self._operation_id
            ):
                return None
            if self._operation_id is None:
                self._failed = "operation_not_bound"
                return None
            if self._deadline_expired_locked():
                return None
            if multi and operation["complete"]:
                self._failed = "record_after_operation_ack"
                self._write_provisional_locked()
                return None
            if len(keys) != len(sizes) or len(keys) != len(component_names):
                self._failed = "batch_shape_mismatch"
                return None
            if self._events >= int(self._config["max_events"]):
                self._failed = "event_budget_exceeded"
                return None
            pending = []
            pending_key_ids = set()
            accounted_bytes = (
                sum(item["logical_bytes"] for item in self._entries.values())
                if multi
                else self._total_bytes
            )
            start = int(context.get("page_start", 0))
            for index, (key, size, component) in enumerate(
                zip(keys, sizes, component_names)
            ):
                if not isinstance(key, str) or not key or len(key.encode()) > 4096:
                    self._failed = "invalid_key"
                    return None
                size = self._size_bytes(size)
                if size <= 0:
                    self._failed = "invalid_size"
                    return None
                key_id = self._key_id(key)
                if key_id in self._entries or key_id in pending_key_ids:
                    self._failed = "duplicate_key"
                    return None
                if component not in self._config["required_components"]:
                    self._failed = "unexpected_component"
                    return None
                if len(self._entries) + len(pending) >= int(self._config["max_keys"]):
                    self._failed = "key_budget_exceeded"
                    return None
                if accounted_bytes + sum(
                    item["logical_bytes"] for item in pending
                ) + size > int(self._config["max_logical_bytes"]):
                    self._failed = "byte_budget_exceeded"
                    return None
                page_indexes = context.get("page_indexes")
                page_index = (
                    page_indexes[index]
                    if isinstance(page_indexes, list) and index < len(page_indexes)
                    else start + index // max(1, int(context.get("key_multiplier", 1)))
                )
                page_range = (
                    operation["page_range"] if multi else self._config["page_range"]
                )
                if not page_range["start"] <= page_index < page_range["end"]:
                    self._failed = "page_outside_selected_range"
                    return None
                if multi and any(
                    item["page_index"] == page_index and item["component"] == component
                    for item in [*self._entries.values(), *pending]
                ):
                    self._failed = "duplicate_page_component"
                    return None
                pending.append(
                    {
                        "key_id": key_id,
                        "key": key,
                        "rank": 0,
                        "component": component,
                        "page_index": page_index,
                        "logical_bytes": size,
                        "native_result": None,
                        "already_present": None,
                    }
                )
                if multi:
                    pending[-1]["operation_id"] = operation_id
                pending_key_ids.add(key_id)
            for entry in pending:
                self._entries[entry["key_id"]] = entry
            self._events += 1
            self._write_provisional_locked()
            return [entry["key_id"] for entry in pending]

    def complete_batch(self, token, *, exists, results):
        if token is None:
            return
        with self._lock:
            if self._sealed or self._failed or self._config is None:
                return
            if self._deadline_expired_locked():
                return
            if len(token) != len(exists) or len(token) != len(results):
                self._failed = "result_shape_mismatch"
                return
            for key_id, existed, result in zip(token, exists, results):
                entry = self._entries.get(key_id)
                if entry is None:
                    self._failed = "prepared_key_lost"
                    return
                if self._config["schema"].endswith(".v2") and (
                    entry["native_result"] is not None
                    or self._operations[entry["operation_id"]]["complete"]
                ):
                    self._failed = "duplicate_or_late_put_result"
                    self._write_provisional_locked()
                    return
                if self._config["schema"].endswith(".v2") and (
                    type(existed) is not int
                    or existed != 0
                    or type(result) is not int
                    or result != 0
                ):
                    self._failed = "put_not_new_success"
                    self._write_provisional_locked()
                    return
                entry["already_present"] = existed == 1
                entry["native_result"] = int(result)
                if existed == 1 or int(result) != 0:
                    self._failed = "put_not_new_success"
            self._total_bytes = sum(
                item["logical_bytes"] for item in self._entries.values()
            )
            self._write_provisional_locked()

    def backup_ack(
        self, *, request_id, operation_id, complete, tokens, expected_tokens
    ):
        with self._lock:
            if (
                self._sealed
                or self._config is None
                or request_id != self._config["request_id"]
            ):
                return False
            if self._config["schema"].endswith(".v2"):
                return self._backup_ack_v2_locked(
                    operation_id=operation_id,
                    complete=complete,
                    tokens=tokens,
                    expected_tokens=expected_tokens,
                )
            if (
                time.monotonic() - self._started
                > int(self._config["max_duration_ms"]) / 1000
            ):
                self._failed = self._failed or "duration_exceeded"
                self._write_provisional_locked()
                return False
            if (
                operation_id != self._operation_id
                or not complete
                or tokens != expected_tokens
            ):
                self._failed = self._failed or "backup_incomplete"
                self._write_provisional_locked()
                return False
            if self._failed:
                return False
            components = {item["component"] for item in self._entries.values()}
            if components != set(self._config["required_components"]):
                self._failed = "missing_component"
                self._write_provisional_locked()
                return False
            page_components = {
                (item["page_index"], item["component"])
                for item in self._entries.values()
            }
            page_range = self._config["page_range"]
            expected_page_components = {
                (page_index, component)
                for page_index in range(page_range["start"], page_range["end"])
                for component in self._config["required_components"]
            }
            if page_components != expected_page_components:
                self._failed = "incomplete_page_components"
                self._write_provisional_locked()
                return False
            if not self._entries or any(
                item["native_result"] != 0 or item["already_present"] is not False
                for item in self._entries.values()
            ):
                self._failed = "put_incomplete"
                self._write_provisional_locked()
                return False
            document = self._producer_document_locked()
            return self._atomic_seal_locked(document)

    def _backup_ack_v2_locked(self, *, operation_id, complete, tokens, expected_tokens):
        if self._failed or self._deadline_expired_locked():
            return False
        operation = self._operations.get(operation_id)
        if (
            type(operation_id) is not int
            or operation is None
            or operation["complete"]
            or complete is not True
            or type(tokens) is not int
            or type(expected_tokens) is not int
            or tokens != operation["expected_tokens"]
            or expected_tokens != operation["expected_tokens"]
        ):
            self._failed = "backup_incomplete_or_duplicate"
            self._write_provisional_locked()
            return False
        entries = [
            item
            for item in self._entries.values()
            if item["operation_id"] == operation_id
        ]
        page_range = operation["page_range"]
        product = {
            (page, component)
            for page in range(page_range["start"], page_range["end"])
            for component in self._config["required_components"]
        }
        if (
            len(entries) != len(product)
            or {(item["page_index"], item["component"]) for item in entries} != product
            or any(
                item["native_result"] != 0 or item["already_present"] is not False
                for item in entries
            )
        ):
            self._failed = "operation_put_or_components_incomplete"
            self._write_provisional_locked()
            return False
        operation.update(tokens=tokens, complete=True)
        self._write_provisional_locked()
        if len(self._operations) != 2 or not all(
            item["complete"] for item in self._operations.values()
        ):
            return False
        document = self._producer_document_locked()
        if self._failed or not _multi_seed_evidence_valid(document):
            self._failed = self._failed or "multi_operation_evidence_invalid"
            self._write_provisional_locked()
            return False
        if self._deadline_expired_locked():
            return False
        return self._atomic_seal_locked(document)

    def _key_id(self, key):
        config = self._config
        return _shared_cache_id(
            config["key_salt"].encode(),
            b"phala.shared-cache-key.v1\0",
            config["case_id"],
            config["epoch"],
            config["tenant_id"],
            key,
        )

    @property
    def sealed_path(self):
        return self._sealed_path

    @staticmethod
    def _size_bytes(size):
        if isinstance(size, (list, tuple)):
            return sum(int(part) for part in size)
        return int(size)

    def _producer_document_locked(self):
        config = self._config
        entries = sorted(
            self._entries.values(),
            key=lambda item: (
                item["rank"],
                item["component"],
                item["page_index"],
                item["key_id"],
            ),
        )
        doc = {
            "schema": "phala.shared-cache.seed-manifest.v1",
            "run_id": config["run_id"],
            "seed_id": config["seed_id"],
            "case_id": config["case_id"],
            "epoch": config["epoch"],
            "tenant_id": config["tenant_id"],
            "backend_tag": config["backend_tag"],
            "model_revision": config["model_revision"],
            "kv_schema": config["kv_schema"],
            "d_worker_id": config["d_worker_id"],
            "store_instance_id": config["store_instance_id"],
            "request_id": config["request_id"],
            "request_ref": config["request_ref"],
            "operation_id": self._operation_id,
            "rank": config["rank"],
            "page_range": config["page_range"],
            "key_salt": config["key_salt"],
            "required_components": config["required_components"],
            "max_snapshots": config.get("max_snapshots", 4),
            "max_duration_ms": config.get("snapshot_max_duration_ms", 120000),
            "max_response_bytes": config.get("max_response_bytes", 1048576),
            "max_total_logical_bytes": config["max_logical_bytes"],
            "key_count": len(entries),
            "logical_bytes": self._total_bytes,
            "keys": [
                {
                    "key_id": item["key_id"],
                    "key": item["key"],
                    "rank": item["rank"],
                    "component": item["component"],
                    "page_index": item["page_index"],
                    "logical_bytes": item["logical_bytes"],
                }
                for item in entries
            ],
            "put_results": [
                {
                    "key_id": item["key_id"],
                    "native_result": item["native_result"],
                    "already_present": item["already_present"],
                }
                for item in entries
            ],
            "backup_ack": {"operation_id": self._operation_id, "complete": True},
        }
        if config["schema"].endswith(".v2"):
            doc["schema"] = "phala.shared-cache.seed-manifest.v2"
            doc.pop("operation_id")
            doc.pop("backup_ack")
            doc["operation_ranges"] = config["operation_ranges"]
            doc["operations"] = [
                dict(
                    operation,
                    key_ids=sorted(
                        item["key_id"]
                        for item in entries
                        if item["operation_id"] == operation["operation_id"]
                    ),
                )
                for page_range in config["operation_ranges"]
                for operation in self._operations.values()
                if operation["page_range"] == page_range
            ]
            for key, entry in zip(doc["keys"], entries):
                key["operation_id"] = entry["operation_id"]
        canonical = json.dumps(
            doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        doc["manifest_sha256"] = __import__("hashlib").sha256(canonical).hexdigest()
        return doc

    def _write_provisional_locked(self):
        if self._fd is None:
            return
        doc = {
            "schema": "phala.shared-cache.seed-provisional.v1",
            "request_id": self._config["request_id"],
            "operation_id": self._operation_id,
            "failed": self._failed,
            "events": self._events,
            "entries": list(self._entries.values()),
        }
        if self._config["schema"].endswith(".v2"):
            doc["schema"] = "phala.shared-cache.seed-provisional.v2"
            doc.pop("operation_id")
            doc["operation_ranges"] = self._config["operation_ranges"]
            doc["operations"] = list(self._operations.values())
        raw = json.dumps(
            doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        if len(raw) > int(self._config["max_artifact_bytes"]):
            self._failed = "artifact_budget_exceeded"
            return
        try:
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.ftruncate(self._fd, 0)
            _write_all(self._fd, raw)
            os.fsync(self._fd)
        except OSError:
            self._failed = "artifact_write_failed"

    def _atomic_seal_locked(self, document):
        raw = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        if len(raw) > int(self._config["max_artifact_bytes"]):
            self._failed = "artifact_budget_exceeded"
            self._write_provisional_locked()
            return False
        temp_path = self._sealed_path + ".tmp"
        if os.path.lexists(temp_path) or os.path.lexists(self._sealed_path):
            self._failed = "artifact_exists"
            self._write_provisional_locked()
            return False
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temp_path, flags, 0o600)
        try:
            _write_all(fd, raw)
            os.fsync(fd)
        finally:
            os.close(fd)
        if self._config["schema"].endswith(".v2") and self._deadline_expired_locked():
            os.unlink(temp_path)
            return False
        try:
            os.link(temp_path, self._sealed_path, follow_symlinks=False)
            os.unlink(temp_path)
            dir_fd = os.open(self._config["output_dir"], os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            self._failed = "seal_failed"
            self._write_provisional_locked()
            return False
        os.close(self._fd)
        self._fd = None
        self._sealed = True
        return True


shared_cache_seed_capture = SharedCacheSeedCapture()


shared_cache_diagnostics = SharedCacheDiagnostics(
    enabled=envs.SGLANG_MOONCAKE_SHARED_CACHE_DIAGNOSTICS.get(),
    key_salt=envs.SGLANG_MOONCAKE_SHARED_CACHE_DIAGNOSTICS_KEY_SALT.get(),
    case_id=envs.SGLANG_MOONCAKE_SHARED_CACHE_DIAGNOSTICS_CASE_ID.get(),
    epoch=envs.SGLANG_MOONCAKE_SHARED_CACHE_DIAGNOSTICS_EPOCH.get(),
    key_ids=envs.SGLANG_MOONCAKE_SHARED_CACHE_DIAGNOSTICS_KEY_IDS.get(),
    max_events=envs.SGLANG_MOONCAKE_SHARED_CACHE_DIAGNOSTICS_MAX_EVENTS.get(),
    max_keys=envs.SGLANG_MOONCAKE_SHARED_CACHE_DIAGNOSTICS_MAX_KEYS.get(),
    max_bytes=envs.SGLANG_MOONCAKE_SHARED_CACHE_DIAGNOSTICS_MAX_BYTES.get(),
    keys_per_event=envs.SGLANG_MOONCAKE_SHARED_CACHE_DIAGNOSTICS_KEYS_PER_EVENT.get(),
    max_requests=envs.SGLANG_MOONCAKE_SHARED_CACHE_DIAGNOSTICS_MAX_REQUESTS.get(),
    max_duration_ms=envs.SGLANG_MOONCAKE_SHARED_CACHE_DIAGNOSTICS_MAX_DURATION_MS.get(),
    reader_manifest=os.getenv("SGLANG_SHARED_CACHE_READER_MANIFEST"),
)
