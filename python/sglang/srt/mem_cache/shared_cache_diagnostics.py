import hashlib
import hmac
import json
import logging
import re
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
_MAX_DURATION_MS = 300000
_MAX_CAPTURE_FAILURES = 1024


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
    ):
        if not self._capture_open():
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
                "room": int(room),
                "index_count": int(index_count),
                "online": bool(online),
                "sender_mode": str(sender_mode),
                "completed": bool(completed),
            }
        )
        self.flush_truncation()


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
)
