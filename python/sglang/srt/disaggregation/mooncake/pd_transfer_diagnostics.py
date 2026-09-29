"""Bounded request/room joins for the classic P/D native transfer API."""

import os
import re
import json
import stat
import threading
import time

from sglang.srt.environ import envs
from sglang.srt.mem_cache.shared_cache_diagnostics import (
    SharedCacheDiagnostics,
    _swallow_capture_errors,
)


class PDBatchDiagnostics(SharedCacheDiagnostics):
    def __init__(
        self, *, enabled=False, key_salt=None, case_id=None, epoch=None,
        request_ids=None, tenant_id="default", config_path=None, **limits,
    ):
        super().__init__(enabled=False, **limits)
        labels = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
        self._request_allowlist = {
            value.strip().lower()
            for value in (request_ids or "")[: 1024 * 65].split(",")[:1024]
            if re.fullmatch(r"[0-9a-fA-F]{64}", value.strip())
        }
        self.enabled = bool(
            enabled and key_salt and len(key_salt.encode()) <= 256
            and case_id and labels.fullmatch(case_id)
            and epoch and labels.fullmatch(epoch)
            and tenant_id and len(tenant_id.encode()) <= 128
            and self._request_allowlist
        )
        self._salt = key_salt.encode() if self.enabled else None
        self._case_id = case_id if self.enabled else None
        self._epoch = epoch if self.enabled else None
        self._tenant_id = tenant_id
        self._rooms = {}
        self._room_external = {}
        self._blocked_rooms = set()
        self._started_at = None
        self._config_path = config_path
        self._load_attempted = False

    def _ensure_armed(self):
        if self.enabled or self._load_attempted or not self._config_path:
            return self.enabled
        with self._lock:
            if self.enabled or self._load_attempted:
                return self.enabled
            try:
                if not os.path.isabs(self._config_path):
                    self._load_attempted = True
                    return False
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(self._config_path, flags)
            except FileNotFoundError:
                return False
            except OSError:
                self._load_attempted = True
                return False
            self._load_attempted = True
            try:
                with os.fdopen(fd, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or stat.S_IMODE(info.st_mode) != 0o600
                        or info.st_size > 65536
                        or os.path.islink(self._config_path)
                    ):
                        return False
                    data = stream.read(65537)
                    if len(data) > 65536:
                        return False
                    config = json.loads(data)
                allowed = {
                    "schema", "key_salt", "case_id", "epoch", "tenant_id",
                    "request_ids", "max_events", "max_bytes", "max_requests",
                    "max_duration_ms",
                }
                if (
                    not isinstance(config, dict) or set(config) - allowed
                    or config.get("schema") != "phala.pd-batch-capture.v1"
                    or not isinstance(config.get("request_ids"), list)
                    or not 1 <= len(config["request_ids"]) <= 1024
                    or any(not isinstance(item, str) for item in config["request_ids"])
                ):
                    return False
                bounds = {"max_events": (2, 4096), "max_bytes": (1024, 67108864),
                          "max_requests": (1, 1024), "max_duration_ms": (1, 300000)}
                for name, (minimum, maximum) in bounds.items():
                    if name in config and (
                        type(config[name]) is not int or not minimum <= config[name] <= maximum
                    ):
                        return False
                replacement = PDBatchDiagnostics(
                    enabled=True, key_salt=config.get("key_salt"),
                    case_id=config.get("case_id"), epoch=config.get("epoch"),
                    tenant_id=config.get("tenant_id", "default"),
                    request_ids=",".join(config["request_ids"]),
                    **{name: config[name] for name in allowed if name.startswith("max_") and name in config},
                    log=self._logger,
                )
                if not replacement.enabled:
                    return False
                for name in (
                    "enabled", "_salt", "_case_id", "_epoch", "_tenant_id",
                    "_request_allowlist", "_max_events", "_max_bytes",
                    "_max_requests", "_max_duration_s",
                ):
                    setattr(self, name, getattr(replacement, name))
                return True
            except Exception:
                return False

    def external_request_ref(self, request_id):
        if not isinstance(request_id, str) or not request_id or len(request_id.encode()) > 256:
            return None
        if not self._ensure_armed():
            return None
        ref = self._request_id(request_id, self._tenant_id)
        return ref if ref in self._request_allowlist else None

    def _start_capture(self, request_ref):
        if not isinstance(request_ref, str) or not re.fullmatch(r"[0-9a-f]{64}", request_ref):
            return False
        if not self._ensure_armed() or request_ref not in self._request_allowlist:
            return False
        with self._lock:
            if self._started_at is None:
                self._started_at = time.monotonic()
        return self._capture_open()

    def _capture_open(self):
        if self._started_at is None:
            return False
        return super()._capture_open()

    def _room_id(self, room):
        return self._stable_id(b"phala.pd-room.v1\0", self._tenant_id, room)

    @_swallow_capture_errors
    def bind_ingress(self, request_ref, rooms, role):
        if not self._start_capture(request_ref):
            return
        if not isinstance(rooms, list):
            rooms = [rooms]
        if len(rooms) > self._max_requests:
            self.record_capture_failure()
            return
        with self._lock:
            if request_ref not in self._requests and len(self._requests) >= self._max_requests:
                self._truncated = True
                self._dropped_requests += 1
                return
            self._requests.add(request_ref)
        for index, room in enumerate(rooms):
            if type(room) is not int or not 0 <= room < 2**64:
                continue
            self._emit({
                "event": "pd_ingress_bind", "engine": "pd",
                "request_id": request_ref, "room_id": self._room_id(room),
                "room_index": index, "role": role, "pid": os.getpid(),
                "capture_tenant": self._tenant_id,
            })
        self.flush_truncation()

    def bind_worker(self, request_id, room, role, rank, external_ref):
        if not self._start_capture(external_ref):
            return
        return self._bind_worker(request_id, room, role, rank, external_ref)

    @_swallow_capture_errors
    def _bind_worker(self, request_id, room, role, rank, external_ref):
        if type(room) is not int or not 0 <= room < 2**64:
            return
        request_ref = self._register_request(request_id, self._tenant_id)
        if request_ref is None:
            return
        with self._lock:
            if room in self._blocked_rooms:
                return
            previous = self._rooms.get(room)
            if previous is not None and previous != request_ref:
                # A reused room cannot adopt a previous request's attribution.
                self._rooms.pop(room, None)
                self._room_external.pop(room, None)
                self._blocked_rooms.add(room)
                self._capture_failures += 1
                return
            if room not in self._rooms and len(self._rooms) >= self._max_requests:
                self._truncated = True
                self._dropped_requests += 1
                return
            self._rooms[room] = request_ref
            self._room_external[room] = external_ref
        self._emit({
            "event": "pd_worker_bind", "engine": "pd",
            "internal_request_id": request_ref, "room_id": self._room_id(room),
            "request_id": external_ref,
            "role": role, "rank": rank, "pid": os.getpid(),
            "capture_tenant": self._tenant_id,
        })
        self.flush_truncation()

    def active_room(self, room):
        if type(room) is not int or not self._capture_open():
            return False
        with self._lock:
            return room in self._rooms and self._events < self._max_events - 1

    @_swallow_capture_errors
    def record_native(self, room, kind, lengths, record, rank):
        if not self.active_room(room):
            return
        attempts = record["attempts"]
        if not isinstance(attempts, list) or len(attempts) > 64:
            self.record_capture_failure()
            return
        known = {"nvlink_intraNode", "tcp", "rdma", "local", "nvlink", "shm"}
        sanitized = []
        for index, item in enumerate(attempts):
            selected = {}
            for name, count in item["selected_transports"].items():
                label = name if name in known else "unrecognized"
                selected[label] = selected.get(label, 0) + int(count)
            sanitized.append({
                "attempt": index, "task_count": int(item["task_count"]),
                "missing_transports": int(item["missing_transports"]),
                "selected_transports": selected,
                "terminal_status": item["terminal_status"]
                if item["terminal_status"] in {
                    "not_submitted", "waiting", "completed", "failed",
                    "timeout", "submit_failed", "deadline_exceeded",
                } else "unrecognized",
                "transferred_bytes": int(item["transferred_bytes"]),
            })
        submitted = sum(lengths)
        complete = bool(
            record["result"] == 0 and submitted > 0 and sanitized
            and not record["diagnostics_truncated"]
            and sanitized[-1]["terminal_status"] == "completed"
            and sanitized[-1]["transferred_bytes"] == submitted
        )
        nvlink_only = bool(complete and all(
            item["task_count"] > 0 and item["missing_transports"] == 0
            and item["selected_transports"] == {"nvlink_intraNode": item["task_count"]}
            for item in sanitized
        ))
        self._emit({
            "event": "pd_native_batch", "engine": "pd",
            "room_id": self._room_id(room),
            "request_id": self._room_external[room],
            "internal_request_id": self._rooms[room],
            "native_batch_id": self._operation_id(
                f"{os.getpid()}:{record['batch_sequence']}", self._tenant_id
            ),
            "kind": kind if kind in {"kv", "state", "aux"} else "other",
            "pid": os.getpid(), "worker": threading.get_native_id(), "rank": rank,
            "capture_tenant": self._tenant_id, "entry_count": len(lengths),
            "submitted_bytes": submitted, "native_result": int(record["result"]),
            "attempts": sanitized, "diagnostics_truncated": bool(record["diagnostics_truncated"]),
            "completed_with_exact_bytes": complete,
            "all_attempts_nvlink_intra": nvlink_only,
        })
        self.flush_truncation()

    @_swallow_capture_errors
    def native_unavailable(self, room):
        self.record_capture_failure()
        self._emit({
            "event": "pd_native_unavailable", "engine": "pd",
            "room_id": self._room_id(room), "pid": os.getpid(),
        })
        self.flush_truncation()


pd_batch_diagnostics = PDBatchDiagnostics(
    config_path=os.getenv("SGLANG_MOONCAKE_PD_DIAGNOSTICS_CONFIG")
    if envs.SGLANG_MOONCAKE_PD_TRANSFER_DIAGNOSTICS.get() else None,
)


def bind_pd_ingress(obj, raw_request, role):
    obj._pd_diagnostic_request_ref = None
    try:
        if raw_request is None:
            return
        ref = pd_batch_diagnostics.external_request_ref(raw_request.headers.get("x-request-id"))
        obj._pd_diagnostic_request_ref = ref
        if ref is None:
            return
        pd_batch_diagnostics.bind_ingress(
            ref,
            getattr(obj, "bootstrap_room", None),
            str(getattr(role, "value", role)),
        )
    except Exception:
        pd_batch_diagnostics.record_capture_failure()
