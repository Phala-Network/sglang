"""Per-request evidence for explicitly bypassed shared-cache reads."""

import hashlib
import json
import logging
import threading
import uuid

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_unattributed_get_calls = 0


def _digest(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def record_unattributed_get():
    global _unattributed_get_calls
    with _lock:
        _unattributed_get_calls += 1


class ColdSharedReadTrace:
    def __init__(self, rid, extra_key, cache_salt, rank):
        self.epoch = uuid.uuid4().hex
        self.rid_hash = _digest(rid)
        self.namespace_hash = _digest((extra_key, cache_salt))
        self.rank = rank
        self._lock = threading.Lock()
        with _lock:
            self._unattributed_start = _unattributed_get_calls
        self._operations = 0
        self._get_inflight = 0
        self._get_calls = 0
        self._get_keys = 0
        self._returned_bytes = 0
        self._get_errors = 0
        self._terminal = None
        self._ended = False
        self._emit("begin")

    def _emit(self, event, **fields):
        logger.info(
            "cold_shared_read %s",
            json.dumps(
                dict(
                    event=event,
                    epoch=self.epoch,
                    rid_hash=self.rid_hash,
                    namespace_hash=self.namespace_hash,
                    rank=self.rank,
                    **fields,
                ),
                sort_keys=True,
            ),
        )

    def operation_begin(self):
        with self._lock:
            if self._terminal is not None:
                raise RuntimeError(
                    "cold shared-read operation issued after request terminal"
                )
            self._operations += 1

    def operation_end(self):
        with self._lock:
            self._operations -= 1
            if self._operations < 0:
                raise RuntimeError("cold shared-read operation accounting underflow")
            self._maybe_end_locked()

    def get_begin(self, key_count):
        with self._lock:
            if self._ended:
                self._emit("late_get_rejected")
                raise RuntimeError("cold shared-read GET after terminal summary")
            self._get_calls += 1
            self._get_keys += key_count
            self._get_inflight += 1

    def get_end(self, results=None, error=False):
        with self._lock:
            self._get_inflight -= 1
            self._get_errors += int(error)
            if results is not None:
                self._returned_bytes += sum(max(0, int(value)) for value in results)
            self._maybe_end_locked()

    def terminal(self, reason):
        with self._lock:
            if self._terminal is None:
                self._terminal = reason
            self._maybe_end_locked()

    def _maybe_end_locked(self):
        if (
            self._ended
            or self._terminal is None
            or self._operations
            or self._get_inflight
        ):
            return
        with _lock:
            unattributed = _unattributed_get_calls - self._unattributed_start
        self._ended = True
        self._emit(
            "end",
            terminal=self._terminal,
            get_calls=self._get_calls,
            get_keys=self._get_keys,
            returned_bytes=self._returned_bytes,
            get_errors=self._get_errors,
            unattributed_get_calls=unattributed,
            inflight=0,
            complete=unattributed == 0,
        )
