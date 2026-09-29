import time
from unittest.mock import patch

from sglang.srt.mem_cache.shared_cache_diagnostics import (
    SharedCacheSeedCapture,
    _write_all,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _capture():
    capture = SharedCacheSeedCapture()
    capture._config = {
        "request_id": "request-1",
        "case_id": "case-1",
        "epoch": "epoch-1",
        "tenant_id": "tenant-1",
        "key_salt": "salt",
        "max_duration_ms": 1000,
        "max_events": 4,
        "max_keys": 8,
        "max_logical_bytes": 1024,
        "required_components": ["kv:0"],
        "request_ref": "a" * 64,
        "page_range": {"start": 0, "end": 1},
        "rank": 0,
    }
    capture._started = time.monotonic()
    capture._operation_id = 7
    return capture


def test_duplicate_keys_inside_one_batch_fail_closed():
    capture = _capture()
    token = capture.prepare_batch(
        {"request_id": "request-1", "operation_id": 7},
        pool="kv",
        component_names=["kv:0", "kv:0"],
        keys=["same-key", "same-key"],
        sizes=[1, 1],
    )
    assert token is None
    assert capture._failed == "duplicate_key"
    assert capture._entries == {}


def test_backup_ack_after_deadline_cannot_seal():
    capture = _capture()
    capture._config["max_duration_ms"] = 1
    capture._started = time.monotonic() - 1
    assert not capture.backup_ack(
        request_id="request-1",
        operation_id=7,
        complete=True,
        tokens=1,
        expected_tokens=1,
    )
    assert capture._failed == "duration_exceeded"


def test_artifact_writer_retries_short_writes():
    chunks = []

    def short_write(_fd, data):
        size = min(2, len(data))
        chunks.append(bytes(data[:size]))
        return size

    with patch("sglang.srt.mem_cache.shared_cache_diagnostics.os.write", short_write):
        _write_all(1, b"abcdef")
    assert b"".join(chunks) == b"abcdef"
