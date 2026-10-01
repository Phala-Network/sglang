"""Focused CPython regression against patched CommonKVManager.update_status."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import platform
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/disaggregation/common/conn.py"
BASE = ROOT / "python/sglang/srt/disaggregation/base/conn.py"


def extract(name, path, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if class_name:
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
        return next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


namespace = {"Optional": Optional}
nodes = [
    extract("KVPoll", BASE),
    extract("update_status", SOURCE, "CommonKVManager"),
    extract("check_status", SOURCE, "CommonKVManager"),
    extract("has_status", SOURCE, "CommonKVManager"),
    extract("clear_status", SOURCE, "CommonKVManager"),
    extract("get_status", SOURCE, "CommonKVManager"),
]
for node in nodes:
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
KVPoll = namespace["KVPoll"]


class Manager:
    update_status = namespace["update_status"]
    check_status = namespace["check_status"]
    has_status = namespace["has_status"]
    clear_status = namespace["clear_status"]
    get_status = namespace["get_status"]

    def __init__(self):
        self.request_status = {}
        self._status_lock = threading.Lock()


room = 8719684572819002400
manager = Manager()
read_complete = threading.Event()
resume = threading.Event()
errors = []


def trace(frame, event, arg):
    lock_body = nodes[1].body[0].body
    read_line = next(node.lineno for node in lock_body if isinstance(node, ast.Assign))
    if (
        frame.f_code is namespace["update_status"].__code__
        and event == "line"
        and frame.f_lineno == read_line
    ):
        read_complete.set()
        if not resume.wait(5):
            raise TimeoutError("interleaving rendezvous timed out")
    return trace


def sender():
    try:
        sys.settrace(trace)
        manager.update_status(room, KVPoll.Bootstrapping)
    except BaseException as exc:
        errors.append(type(exc).__name__)
    finally:
        sys.settrace(None)


thread = threading.Thread(target=sender)
thread.start()
assert read_complete.wait(5), "sender did not reach update_status"
metadata_done = threading.Event()


def metadata():
    manager.update_status(room, KVPoll.WaitingForInput)
    metadata_done.set()


metadata_thread = threading.Thread(target=metadata)
metadata_thread.start()
assert not metadata_done.wait(0.05), "metadata entered while sender held status lock"
resume.set()
thread.join(5)
metadata_thread.join(5)
assert not thread.is_alive() and not metadata_thread.is_alive() and not errors, errors
assert manager.check_status(room) == KVPoll.WaitingForInput

manager.update_status(room, KVPoll.Failed)
manager.update_status(room, KVPoll.Success)
assert manager.check_status(room) == KVPoll.Failed, "Failed must be terminal"
manager.clear_status(room)
manager.update_status(room, KVPoll.Success)
assert not manager.has_status(room), (
    "late terminal status must not resurrect a cleared room"
)
manager.update_status(room, KVPoll.WaitingForInput)
assert manager.check_status(room) == KVPoll.WaitingForInput, (
    "early metadata may create its room"
)

# A reader racing clear sees one locked snapshot (None after clear), never a
# membership-then-index KeyError.
read_values = []
manager.update_status(room, KVPoll.WaitingForInput)
reader = threading.Thread(target=lambda: read_values.append(manager.get_status(room)))
clearer = threading.Thread(target=lambda: manager.clear_status(room))
reader.start()
clearer.start()
reader.join(5)
clearer.join(5)
assert not reader.is_alive() and not clearer.is_alive()
assert read_values[0] in (KVPoll.WaitingForInput, None)
assert manager.get_status(room) is None

# Execute actual common/Mooncake method bodies, pausing just before their
# snapshot read so clear wins deterministically. Unrelated runtime services
# are fixture state; these are not full runtime imports or GPU tests.
MOONCAKE = ROOT / "python/sglang/srt/disaggregation/mooncake/conn.py"
namespace.update(
    logger=logging.getLogger("status-regression"),
    DisaggregationMode=SimpleNamespace(PREFILL="prefill"),
)
method_cases = [
    (
        "conclude_transfer",
        SOURCE,
        "CommonKVManager",
        dict(bootstrap_room=room, status=KVPoll.Failed),
    ),
    (
        "apply_prefill_status",
        SOURCE,
        "CommonKVManager",
        dict(bootstrap_room=room, status=KVPoll.Failed, prefill_rank=0),
    ),
    (
        "_handle_node_failure",
        SOURCE,
        "CommonKVManager",
        dict(failed_bootstrap_addr="peer"),
    ),
    (
        "add_transfer_request",
        MOONCAKE,
        "MooncakeKVManager",
        dict(
            bootstrap_room=room,
            kv_indices=[],
            index_slice=slice(None),
            is_last_chunk=False,
        ),
    ),
]
actual_method_controls = []
for method_name, method_source, cls_name, kwargs in method_cases:
    method_node = extract(method_name, method_source, cls_name)
    exec(
        compile(
            ast.Module(body=[method_node], type_ignores=[]), str(method_source), "exec"
        ),
        namespace,
    )
    method = namespace[method_name]
    instance = Manager()
    instance.update_status(room, KVPoll.WaitingForInput)
    instance.connection_lock = threading.Lock()
    instance.connection_pool = {}
    instance.prefill_info_table = {}
    instance.addr_to_rooms_tracker = {"peer": {room}}
    instance.disaggregation_mode = "prefill"
    instance.failure_records = {}
    instance.record_failure = lambda *args: (_ for _ in ()).throw(
        AssertionError("missing room recorded failure")
    )
    snapshot_entered, snapshot_resume = threading.Event(), threading.Event()
    method_errors = []

    def snapshot_trace(frame, event, arg):
        if frame.f_code is namespace["get_status"].__code__ and event == "call":
            snapshot_entered.set()
            assert snapshot_resume.wait(5)
        return snapshot_trace

    def invoke():
        try:
            sys.settrace(snapshot_trace)
            method(instance, **kwargs)
        except BaseException as exc:
            method_errors.append(repr(exc))
        finally:
            sys.settrace(None)

    worker = threading.Thread(target=invoke)
    worker.start()
    assert snapshot_entered.wait(5), method_name
    instance.clear_status(room)
    snapshot_resume.set()
    worker.join(5)
    assert not worker.is_alive() and not method_errors, (method_name, method_errors)
    assert instance.failure_records == {} and instance.get_status(room) is None
    actual_method_controls.append(method_name + "_clear_before_snapshot")

print(
    json.dumps(
        {
            "python": platform.python_version(),
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "forced_interleave": "serialized_by_status_lock",
            "final_after_interleave": "WaitingForInput",
            "failed_terminal": True,
            "late_after_clear_dropped": True,
            "early_metadata_creation": True,
            "clear_read_snapshot": True,
            "actual_method_clear_controls": actual_method_controls,
            "boundary": "offline forced rendezvous; no claim about natural incident frequency or attribution",
        },
        indent=2,
    )
)
