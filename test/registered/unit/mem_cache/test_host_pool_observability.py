"""CPU-only tests of the production serializer with real torch storages."""

import ast
import importlib.util
import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MODULE_PATH = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/observability/host_pool.py"
)
spec = importlib.util.spec_from_file_location("host_pool_observability", MODULE_PATH)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


class DeepSeekV4PagedHostPool:
    """Lightweight known-pool metadata; all tensor/storage methods are real."""

    def __init__(self, buffer, *, capacity=32, free=24):
        self.kv_buffer = buffer
        self.size = self.logical_size = capacity
        self.free = free
        self.page_size = 4
        self.layout = "layer_first"
        self.item_bytes = 16
        self.pin_memory = False
        self.lock = threading.RLock()

    def available_size(self):
        return self.free


def group(**pools):
    return SimpleNamespace(
        entries=[
            SimpleNamespace(name=name, host_pool=pool) for name, pool in pools.items()
        ]
    )


def scheduler(primary=None, writeback=None):
    return SimpleNamespace(
        tree_cache=SimpleNamespace(host_pool_group=primary),
        decode_offload_manager=SimpleNamespace(decode_host_mem_pool=writeback),
        tp_rank=0,
        pp_rank=0,
        dp_rank=0,
    )


class TestHostPoolObservability(unittest.TestCase):
    def test_real_views_group_and_worker_alias_dedup(self):
        buffer = torch.arange(64, dtype=torch.float32)
        pool = DeepSeekV4PagedHostPool(buffer)
        pool.v_buffer = buffer[4:20:2]
        writeback_alias = DeepSeekV4PagedHostPool(buffer[16:32])
        independent = DeepSeekV4PagedHostPool(torch.zeros(8, dtype=torch.uint8))
        report = observer.host_pool_observability(
            scheduler(group(c4=pool), group(c4=writeback_alias, state=independent)),
            role="decode",
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["unique_backing_bytes"], 264)
        self.assertEqual(report["logical_capacity_sum"], 96)
        self.assertEqual(
            [g["unique_backing_bytes"] for g in report["groups"]], [256, 264]
        )
        entry = report["groups"][0]["entries"][0]
        self.assertEqual(entry["logical_capacity"], 32)
        self.assertEqual(entry["physical_slot_used"], 8)
        self.assertEqual(entry["physical_slot_free"], 24)
        first, view = entry["tensors"]
        self.assertEqual(first["backing_id"], view["backing_id"])
        self.assertEqual(view["byte_offset"], 16)
        self.assertEqual(view["byte_span"], 60)
        self.assertEqual(view["tensor_bytes"], 32)
        self.assertFalse(view["contiguous"])
        self.assertIsNone(entry["metadata_bytes"])
        self.assertEqual(entry["metadata_status"], "unknown")
        self.assertEqual(entry["allocator_ownership_status"], "unknown")
        self.assertIsNone(entry["allocator_class"])
        payload = json.dumps(report)
        self.assertNotIn(str(buffer.data_ptr()), payload)
        self.assertNotIn(hex(buffer.data_ptr()), payload)
        self.assertNotIn("HugeTLB", payload)

    def test_overlapping_distinct_storage_wrappers(self):
        backing = bytearray(128)
        first = torch.frombuffer(backing, dtype=torch.uint8, count=96)
        second = torch.frombuffer(backing, dtype=torch.uint8, count=96, offset=32)
        self.assertNotEqual(
            first.untyped_storage().data_ptr(), second.untyped_storage().data_ptr()
        )
        report = observer.host_pool_observability(
            scheduler(
                group(
                    c4=DeepSeekV4PagedHostPool(first),
                    c128=DeepSeekV4PagedHostPool(second),
                )
            ),
            role="prefill",
        )
        self.assertEqual(report["unique_backing_bytes"], 128)
        self.assertEqual(len(report["allocations"]), 1)
        rows = report["backings"]
        self.assertEqual(rows[0]["allocation_id"], rows[1]["allocation_id"])
        self.assertEqual(sorted(r["allocation_byte_offset"] for r in rows), [0, 32])

    def test_missing_data_and_capacity_are_unknown(self):
        pool = DeepSeekV4PagedHostPool(None)
        pool.logical_size = None
        report = observer.host_pool_observability(
            scheduler(group(state=pool)), role="decode"
        )
        entry = report["groups"][0]["entries"][0]
        self.assertEqual(report["status"], "incomplete")
        self.assertIsNone(report["unique_backing_bytes"])
        self.assertIsNone(entry["physical_slot_free"])
        self.assertIsNone(entry["unique_backing_bytes"])

    def test_logical_anchor_has_no_data_but_metadata_unknown(self):
        anchor = type("LogicalHostPool", (DeepSeekV4PagedHostPool,), {})(None)
        report = observer.host_pool_observability(
            scheduler(group(kv=anchor)), role="prefill"
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["unique_backing_bytes"], 0)
        self.assertIsNone(report["metadata_bytes"])

    def test_dcp_does_not_invent_physical_occupancy(self):
        pool = DeepSeekV4PagedHostPool(
            torch.zeros(16, dtype=torch.uint8), capacity=16, free=24
        )
        pool.logical_size = 32
        pool.dcp_size = 2
        report = observer.host_pool_observability(
            scheduler(group(kv=pool)), role="prefill"
        )
        entry = report["groups"][0]["entries"][0]
        self.assertEqual(entry["logical_used"], 8)
        self.assertIsNone(entry["physical_slot_used"])
        self.assertEqual(entry["capacity_status"], "unknown")

    def test_busy_lock_is_nonblocking_and_does_not_invent_zero(self):
        pool = DeepSeekV4PagedHostPool(torch.zeros(8))
        pool.lock = threading.Lock()
        with pool.lock:
            report = observer.host_pool_observability(
                scheduler(group(kv=pool)), role="decode"
            )
        entry = report["groups"][0]["entries"][0]
        self.assertIn("capacity_lock_unavailable", entry["issues"])
        self.assertIsNone(entry["physical_slot_used"])

    def test_bounded_entries_and_tensors_are_incomplete(self):
        buffer = torch.zeros(1)
        report = observer.host_pool_observability(
            scheduler(
                group(kv=DeepSeekV4PagedHostPool([buffer] * (observer.MAX_TENSORS + 1)))
            ),
            role="decode",
        )
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(
            len(report["groups"][0]["entries"][0]["tensors"]), observer.MAX_TENSORS
        )
        report = observer.host_pool_observability(
            scheduler(
                group(
                    **{
                        str(i): DeepSeekV4PagedHostPool(buffer)
                        for i in range(observer.MAX_ENTRIES + 1)
                    }
                )
            ),
            role="prefill",
        )
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["groups"][0]["status"], "incomplete")
        self.assertIsNone(report["groups"][0]["unique_backing_bytes"])

    def test_scheduler_readback_hook_and_fail_closed_error(self):
        # Execute the actual scheduler hook without importing GPU worker code.
        path = MODULE_PATH.parents[1] / "managers/scheduler.py"
        tree = ast.parse(path.read_text())
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "get_internal_state"
        )
        hook = next(
            node
            for node in method.body
            if isinstance(node, ast.Try)
            and "host_pool_observability" in ast.unparse(node)
        )
        code = compile(ast.Module(body=[hook], type_ignores=[]), str(path), "exec")
        state = {}
        namespace = dict(
            ret=state,
            self=scheduler(group(c4=DeepSeekV4PagedHostPool(torch.zeros(8)))),
            get_disagg=lambda: SimpleNamespace(disaggregation_mode="prefill"),
            host_pool_observability=observer.host_pool_observability,
        )
        exec(code, namespace)
        self.assertEqual(state["host_pool_observability"]["worker"]["role"], "prefill")
        self.assertEqual(state["host_pool_observability"]["unique_backing_bytes"], 32)

        def unavailable(*args, **kwargs):
            raise RuntimeError("private path and address")

        namespace["host_pool_observability"] = unavailable
        exec(code, namespace)
        self.assertEqual(state["host_pool_observability"]["status"], "incomplete")
        self.assertIsNone(state["host_pool_observability"]["unique_backing_bytes"])
        self.assertNotIn("private", json.dumps(state))

    def test_unreviewed_pool_kind_is_incomplete(self):
        pool = type("FutureHostPool", (DeepSeekV4PagedHostPool,), {})(torch.zeros(8))
        report = observer.host_pool_observability(
            scheduler(group(kv=pool)), role="prefill"
        )
        self.assertEqual(report["status"], "incomplete")
        self.assertIsNone(report["unique_backing_bytes"])
        self.assertIn(
            "unsupported_pool_kind", report["groups"][0]["entries"][0]["issues"]
        )

    def test_absent_and_existing_controller_path(self):
        self.assertEqual(
            observer.host_pool_observability(scheduler(), role="null")["status"],
            "not_present",
        )
        worker = scheduler()
        worker.tree_cache.cache_controller = SimpleNamespace(
            mem_pool_host=DeepSeekV4PagedHostPool(torch.zeros(2))
        )
        report = observer.host_pool_observability(worker, role="prefill")
        self.assertEqual(report["unique_backing_bytes"], 8)


if __name__ == "__main__":
    unittest.main()
