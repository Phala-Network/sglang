"""CPU source tests; native master/SSD behavior requires its own C++ tests."""

import ast
import importlib.util
import json
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "python/sglang/srt/managers/shared_cache_control.py"
spec = importlib.util.spec_from_file_location("control", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def scheduler(completed=None):
    calls = []

    def clear(keys):
        calls.append(list(keys))
        return list(keys) if completed is None else completed

    backend = SimpleNamespace(
        store=SimpleNamespace(batch_memory_replica_clear=clear),
        config=SimpleNamespace(tenant_id="tenant-current"),
    )
    controller = SimpleNamespace(
        storage_backend=backend,
        ack_backup_queue=Queue(),
        backup_queue=Queue(),
        ack_prefetch_queue=Queue(),
    )
    obj = SimpleNamespace(
        decode_offload_manager=SimpleNamespace(cache_controller=controller),
        is_fully_idle=lambda: True,
        disaggregation_mode="decode",
        server_args=SimpleNamespace(
            tp_size=1, dp_size=1, pp_size=1, dcp_size=1, attn_cp_size=1, nnodes=1
        ),
    )
    return obj, calls


class Tests(unittest.TestCase):
    def invoke(self, obj, keys):
        # Actual drain predicate, zero sleep; deterministic CPU fixture only.
        with patch.object(m.time, "monotonic", side_effect=[0, 1]):
            return m.clear_current_memory(obj, keys)

    def test_repeat_original_store_no_artifacts(self):
        obj, calls = scheduler()
        with patch.dict(m.os.environ, {}, clear=True):
            for _ in range(2):
                result = self.invoke(obj, ["k1", "k2"])
                self.assertTrue(result["success"])
                self.assertEqual(result["tenant_id"], "tenant-current")
        self.assertEqual(calls, [["k1", "k2"], ["k1", "k2"]])

    def test_partial_is_explicit(self):
        obj, _ = scheduler(["k2"])
        result = self.invoke(obj, ["k1", "k2"])
        self.assertFalse(result["success"])
        self.assertEqual(result["completed_keys"], ["k2"])
        self.assertEqual(result["remaining_keys"], ["k1"])

    def test_invalid_keys_refuse_before_native(self):
        obj, calls = scheduler()
        for keys in ([], ["a", "a"], ["*"], [""], [True], "a", ["a"] * 257):
            with self.subTest(keys=keys), self.assertRaises(m.SharedCacheControlError):
                m.clear_current_memory(obj, keys)
        self.assertEqual(calls, [])

    def test_busy_and_wrong_role_refuse(self):
        obj, calls = scheduler()
        obj.is_fully_idle = lambda: False
        with self.assertRaises(m.SharedCacheControlError):
            self.invoke(obj, ["k"])
        obj.is_fully_idle = lambda: True
        obj.disaggregation_mode = "prefill"
        with self.assertRaises(m.SharedCacheControlError):
            self.invoke(obj, ["k"])
        self.assertEqual(calls, [])

    def test_native_capability_and_tenant_required(self):
        obj, calls = scheduler()
        obj.decode_offload_manager.cache_controller.storage_backend.config.tenant_id = (
            None
        )
        with self.assertRaises(m.SharedCacheControlError):
            self.invoke(obj, ["k"])
        self.assertEqual(calls, [])
        obj, _ = scheduler()
        obj.decode_offload_manager.cache_controller.storage_backend.store = object()
        with self.assertRaises(m.SharedCacheControlError):
            self.invoke(obj, ["k"])

    def test_bad_native_result_is_unknown(self):
        for result in (["other"], ["k", "k"], "k", [1]):
            obj, _ = scheduler(result)
            with (
                self.subTest(result=result),
                self.assertRaises(m.SharedCacheControlError) as exc,
            ):
                self.invoke(obj, ["k"])
            self.assertTrue(exc.exception.unknown)

    def test_body_legacy_preserved_and_new_strict(self):
        legacy = {
            "manifest_id": "old",
            "manifest_sha256": "a" * 64,
            "request_id": "old",
        }
        self.assertEqual(m.parse_memory_clear_body(json.dumps(legacy).encode()), legacy)
        self.assertEqual(m.parse_memory_clear_body(b'{"keys":["k"]}'), {"keys": ["k"]})
        for body in (
            b'{"keys":["k"],"tenant_id":"other"}',
            b'{"keys":["k"],"segment":""}',
            b'{"keys":["k"],"manifest_id":"x"}',
            b"x" * 65537,
        ):
            with self.assertRaises(m.SharedCacheControlError):
                m.parse_memory_clear_body(body)

    def test_actual_scheduler_handler_partial_and_unknown(self):
        tree = ast.parse(
            (ROOT / "python/sglang/srt/managers/scheduler.py").read_text(
                encoding="utf-8"
            )
        )
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "clear_shared_cache_memory"
        )
        method.decorator_list = []
        scope = {
            "SharedCacheClearMemoryReqInput": SimpleNamespace,
            "SharedCacheClearMemoryReqOutput": SimpleNamespace,
            "SharedCacheControlError": m.SharedCacheControlError,
            "clear_current_memory": lambda *args: {
                "success": False,
                "completed_keys": ["k1"],
                "remaining_keys": ["k2"],
            },
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), "handler", "exec"),
            scope,
        )
        obj = SimpleNamespace()
        req = SimpleNamespace(
            keys=["k1", "k2"], manifest_id="", manifest_sha256="", request_id=""
        )
        result = scope[method.name](obj, req)
        self.assertFalse(result.success)
        self.assertEqual(result.receipt["completed_keys"], ["k1"])

        def unknown(*args):
            raise m.SharedCacheControlError("unknown", unknown=True)

        scope["clear_current_memory"] = unknown
        result = scope[method.name](obj, req)
        self.assertTrue(result.unknown)
        self.assertTrue(obj._shared_cache_clear_unknown)
        result = scope[method.name](obj, req)
        self.assertEqual(result.reason, "prior_clear_result_unknown")


if __name__ == "__main__":
    unittest.main()
