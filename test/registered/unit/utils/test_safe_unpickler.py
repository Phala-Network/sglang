"""Unit tests for SafeUnpickler exact-name allowlist hardening.

Covers CVE-2026-15969 (GHSA-h74r-pwx2-6qr2): the previous prefix-based
allowlist let an attacker chain builtins.getattr + builtins.__import__ (or
operator.attrgetter + pickletools.sys) into os.system() even though
("os", "system") was on the deny-list.
"""

import ast
import io
import operator
import pickle
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.utils.common import (
    SafeUnpickler,
    _looks_like_safetensors_payload,
    deserialize_tensor_payload,
    safe_pickle_loads,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Reduce:
    """Pickles as callable(*args) so we can emit a GLOBAL for any callable."""

    def __init__(self, func, args):
        self._func = func
        self._args = args

    def __reduce__(self):
        return (self._func, self._args)


def _pglob(module, name):
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n"


def _pstr(s):
    b = s.encode()
    return b"\x8c" + bytes([len(b)]) + b


_PROTO = b"\x80\x04"
_T1, _T2, _REDUCE, _STOP = b"\x85", b"\x86", b"R", b"."


def _build_reflection_chain(value):
    """Exercise the CVE reflection primitives without executing a command."""
    return (
        _PROTO
        + _pglob("builtins", "getattr")
        + _pglob("builtins", "__import__")
        + _pstr("builtins")
        + _T1
        + _REDUCE
        + _pstr("len")
        + _T2
        + _REDUCE
        + _pstr(value)
        + _T1
        + _REDUCE
        + _STOP
    )


def _build_operator_bypass_chain(value):
    """Exercise the operator/pickletools bypass without executing a command."""
    return (
        _PROTO
        + _pglob("operator", "attrgetter")
        + _pstr("len")
        + _T1
        + _REDUCE
        + _pglob("operator", "itemgetter")
        + _pstr("builtins")
        + _T1
        + _REDUCE
        + _pglob("operator", "attrgetter")
        + _pstr("modules")
        + _T1
        + _REDUCE
        + _pglob("pickletools", "sys")
        + _T1
        + _REDUCE
        + _T1
        + _REDUCE
        + _T1
        + _REDUCE
        + _pstr(value)
        + _T1
        + _REDUCE
        + _STOP
    )


def _build_io_struct_nested_pickle_chain(evil: bytes) -> bytes:
    """io_struct nested-pickle bypass:
    _maybe_unwrap_pickle(PickleWrapper(evil)) -> plain pickle.loads(evil)."""
    assert len(evil) < 256
    return (
        _PROTO
        + _pglob("sglang.srt.managers.io_struct", "_maybe_unwrap_pickle")
        + _pglob("sglang.srt.managers.io_struct", "PickleWrapper")
        + b"\x43"
        + bytes([len(evil)])
        + evil  # SHORT_BINBYTES
        + _T1
        + _REDUCE
        + _T1
        + _REDUCE
        + _STOP
    )


class TestSafeUnpickler(CustomTestCase):
    def _blocked(self, func, args):
        payload = pickle.dumps(_Reduce(func, args))
        with self.assertRaises(RuntimeError):
            safe_pickle_loads(payload)

    def test_import_blocked(self):
        # __import__("os") is the first half of the RCE chain
        self._blocked(__import__, ("os",))

    def test_getattr_blocked(self):
        # getattr(os, "system") is the second half of the RCE chain
        self._blocked(getattr, (object(), "__class__"))

    def test_operator_primitives_blocked(self):
        # operator.attrgetter / itemgetter / getitem bypass primitives
        self._blocked(operator.attrgetter, ("system",))
        self._blocked(operator.itemgetter, ("os",))
        self._blocked(operator.getitem, (dict(), "k"))

    def test_pickletools_blocked(self):
        with self.assertRaises(RuntimeError):
            safe_pickle_loads(_pglob("pickletools", "sys") + _STOP)

    def test_dangerous_builtins_blocked(self):
        for func, args in [
            (setattr, (object(), "x", 1)),
            (eval, ("1",)),
            (exec, ("pass",)),
            (open, ("/etc/passwd",)),
        ]:
            self._blocked(func, args)

    def test_full_rce_chain_blocked(self):
        with self.assertRaises(RuntimeError):
            SafeUnpickler(io.BytesIO(_build_reflection_chain("not evaluated"))).load()

    def test_full_operator_bypass_chain_blocked(self):
        with self.assertRaises(RuntimeError):
            SafeUnpickler(
                io.BytesIO(_build_operator_bypass_chain("not evaluated"))
            ).load()

    def test_dynamic_import_blocked(self):
        # sglang.srt.utils prefix is not trusted anymore: GLOBAL to the
        # reflective dynamic_import helper must be rejected (would otherwise
        # return os.system for dynamic_import("os.system") -> RCE).
        with self.assertRaises(RuntimeError):
            SafeUnpickler(
                io.BytesIO(
                    _PROTO
                    + _pglob("sglang.srt.utils.common", "dynamic_import")
                    + _pstr("os.system")
                    + _T1
                    + _REDUCE
                    + _STOP
                )
            ).load()

    def test_dynamic_import_blocked_via_weight_updater(self):
        # Same function re-exported into another previously-prefix-trusted module.
        with self.assertRaises(RuntimeError):
            SafeUnpickler(
                io.BytesIO(
                    _PROTO
                    + _pglob(
                        "sglang.srt.model_executor.model_runner_components.weight_updater",
                        "dynamic_import",
                    )
                    + _pstr("os.system")
                    + _T1
                    + _REDUCE
                    + _STOP
                )
            ).load()

    def test_io_struct_nested_pickle_blocked(self):
        # Nested unrestricted pickle.loads bypass: outer GLOBALs to
        # io_struct._maybe_unwrap_pickle / PickleWrapper must be rejected now
        # that sglang.srt.managers. is not prefix-trusted.
        evil = _build_reflection_chain("not evaluated")
        with self.assertRaises(RuntimeError):
            SafeUnpickler(io.BytesIO(_build_io_struct_nested_pickle_chain(evil))).load()

    def test_torch_load_entry_points_blocked(self):
        for module, name in [
            ("torch", "load"),
            ("torch.hub", "load"),
            ("torch.utils.cpp_extension", "load"),
            ("torch.utils.cpp_extension", "load_inline"),
            ("torch.jit", "load"),
        ]:
            with self.assertRaises(RuntimeError):
                SafeUnpickler(io.BytesIO(_PROTO + _pglob(module, name) + _STOP)).load()

    def test_sglang_internal_class_not_allowlisted_blocked(self):
        # A random sglang.srt class (e.g. under managers) must not be loadable.
        with self.assertRaises(RuntimeError):
            SafeUnpickler(
                io.BytesIO(
                    _PROTO
                    + _pglob("sglang.srt.managers.io_struct", "PickleWrapper")
                    + _STOP
                )
            ).load()

    def test_benign_payloads_still_load(self):
        for obj in [
            {"a": 1, "b": [1, 2, 3]},
            [1, 2, 3],
            (1, "x"),
            b"bytes",
            {1, 2, 3},
            frozenset({1, 2}),
            "hello",
            42,
            3.14,
            True,
        ]:
            self.assertEqual(safe_pickle_loads(pickle.dumps(obj)), obj)

    def test_tensor_roundtrip(self):
        import torch

        t = torch.zeros(2, 3)
        restored = safe_pickle_loads(pickle.dumps(t))
        torch.testing.assert_close(restored, t)

    def test_flattened_tensor_bucket_roundtrip(self):
        import torch

        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        ftb = FlattenedTensorBucket(flattened_tensor=torch.randn(10), metadata={"a": 1})
        restored = safe_pickle_loads(pickle.dumps(ftb))
        torch.testing.assert_close(restored.flattened_tensor, ftb.flattened_tensor)
        self.assertEqual(restored.metadata, ftb.metadata)

    def test_local_serialized_tensor_roundtrip(self):
        import torch

        from sglang.srt.model_executor.model_runner_components.weight_updater import (
            LocalSerializedTensor,
        )

        lst = LocalSerializedTensor(
            values=[pickle.dumps(torch.randn(2)) for _ in range(2)]
        )
        restored = safe_pickle_loads(pickle.dumps(lst))
        torch.testing.assert_close(restored.get(0), lst.get(0))

    def test_safetensors_wire_format_roundtrip(self):
        import safetensors.torch
        import torch

        payload = safetensors.torch.save({"a": torch.randn(2, 3), "b": torch.ones(4)})
        self.assertTrue(_looks_like_safetensors_payload(payload))
        self.assertFalse(_looks_like_safetensors_payload(pickle.dumps({"x": 1})))

        restored = deserialize_tensor_payload(payload)
        self.assertEqual(set(restored), {"a", "b"})
        torch.testing.assert_close(restored["a"], safetensors.torch.load(payload)["a"])

    def test_legacy_pickle_still_accepted_via_wire_helper(self):
        # Backward-compat: old pickle clients still work through the hardened
        # SafeUnpickler path (no reflection primitives reachable anymore).
        import torch

        t = torch.zeros(2, 3)
        blob = pickle.dumps({"x": t})
        restored = deserialize_tensor_payload(blob)
        torch.testing.assert_close(restored["x"], t)

    def test_serialized_management_routes_require_admin_key(self):
        """The two attacker-controlled deserialization routes must fail closed."""
        source = (
            Path(__file__).resolve().parents[4]
            / "python"
            / "sglang"
            / "srt"
            / "entrypoints"
            / "http_server.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in (
            "update_weights_from_tensor",
            "load_lora_adapter_from_tensors",
        ):
            decorators = ast.dump(functions[name], include_attributes=False)
            self.assertIn("ADMIN_FORCE", decorators, name)

    def test_malformed_weight_payload_is_request_level_failure(self):
        """A worker decode/type failure must not escape into the scheduler loop."""
        import torch

        from sglang.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager,
        )

        class RejectingWorker:
            def update_weights_from_tensor(self, _request):
                raise TypeError("serialized tensor payload is not an iterable")

        manager = SchedulerWeightUpdaterManager(
            tp_worker=RejectingWorker(),
            draft_worker=None,
            tp_cpu_group=object(),
            memory_saver_adapter=None,
            flush_cache=lambda **_kwargs: True,
            is_fully_idle=lambda: True,
        )
        request = SimpleNamespace(disable_draft_model=False)
        with patch.object(torch.distributed, "barrier") as barrier:
            result = manager.update_weights_from_tensor(request)

        self.assertFalse(result.success)
        self.assertIn("not an iterable", result.message)
        barrier.assert_called_once_with(group=manager.tp_cpu_group)


if __name__ == "__main__":
    unittest.main()
