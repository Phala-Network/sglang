"""CPU-only deterministic contract tests for the CC fusion overlay.

Functions are loaded from the actual source/image files with GPU dependencies
stubbed. These tests prove dispatch and allocation ordering, not CUDA behavior.
Run the distributed GPU acceptance separately before serving traffic.
"""

import ast
import ctypes
import importlib.util
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SGLANG_ROOT = Path(os.environ.get("SGLANG_SOURCE_ROOT", "/sglang/python/sglang"))
FI_ROOT = Path(os.environ.get("FLASHINFER_SOURCE_ROOT", "/opt/flashinfer/flashinfer"))
FUSION = SGLANG_ROOT / "srt/layers/flashinfer_comm_fusion.py"
CC = SGLANG_ROOT / "srt/utils/confidential_compute.py"


def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    missing = set(names) - {node.name for node in nodes}
    if missing:
        raise AssertionError(f"Missing contract functions: {sorted(missing)}")
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class DispatchTests(unittest.TestCase):
    def resolve(self, backend="auto", cc=True, multi=False, sm100=True, sm90=False):
        ns = functions(FUSION, ["_resolve_backend", "_mnnvl_supported"], {
            "is_confidential_compute": lambda: cc,
            "get_platform": lambda: SimpleNamespace(is_sm100=sm100, is_sm90=sm90),
        })
        return ns["_resolve_backend"](backend, multi)

    def test_cc_auto_selects_multicast_free_backend(self):
        self.assertEqual(self.resolve(), "trtllm")

    def test_cc_explicit_trtllm(self):
        self.assertEqual(self.resolve("trtllm"), "trtllm")

    def test_cc_refuses_mnnvl(self):
        with self.assertRaisesRegex(ValueError, "multicast"):
            self.resolve("mnnvl")

    def test_cc_refuses_multi_node(self):
        for backend in ("auto", "trtllm", "mnnvl"):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                self.resolve(backend, multi=True)

    def test_unknown_backend_rejected(self):
        for cc in (False, True):
            with self.subTest(cc=cc), self.assertRaises(ValueError):
                self.resolve("typo", cc=cc)

    def test_non_cc_dispatch_unchanged(self):
        self.assertEqual(self.resolve(cc=False), "mnnvl")
        self.assertEqual(self.resolve(cc=False, multi=True), "mnnvl")
        self.assertEqual(self.resolve(cc=False, sm100=False, sm90=True), "trtllm")

    def test_unsupported_arch_stays_rejected(self):
        for cc in (False, True):
            with self.subTest(cc=cc), self.assertRaises(ValueError):
                self.resolve(cc=cc, sm100=False, sm90=False)


class DetectorTests(unittest.TestCase):
    def load_detector(self):
        spec = importlib.util.spec_from_file_location("cc_detector_under_test", CC)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_valid_software_overrides(self):
        for value, expected in (("0", False), ("1", True)):
            with patch.dict(os.environ, {"SGLANG_CONFIDENTIAL_COMPUTE": value}):
                self.assertEqual(self.load_detector().is_confidential_compute(), expected)

    def test_invalid_override_is_not_silently_false(self):
        for value in ("yes", "false", "2", ""):
            with patch.dict(os.environ, {"SGLANG_CONFIDENTIAL_COMPUTE": value}):
                with self.assertRaises(ValueError):
                    self.load_detector().is_confidential_compute()

    def test_nvml_modes_and_shutdown(self):
        for feature in (0, 1, 2):
            nvml = SimpleNamespace(nvmlInit=Mock(), nvmlShutdown=Mock(),
                NVMLError_NotSupported=OSError, NVMLError_FunctionNotFound=LookupError,
                nvmlSystemGetConfComputeState=Mock(return_value=SimpleNamespace(ccFeature=feature)))
            with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {
                "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)), "pynvml": nvml,
            }):
                detector = self.load_detector()
                self.assertEqual(detector.is_confidential_compute(), feature != 0)
                detector.is_confidential_compute()
                nvml.nvmlInit.assert_called_once()
                nvml.nvmlShutdown.assert_called_once()

    def test_query_failure_is_not_claimed_as_cc(self):
        nvml = SimpleNamespace(nvmlInit=Mock(), nvmlShutdown=Mock(),
            NVMLError_NotSupported=OSError, NVMLError_FunctionNotFound=LookupError,
            nvmlSystemGetConfComputeState=Mock(side_effect=RuntimeError("query failed")))
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)), "pynvml": nvml,
        }):
            self.assertFalse(self.load_detector().is_confidential_compute())
            nvml.nvmlShutdown.assert_called_once()

    def check_both_libraries(self, feature, multigpu, settings_error=None):
        class Settings(ctypes.Structure):
            _fields_ = [("ccFeature", ctypes.c_uint), ("multiGpuMode", ctypes.c_uint)]

        def settings(ptr):
            if settings_error:
                raise settings_error
            ptr._obj.ccFeature = feature
            ptr._obj.multiGpuMode = multigpu
            return 0

        nvml = SimpleNamespace(nvmlInit=Mock(), nvmlShutdown=Mock(),
            NVMLError_NotSupported=OSError, NVMLError_FunctionNotFound=LookupError,
            c_nvmlSystemConfComputeSettings_v1_t=Settings,
            nvmlSystemGetConfComputeSettings=Mock(side_effect=settings),
            NVML_CC_SYSTEM_MULTIGPU_PROTECTED_PCIE=1, _nvmlCheckReturn=Mock(),
            nvmlSystemGetConfComputeState=Mock(return_value=SimpleNamespace(ccFeature=feature)))
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        fi_ns = functions(FI_ROOT / "utils.py", ["is_confidential_compute"],
            {"torch": torch, "os": os, "logger": logging.getLogger("fi-cc-test")})
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {
            "torch": torch, "pynvml": nvml,
        }):
            values = (self.load_detector().is_confidential_compute(),
                      fi_ns["is_confidential_compute"]())
        self.assertEqual(nvml.nvmlShutdown.call_count, 2)
        return values, nvml

    def test_ppcie_only_is_cc_in_both_libraries(self):
        values, nvml = self.check_both_libraries(feature=0, multigpu=1)
        self.assertEqual(values, (True, True))
        nvml.nvmlSystemGetConfComputeState.assert_not_called()

    def test_cc_settings_disabled_in_both_libraries(self):
        values, _ = self.check_both_libraries(feature=0, multigpu=0)
        self.assertEqual(values, (False, False))

    def test_cc_feature_on_in_both_libraries(self):
        values, _ = self.check_both_libraries(feature=1, multigpu=0)
        self.assertEqual(values, (True, True))

    def test_unsupported_settings_falls_back_in_both_libraries(self):
        for error in (OSError("not supported"), LookupError("missing function")):
            with self.subTest(error=type(error).__name__):
                values, nvml = self.check_both_libraries(1, 0, error)
                self.assertEqual(values, (True, True))
                self.assertEqual(nvml.nvmlSystemGetConfComputeState.call_count, 2)

    def test_query_error_is_not_treated_as_supported_settings(self):
        values, nvml = self.check_both_libraries(1, 1, RuntimeError("query denied"))
        self.assertEqual(values, (False, False))
        nvml.nvmlSystemGetConfComputeState.assert_not_called()


class RankAgreementTests(unittest.TestCase):
    def check(self, states=None, sg=True, fi=True, guard=1, cpu_group="cpu"):
        from typing import Optional

        def gather(out, local, group):
            self.assertEqual(group, "cpu")
            out[:] = states if states is not None else [local] * 8

        mods = {"flashinfer.comm": SimpleNamespace(trtllm_ar=SimpleNamespace(CC_ALLOCATION_GUARD_VERSION=guard)),
                "flashinfer.utils": SimpleNamespace(is_confidential_compute=lambda: fi)}
        ns = functions(FUSION, ["_collective_cc_workspace_mode"], {
            "Optional": Optional, "is_confidential_compute": lambda: sg,
            "_TorchDistBackend": object(), "logger": logging.getLogger("cc-test"),
            "dist": SimpleNamespace(get_world_size=lambda group: 8, all_gather_object=gather),
        })
        with patch.dict(sys.modules, mods):
            return ns["_collective_cc_workspace_mode"](cpu_group, "trtllm")

    def test_eight_rank_cc_agreement(self):
        self.assertIs(self.check(), True)

    def test_eight_rank_non_cc_agreement(self):
        self.assertIs(self.check(sg=False, fi=False, guard=0), False)

    def test_library_mode_mismatch_rejects(self):
        self.assertIsNone(self.check(sg=True, fi=False))

    def test_rank_mode_mismatch_rejects(self):
        self.assertIsNone(self.check(states=[(True, True, "trtllm", True)] * 7 + [(False, False, "trtllm", True)]))

    def test_one_rank_detection_failure_rejects(self):
        self.assertIsNone(self.check(states=[(True, True, "trtllm", True)] * 7 + [None]))

    def test_missing_actual_allocation_guard_rejects(self):
        self.assertIsNone(self.check(guard=0))

    def test_no_cpu_group_rejects(self):
        self.assertIsNone(self.check(cpu_group=None))


class ActualAllocationTests(unittest.TestCase):
    def allocate(self, local_fail=False, peer_fail=False, backend=True):
        events = []
        tensor = object()

        def empty(*args, **kwargs):
            events.append("allocate")
            if local_fail:
                raise RuntimeError("injected rank-local OOM")
            return tensor

        def vote(error):
            events.append("cpu_vote")
            return [error] + (["peer OOM"] if peer_fail else [None]) + [None] * 6

        def rendezvous(held, group):
            self.assertIs(held, tensor)
            self.assertEqual(events, ["allocate", "cpu_vote"] if backend else ["allocate"])
            events.append("rendezvous")
            return SimpleNamespace(get_buffer=lambda *a, **k: SimpleNamespace(data_ptr=lambda: 123))

        torch = SimpleNamespace(dtype=object, device=object, Tensor=object,
            empty=lambda *a, **k: SimpleNamespace(element_size=lambda: 2))
        ns = functions(FI_ROOT / "comm/torch_symmetric_memory.py", ["_alloc_symm_buffer_bytes"], {
            "torch": torch, "Any": object,
            "_enable_symm_mem_for_group": lambda name: None,
            "symm_mem": SimpleNamespace(empty=empty, rendezvous=rendezvous),
        })
        self.events = events
        return ns["_alloc_symm_buffer_bytes"](128, 8, "bf16", "cuda", "tp-subgroup",
                comm_backend=SimpleNamespace(allgather=vote) if backend else None)

    def test_success_retains_allocation_through_vote(self):
        self.assertEqual(self.allocate()[0], [123] * 8)
        self.assertEqual(self.events, ["allocate", "cpu_vote", "rendezvous"])

    def test_local_oom_does_not_enter_rendezvous(self):
        with self.assertRaisesRegex(RuntimeError, "IPC allocation failed"):
            self.allocate(local_fail=True)
        self.assertEqual(self.events, ["allocate", "cpu_vote"])

    def test_peer_oom_does_not_enter_rendezvous(self):
        with self.assertRaisesRegex(RuntimeError, "peer OOM"):
            self.allocate(peer_fail=True)
        self.assertEqual(self.events, ["allocate", "cpu_vote"])

    def test_legacy_caller_without_guard_preserved(self):
        self.assertEqual(self.allocate(backend=False)[0], [123] * 8)

    def test_guard_is_wired_into_trtllm_allocation(self):
        tree = ast.parse((FI_ROOT / "comm/trtllm_ar.py").read_text())
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "trtllm_create_ipc_workspace_for_all_reduce_fusion")
        calls = [n for n in ast.walk(factory) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_alloc_symm_buffer_bytes"]
        self.assertTrue(calls)
        self.assertTrue(all(any(k.arg == "comm_backend" for k in call.keywords) for call in calls))


if __name__ == "__main__":
    unittest.main(verbosity=2)
