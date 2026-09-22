"""CPU regressions for the real FlashInfer TRTLLM workspace destroy method."""

import ast
import gc
import os
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

source = Path(
    os.environ.get(
        "FLASHINFER_ALLREDUCE_SOURCE",
        "/opt/sglang/lib/python3.12/site-packages/flashinfer/comm/allreduce.py",
    )
)
tree = ast.parse(source.read_text())
workspace_class = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "TRTLLMAllReduceFusionWorkspace"
)
destroy_node = next(
    node
    for node in workspace_class.body
    if isinstance(node, ast.FunctionDef) and node.name == "destroy"
)


class Allocation:
    pass


class WorkspaceCleanupContract(unittest.TestCase):
    def setUp(self):
        self.registry = {}
        self.release = Mock(
            side_effect=lambda handles: self.registry.pop(id(handles), None)
        )
        scope = {"trtllm_destroy_ipc_workspace_for_all_reduce_fusion": self.release}
        exec(
            compile(
                ast.Module(body=[destroy_node], type_ignores=[]), str(source), "exec"
            ),
            scope,
        )
        self.destroy = scope["destroy"]

    def workspace(self, cc):
        handles, allocation, tensor = [[1, 2]], Allocation(), Allocation()
        mem_handles = [] if cc else [allocation]
        if cc:
            self.registry[id(handles)] = [allocation]
        workspace = SimpleNamespace(
            ipc_handles=handles,
            workspace_tensor=tensor,
            mem_handles=mem_handles,
            metadata={},
            _internal_workspace=(handles, tensor, mem_handles, {}),
        )
        return workspace, weakref.ref(allocation), weakref.ref(tensor), id(handles)

    def test_cc_allocator_registry_released(self):
        workspace, allocation, _, key = self.workspace(True)
        self.destroy(workspace)
        gc.collect()
        self.assertNotIn(key, self.registry)
        self.assertIsNone(allocation())

    def test_internal_tuple_does_not_retain_non_cc_memory(self):
        workspace, allocation, tensor, _ = self.workspace(False)
        self.destroy(workspace)
        gc.collect()
        self.assertIsNone(allocation())
        self.assertIsNone(tensor())
        self.assertIsNone(workspace._internal_workspace)

    def test_double_destroy_is_idempotent(self):
        workspace, _, _, _ = self.workspace(True)
        self.destroy(workspace)
        self.destroy(workspace)
        self.release.assert_called_once()
        self.assertTrue(workspace._destroyed)

    def test_release_failure_preserves_handles_for_retry(self):
        workspace, _, _, key = self.workspace(True)
        self.release.side_effect = RuntimeError("injected release failure")
        with self.assertRaisesRegex(RuntimeError, "injected release failure"):
            self.destroy(workspace)
        self.assertEqual(id(workspace.ipc_handles), key)
        self.assertFalse(getattr(workspace, "_destroyed", False))
        self.release.side_effect = lambda handles: self.registry.pop(id(handles), None)
        self.destroy(workspace)
        self.assertEqual(self.registry, {})

    def test_recreate_does_not_accumulate_registry_entries(self):
        for _ in range(3):
            workspace, allocation, _, _ = self.workspace(True)
            self.assertEqual(len(self.registry), 1)
            self.destroy(workspace)
            gc.collect()
            self.assertEqual(self.registry, {})
            self.assertIsNone(allocation())


if __name__ == "__main__":
    unittest.main(verbosity=2)
