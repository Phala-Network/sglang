"""Dependency-free boundary regression for the real GPU preflight matrix."""

import ast
import ctypes
from pathlib import Path
import unittest


source = Path(__file__).with_name("gpu_preflight.py")
tree = ast.parse(source.read_text())
definition = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "preflight_oneshot_modes")
namespace = {}
exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
modes = namespace["preflight_oneshot_modes"]
host_definition = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "valid_host_allocation")
exec(compile(ast.Module(body=[host_definition], type_ignores=[]), str(source), "exec"), namespace)
valid_host = namespace["valid_host_allocation"]


class PreflightModeContract(unittest.TestCase):
    def test_tp8_small_decode_shapes_keep_oneshot(self):
        for tokens in (1, 8):
            with self.subTest(tokens=tokens):
                self.assertEqual(modes(tokens, 8), (True,))

    def test_tp8_supported_shapes_keep_both_modes(self):
        for tokens in (9, 32, 48, 288, 384):
            with self.subTest(tokens=tokens):
                self.assertEqual(modes(tokens, 8), (True, False))

    def test_boundary_uses_actual_world_size(self):
        for world_size in (2, 4, 8, 16):
            with self.subTest(world_size=world_size):
                self.assertEqual(modes(world_size, world_size), (True,))
                self.assertEqual(modes(world_size + 1, world_size), (True, False))

    def test_existing_shape_matrix_is_preserved(self):
        expected = (1, 8, 32, 48, 288, 384)
        tuples = [tuple(x.value for x in node.elts)
                  for node in ast.walk(tree) if isinstance(node, ast.Tuple)
                  and all(isinstance(x, ast.Constant) for x in node.elts)]
        self.assertIn(expected, tuples)
        self.assertEqual(sum(len(modes(tokens, 8)) for tokens in expected), 10)

    def test_reference_accumulates_in_fp32_before_collective(self):
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "reduced"
                               for target in node.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.unparse(assignments[0].value), "src.float()")
        calls = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertIn("dist.all_reduce(reduced)", calls)
        self.assertIn("dist.all_reduce(reduced_bf16)", calls)

    def test_numeric_tolerances_not_relaxed(self):
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == "torch.testing.assert_close"
                 and ast.unparse(node.args[1]) in ("reduced", "residual_ref")]
        self.assertEqual(len(calls), 4)
        for call in calls:
            self.assertEqual({kw.arg: ast.literal_eval(kw.value) for kw in call.keywords},
                             {"rtol": 0.03, "atol": 0.01})

    def test_cc_native_host_allocation_accepts_managed_reporting(self):
        self.assertTrue(valid_host(True, False, True, 0, 3))
        self.assertTrue(valid_host(True, False, True, 0, 1))

    def test_native_pinned_tensor_is_preserved(self):
        self.assertTrue(valid_host(True, True, False, 0, 1))

    def test_unregistered_device_or_unproven_managed_memory_is_rejected(self):
        for args in [(True, False, True, 1, 3), (True, False, False, 0, 3),
                     (True, False, True, 0, 0), (True, False, True, 0, 2),
                     (False, True, True, 0, 3)]:
            with self.subTest(args=args):
                self.assertFalse(valid_host(*args))

    def test_cuda13_pointer_attribute_reserved_tail_is_present(self):
        definition = next(node for node in ast.walk(tree)
                          if isinstance(node, ast.ClassDef) and node.name == "Attributes")
        scope = {"ctypes": ctypes}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), scope)
        attributes = scope["Attributes"]
        self.assertEqual(attributes._fields_[-1], ("reserved", ctypes.c_long * 8))
        self.assertEqual(attributes.reserved.offset, 8 + 2 * ctypes.sizeof(ctypes.c_void_p))
        self.assertEqual(ctypes.sizeof(attributes),
                         attributes.reserved.offset + 8 * ctypes.sizeof(ctypes.c_long))


if __name__ == "__main__":
    unittest.main(verbosity=2)
