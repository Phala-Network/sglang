"""Source contracts for device-scoped KV dispatch; no GPU emulation claim."""

import ast
import importlib.util
import inspect
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import List
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
EXTENSION = ROOT / "python/sglang/kernels/kvcache_incremental"
WRAPPER = ROOT / "python/sglang/kernels/aot/python/sgl_kernel/kvcacheio.py"
HOST = ROOT / "python/sglang/srt/mem_cache/pool_host/common.py"


class Tensor:
    def __init__(self, device="cuda", index=1):
        self.device = SimpleNamespace(type=device, index=index)

    def data_ptr(self):
        return 11


class Operator:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name
        self.default = self

    def __call__(self, *args):
        self.owner.calls.append((self.name, args))
        return self.owner.pointer


class Namespace:
    def __init__(self, pointer):
        self.pointer, self.calls = pointer, []

    def __getattr__(self, name):
        return Operator(self, name)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KvBackendTests(unittest.TestCase):
    def setUp(self):
        self.public, self.private = Namespace(17), Namespace(29)
        self.caps = {0: (8, 9), 1: (9, 0), 2: (10, 0), 3: (10, 3)}
        self.probed, self.loaded, self.libraries = [], [], []
        self.torch = ModuleType("torch")
        self.torch.Tensor = Tensor
        self.torch.uint64 = "uint64"
        self.torch.version = SimpleNamespace(hip=None, cuda="13.0")
        self.torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=self.capability,
            current_device=lambda: 0,
        )
        self.torch.ops = SimpleNamespace(
            sgl_kernel=self.public,
            phala_kvcache_incremental=self.private,
            load_library=self.libraries.append,
        )
        self.provider = ModuleType("phala_kvcache_incremental")
        self.provider.op_namespace = self.namespace
        context = patch.dict(
            sys.modules,
            {"torch": self.torch, "phala_kvcache_incremental": self.provider},
        )
        context.start()
        self.addCleanup(context.stop)
        self.wrapper = load(WRAPPER, "tested_kv_wrapper")

    def capability(self, index):
        self.probed.append(index)
        return self.caps[index]

    def namespace(self):
        self.loaded.append(True)
        return self.private

    def test_import_does_not_probe_or_load_cuda(self):
        self.assertEqual(self.probed, [])
        self.assertEqual(self.loaded, [])

    def test_each_supported_device_is_selected_and_cached_independently(self):
        for index in (1, 2, 3):
            for _ in range(2):
                self.assertIs(
                    self.wrapper._transfer_ops(Tensor(index=index)), self.private
                )
        self.assertEqual(self.probed, [1, 2, 3])
        self.assertEqual(len(self.loaded), 3)
        self.assertIs(self.wrapper._transfer_ops(Tensor(index=0)), self.public)

    def test_cpu_hip_and_unavailable_cuda_keep_public_ops(self):
        self.assertIs(self.wrapper._transfer_ops(Tensor("cpu", None)), self.public)
        self.wrapper._is_hip = True
        self.assertIs(self.wrapper._transfer_ops(Tensor()), self.public)
        self.wrapper._is_hip = False
        self.torch.cuda.is_available = lambda: False
        self.assertIs(self.wrapper._transfer_ops(Tensor()), self.public)
        self.assertEqual(self.probed, [])
        self.assertEqual(self.loaded, [])

    def test_all_transfer_wrappers_preserve_arguments_and_actual_device(self):
        functions = {
            name: function
            for name, function in vars(self.wrapper).items()
            if name.startswith("transfer_") and callable(function)
        }
        self.assertEqual(len(functions), 14)
        for name, function in functions.items():
            args = []
            for param in inspect.signature(function).parameters.values():
                if param.annotation is Tensor:
                    args.append(Tensor())
                elif param.annotation == List[Tensor]:
                    args.append([Tensor()])
                elif param.annotation == List[int]:
                    args.append([8])
                else:
                    args.append(8)
            with self.subTest(operation=name):
                function(*args)
                self.assertEqual(self.private.calls[-1], (name, tuple(args)))
        self.assertEqual(self.public.calls, [])
        self.assertEqual(self.probed, [1])

    def test_direct_transfers_find_cuda_in_either_tensor_group(self):
        for src, dst in (
            (Tensor("cpu", None), Tensor()),
            (Tensor(), Tensor("cpu", None)),
        ):
            self.wrapper.transfer_kv_direct(
                [src], [dst], Tensor("cpu"), Tensor("cpu"), 1
            )
            self.assertEqual(self.private.calls[-1][0], "transfer_kv_direct")
            self.wrapper.transfer_embedding_ranges_direct(src, dst, [0], [0], [1])
            self.assertEqual(
                self.private.calls[-1][0], "transfer_embedding_ranges_direct"
            )
        self.assertEqual(self.probed, [1])

    def test_cpu_cache_copy_is_not_replaced(self):
        values = [Tensor("cpu", None) for _ in range(4)]
        self.wrapper.copy_all_layer_kv_cache_cpu(*values)
        self.assertEqual(
            self.public.calls, [("copy_all_layer_kv_cache_cpu", tuple(values))]
        )
        self.assertEqual(self.loaded, [])

    def test_pointer_resolution_uses_explicit_target_and_rejects_negative(self):
        tensor = Tensor("cpu", None)
        self.assertEqual(self.wrapper.get_device_accessible_ptr(tensor, 2), 29)
        self.assertEqual(
            self.private.calls[-1], ("get_device_accessible_ptr", (tensor, 2))
        )
        with self.assertRaisesRegex(RuntimeError, "non-negative"):
            self.wrapper.get_device_accessible_ptr(tensor, -1)
        self.assertEqual(self.probed, [2])

    def test_missing_selected_backend_does_not_fall_back(self):
        with patch.object(
            self.provider, "op_namespace", side_effect=ImportError("missing native")
        ):
            with self.assertRaisesRegex(ImportError, "missing native"):
                self.wrapper.get_device_accessible_ptr(Tensor("cpu"), 1)
        self.assertEqual(self.public.calls, [])

    def test_missing_pointer_operator_cannot_return_raw_address(self):
        self.torch.ops.sgl_kernel = SimpleNamespace()
        with self.assertRaisesRegex(ImportError, "pointer resolution is required"):
            self.wrapper.get_device_accessible_ptr(Tensor("cpu"), 0)

    def host_functions(self):
        names = {"_resolve_device_accessible_ptr_fn", "make_kernel_ptr_table"}
        tree = ast.parse(HOST.read_text())
        body = [
            node
            for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in names)
            or (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        ]
        self.assertEqual(sum(isinstance(node, ast.FunctionDef) for node in body), 2)
        namespace = {"torch": self.torch, "lru_cache": lru_cache}
        self.torch.device = lambda value: SimpleNamespace(type="cuda", index=1)
        self.torch.tensor = lambda values, **kwargs: values
        exec(
            compile(ast.Module(body=body, type_ignores=[]), str(HOST), "exec"),
            namespace,
        )
        return namespace

    def test_host_pointer_table_requires_alias_for_registered_cuda(self):
        functions = self.host_functions()
        package = ModuleType("sgl_kernel")
        package.__path__ = []
        with patch.dict(
            sys.modules,
            {"sgl_kernel": package, "sgl_kernel.kvcacheio": self.wrapper},
        ):
            table = functions["make_kernel_ptr_table"](
                [Tensor("cpu")], "cuda:1", host_memory_registered=True
            )
            self.assertEqual(table, [29])
        functions["_resolve_device_accessible_ptr_fn"] = lambda: None
        with self.assertRaisesRegex(ImportError, "pointer resolution is required"):
            functions["make_kernel_ptr_table"](
                [Tensor("cpu")], "cuda:1", host_memory_registered=True
            )
        self.assertEqual(
            functions["make_kernel_ptr_table"](
                [Tensor("cpu")], "cuda:1", host_memory_registered=False
            ),
            [11],
        )

    def test_missing_wrapper_pointer_function_fails_closed(self):
        functions = self.host_functions()
        with patch.dict(sys.modules, {"sgl_kernel.kvcacheio": SimpleNamespace()}):
            with self.assertRaisesRegex(ImportError, "fallback is disabled"):
                functions["_resolve_device_accessible_ptr_fn"]()

    def test_native_loader_checks_library_count_and_all_required_ops(self):
        loader = load(
            EXTENSION / "python/phala_kvcache_incremental/__init__.py",
            "tested_private_loader",
        )
        self.assertEqual(len(loader._REQUIRED_OPS), 15)
        for files in ([], [Path("one.so"), Path("two.so")]):
            with patch.object(Path, "glob", return_value=files):
                with self.assertRaisesRegex(ImportError, "exactly one"):
                    loader.ensure_loaded()
        self.torch.ops.phala_kvcache_incremental = SimpleNamespace()
        with patch.object(Path, "glob", return_value=[Path("one.so")]):
            with self.assertRaisesRegex(ImportError, "did not register"):
                loader.ensure_loaded()
        self.assertFalse(loader._LOADED)

    def test_native_loader_runs_once_under_concurrency(self):
        loader = load(
            EXTENSION / "python/phala_kvcache_incremental/__init__.py",
            "tested_concurrent_loader",
        )
        with patch.object(Path, "glob", return_value=[Path("one.so")]):
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(lambda _: loader.ensure_loaded(), range(20)))
        self.assertTrue(loader._LOADED)
        self.assertEqual(self.libraries, ["one.so"])

    def test_binding_generator_requires_exact_transfer_source(self):
        generator = load(
            EXTENSION / "scripts/generate_binding.py", "tested_binding_generator"
        )
        source = ROOT / "python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu"
        lock = EXTENSION / "source-lock.json"
        output = generator.generate(source, lock)
        self.assertIn(b"namespace phala_kvcache_incremental_impl", output)
        self.assertEqual(output.count(b'm.def("'), 15)
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "transfer.cu"
            altered.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(RuntimeError, "hash does not match"):
                generator.generate(altered, lock)

    def test_zero_work_and_invalid_quota_are_guarded_before_launch_arithmetic(self):
        source = (
            ROOT / "python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu"
        ).read_text(encoding="utf-8")
        body = source.split("void transfer_kv_launcher(", 1)[1].split(
            "\nvoid transfer_kv_per_layer(", 1
        )[0]
        empty = body.index("if (num_items == 0)")
        self.assertLess(empty, body.index("OptionalCUDAGuard"))
        self.assertLess(empty, body.index("resolve_device_accessible_ptr(src_k)"))
        self.assertLess(body.index("Block quota must be positive"), empty)
        self.assertLess(body.index("Warp count must be positive"), empty)
        self.assertLess(body.index("Launch quota product exceeds int64 range"), empty)
        self.assertIn("return x / y + (x % y != 0);", body)


if __name__ == "__main__":
    unittest.main()
