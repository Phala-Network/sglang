"""Read-only seed selector discovery using the actual component formatter."""

import ast
import copy
import hashlib
import importlib.util
import json
import sys
import types
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
if importlib.util.find_spec("sglang") is None:
    for package in ("sglang", "sglang.test", "sglang.test.ci"):
        module = types.ModuleType(package)
        module.__path__ = [str(ROOT / "python" / package.replace(".", "/"))]
        module.__spec__ = importlib.util.spec_from_loader(
            package, loader=None, is_package=True
        )
        sys.modules[package] = module

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

spec = importlib.util.spec_from_file_location(
    "host_pool", ROOT / "python/sglang/srt/observability/host_pool.py"
)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def fixture():
    # Execute the exact existing enum and read-only formatter, avoiding GPU imports.
    base = ROOT / "python/sglang/srt/mem_cache"
    tree = ast.parse((base / "hicache_storage.py").read_text())
    enum = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PoolName"
    )
    ns = {"Enum": Enum}
    exec(compile(ast.Module(body=[enum], type_ignores=[]), "PoolName", "exec"), ns)
    tree = ast.parse((base / "storage/mooncake_store/mooncake_store.py").read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_get_hybrid_page_component_keys"
    )
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), "component_formatter", "exec"
        ),
        ns,
    )
    names = [ns["PoolName"].DEEPSEEK_V4_C4, ns["PoolName"].DEEPSEEK_V4_C128]
    store = SimpleNamespace(
        registered_pools={name: object() for name in names}, mla_suffix="fixture"
    )
    store._get_hybrid_page_component_keys = types.MethodType(
        ns["_get_hybrid_page_component_keys"], store
    )
    schema = {
        "version": 1,
        "revision": None,
        "layout": "KVLayout.V4",
        "unified": True,
        "uniform_fp8": False,
        "layers": [[4, 0], [128, 0]],
        "layer_range": [0, 2],
        "topology": [1, 1, 1],
        "cp_rank": 0,
        "pools": [
            [str(name), 128, "layer_first", 1, "torch.uint8", 512, None]
            for name in names
        ],
    }
    cfg = SimpleNamespace(
        tp_size=1,
        pp_size=1,
        tp_rank=0,
        extra_config={"extra_backend_tag": "dsv4-v1-" + "a" * 64},
    )
    cc = SimpleNamespace(
        storage_config=cfg, storage_backend=store, should_backup=lambda transfer: True
    )
    manager = SimpleNamespace(
        is_dsv4=True,
        cache_controller=cc,
        decode_host_mem_pool=SimpleNamespace(storage_schema=schema),
    )
    return SimpleNamespace(decode_offload_manager=manager), schema, store


class TestSeedSchemaObservability(unittest.TestCase):
    def test_exact_hash_components_null_revision_and_no_mutation(self):
        worker, schema, store = fixture()
        before = copy.deepcopy(schema)
        registered = dict(store.registered_pools)
        result = observer._seed_schema(worker)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["storage_schema"], before)
        self.assertEqual(
            result["kv_schema"],
            hashlib.sha256(
                json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )
        self.assertIsNone(result["model_revision"])
        self.assertEqual(
            result["required_components"], ["deepseek_v4_c128:0", "deepseek_v4_c4:0"]
        )
        self.assertEqual(schema, before)
        self.assertEqual(store.registered_pools, registered)
        self.assertIsNone(result["page_range"])
        self.assertEqual(result["page_range_status"], "operation_dependent")
        # Returned data is detached and cannot mutate the live schema.
        result["storage_schema"]["layers"].clear()
        self.assertEqual(schema, before)
        report = observer.host_pool_observability(
            SimpleNamespace(decode_offload_manager=worker.decode_offload_manager),
            role="decode",
        )
        self.assertEqual(report["seed_schema"]["status"], "complete")

    def test_actual_schema_builder_tag_and_arm_selectors_agree(self):
        worker, _, store = fixture()
        manager = worker.decode_offload_manager
        cc = manager.cache_controller
        base = ROOT / "python/sglang/srt"

        def load_function(path, name, namespace):
            tree = ast.parse(path.read_text())
            node = next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            node.decorator_list = []
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
                namespace,
            )
            return namespace[name]

        group = SimpleNamespace(
            entries=[
                SimpleNamespace(
                    name=name,
                    host_pool=SimpleNamespace(
                        page_size=128,
                        layout="layer_first",
                        layer_num=1,
                        dtype="torch.uint8",
                        item_bytes=512,
                    ),
                )
                for name in store.registered_pools
            ]
        )
        model = SimpleNamespace(revision=None)
        parallel = SimpleNamespace(
            attn_tp_size=1, attn_cp_size=1, pp_size=1, attn_cp_rank=0
        )
        kvcache = SimpleNamespace(
            kv_layout="KVLayout.V4",
            _unified_kv=True,
            uniform_fp8=False,
            start_layer=0,
            end_layer=2,
            layer_mapping=[
                SimpleNamespace(compress_ratio=4, compress_layer_id=0),
                SimpleNamespace(compress_ratio=128, compress_layer_id=0),
            ],
        )
        build = load_function(
            base / "mem_cache/hybrid_cache/hybrid_pool_assembler.py",
            "deepseek_v4_storage_schema",
            dict(
                get_parallel=lambda: parallel,
                get_model=lambda: model,
                shared_cache_diagnostics=SimpleNamespace(
                    record_schema=lambda schema: None
                ),
            ),
        )
        schema = build(kvcache, group)
        manager.decode_host_mem_pool.storage_schema = schema
        tag = load_function(
            base / "mem_cache/hybrid_cache/hybrid_cache_controller.py",
            "_storage_config_with_schema",
            dict(hashlib=hashlib, json=json),
        )
        cc.storage_config.extra_config = tag(
            {"extra_backend_tag": "fixture-caller"}, schema
        )
        actual = {}

        def capture(selectors, **identity):
            actual.update(selectors)
            return True

        arm = load_function(
            base / "disaggregation/decode_kvcache_offload_manager.py",
            "_arm_seed_operation",
            dict(
                os=SimpleNamespace(getenv=lambda name: "fixture-enabled"),
                re=__import__("re"),
                hashlib=hashlib,
                json=json,
                shared_cache_seed_capture=SimpleNamespace(
                    arm=capture, fail_closed=lambda: None
                ),
            ),
        )
        manager.shared_cache_d_worker_id = "fixture-worker"
        store.shared_cache_store_instance_id = "fixture-store"
        store.config = SimpleNamespace(tenant_id="fixture")
        req = SimpleNamespace(pd_diagnostic_request_ref="b" * 64, rid="fixture-request")
        transfers = [SimpleNamespace(name=name) for name in store.registered_pools]
        self.assertTrue(arm(manager, req, ["fixture-page"], transfers, 3))
        readback = observer._seed_schema(worker)
        self.assertEqual(readback["status"], "complete")
        for key in (
            "kv_schema",
            "backend_tag",
            "model_revision",
            "required_components",
            "rank",
        ):
            self.assertEqual(readback[key], actual[key])
        self.assertEqual(actual["page_range"], {"start": 3, "end": 4})
        self.assertIsNone(readback["page_range"])

    def test_empty_page_keys_and_no_capture_or_io(self):
        worker, _, store = fixture()
        original = store._get_hybrid_page_component_keys
        seen = []

        def observe(keys, transfer):
            self.assertEqual(keys, [])
            seen.append(str(transfer.name))
            return original(keys, transfer)

        store._get_hybrid_page_component_keys = observe
        self.assertEqual(observer._seed_schema(worker)["status"], "complete")
        self.assertEqual(len(seen), 2)
        # No store I/O or capture API is present on this fixture.

    def test_private_unknown_fields_and_paths_fail_closed(self):
        for change in (
            {"secret": "private-credential"},
            {"revision": "/private/model/path"},
            {"layout": "private/path"},
        ):
            worker, schema, _ = fixture()
            schema.update(change)
            result = observer._seed_schema(worker)
            self.assertEqual(result["status"], "incomplete")
            self.assertNotIn("private", json.dumps(result))
            self.assertNotIn("storage_schema", result)

    def test_bounds_registration_and_topology_fail_closed(self):
        worker, schema, _ = fixture()
        schema["layers"] = [[4, 0]] * 1025
        self.assertEqual(observer._seed_schema(worker)["status"], "incomplete")
        worker, _, _ = fixture()
        worker.decode_offload_manager.cache_controller.should_backup = lambda transfer: (
            False
        )
        self.assertEqual(observer._seed_schema(worker)["status"], "incomplete")
        worker, _, _ = fixture()
        worker.decode_offload_manager.cache_controller.storage_config.tp_size = 2
        self.assertEqual(observer._seed_schema(worker)["status"], "incomplete")
        self.assertEqual(
            observer._seed_schema(SimpleNamespace())["status"], "not_present"
        )


if __name__ == "__main__":
    unittest.main()
