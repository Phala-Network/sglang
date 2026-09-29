"""Real clear-manifest validation, independent of POSIX private-file permissions."""

import ast
import hashlib
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

spec = importlib.util.spec_from_file_location(
    "clear_control", ROOT / "python/sglang/srt/managers/shared_cache_control.py"
)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def manifest():
    document = {
        "manifest_version": 1,
        "manifest_id": "fixture",
        "run_id": "run",
        "case_id": "case",
        "epoch": "epoch",
        "model_revision": None,
        "kv_schema": "schema",
        "tenant": "default",
        "backend_tag": "backend",
        "original_D_worker_id": "worker",
        "original_writer_client_id": "1-2",
        "original_store_instance_id": "store",
        "exact_nonempty_MEMORY_segment": "segment",
        "key_salt": "fixture-salt",
        "required_components": ["kv"],
        "keys": [
            {
                "key": "fixture-key",
                "logical_bytes": 16,
                "rank": 0,
                "component": "kv",
                "page_range": "2-3",
                "ssd_replica_ids": [1],
                "ssd_owner_uuid": "3-4",
                "ssd_scope": "scope",
            }
        ],
    }
    document["keys"][0]["key_id"] = control._stable_id(
        b"fixture-salt",
        b"phala.shared-cache-key.v1\0",
        "case",
        "epoch",
        "default",
        "fixture-key",
    )
    return document


def derived_backend_tag():
    # Execute the real producer method without importing CUDA dependencies.
    source = (
        ROOT / "python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_storage_config_with_schema"
    )
    method.decorator_list = []
    namespace = {"hashlib": hashlib, "json": json}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["_storage_config_with_schema"](
        {}, {"model_revision": None, "page_size": 256}
    )["extra_backend_tag"]


def load(document):
    # Only the private OS-file boundary is substituted on Windows. Actual
    # load_manifest, canonical hash, key HMAC and all metadata checks execute.
    document["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    with patch.object(control, "_private_json", return_value=document):
        return control.load_manifest(
            "fixture.json", "fixture", document["manifest_sha256"]
        )


class TestClearManifestRevision(unittest.TestCase):
    def test_explicit_null_is_preserved(self):
        result = load(manifest())
        self.assertIn("model_revision", result)
        self.assertIsNone(result["model_revision"])

    def test_existing_nonnull_labels_are_preserved(self):
        for value in ("main", "revision-1", "a" * 64):
            with self.subTest(value=value):
                document = manifest()
                document["model_revision"] = value
                self.assertEqual(load(document)["model_revision"], value)

    def test_actual_derived_tag_colon_components_and_null_revision(self):
        document = manifest()
        document["backend_tag"] = derived_backend_tag()
        self.assertEqual(len(document["backend_tag"]), 72)
        document["required_components"] = ["swa:0", "swa:1"]
        original = document["keys"][0]
        document["keys"] = []
        for component in document["required_components"]:
            item = dict(original, component=component, key="fixture-" + component)
            item["key_id"] = control._stable_id(
                b"fixture-salt",
                b"phala.shared-cache-key.v1\0",
                "case",
                "epoch",
                "default",
                item["key"],
            )
            document["keys"].append(item)
        result = load(document)
        self.assertEqual(result["backend_tag"], document["backend_tag"])
        self.assertEqual(result["required_components"], ["swa:0", "swa:1"])
        self.assertIsNone(result["model_revision"])
        self.assertEqual(result["keys"], document["keys"])

    def test_backend_tag_bad_types_grammar_and_lengths(self):
        for value in (
            None,
            False,
            1,
            [],
            {},
            "",
            "a" * 65,
            "dsv4-v1-" + "a" * 63,
            "dsv4-v1-" + "a" * 65,
            "dsv4-v1-" + "G" * 64,
            "backend/tag",
            "backend:0",
        ):
            with self.subTest(value=value):
                document = manifest()
                document["backend_tag"] = value
                with self.assertRaisesRegex(
                    control.SharedCacheControlError, "invalid_manifest_metadata"
                ):
                    load(document)

    def test_component_grammar_and_length_in_both_locations(self):
        for value in (
            None,
            False,
            1,
            [],
            {},
            "",
            "a" * 65,
            "pool:",
            ":0",
            "pool:-1",
            "pool:01",
            "pool:0:1",
            "pool/name:0",
            "p" * 65 + ":0",
            "p:" + "1" * 95,
        ):
            for field in ("required_components", "component"):
                with self.subTest(value=value, field=field):
                    document = manifest()
                    if field == "required_components":
                        document[field] = [value]
                    else:
                        document["keys"][0][field] = value
                    with self.assertRaisesRegex(
                        control.SharedCacheControlError, "invalid_component"
                    ):
                        load(document)

    def test_field_grammar_does_not_widen_general_labels(self):
        for field in ("run_id", "case_id", "epoch", "tenant", "kv_schema"):
            for value in (derived_backend_tag(), "pool:0"):
                with self.subTest(field=field, value=value):
                    document = manifest()
                    document[field] = value
                    with self.assertRaisesRegex(
                        control.SharedCacheControlError, "invalid_manifest_metadata"
                    ):
                        load(document)
        for value in ("legacy-tag", "a" * 64):
            document = manifest()
            document["backend_tag"] = value
            document["required_components"] = [value]
            document["keys"][0]["component"] = value
            self.assertEqual(load(document)["backend_tag"], value)

    def test_missing_is_not_null(self):
        document = manifest()
        del document["model_revision"]
        with self.assertRaisesRegex(
            control.SharedCacheControlError, "invalid_manifest_metadata"
        ):
            load(document)

    def test_bad_types_empty_and_invalid_labels_rejected(self):
        for value in (False, True, 1, 0, [], {}, "", " ", "branch/path", "a" * 65):
            with self.subTest(value=value):
                document = manifest()
                document["model_revision"] = value
                with self.assertRaisesRegex(
                    control.SharedCacheControlError, "invalid_manifest_metadata"
                ):
                    load(document)


if __name__ == "__main__":
    unittest.main()
