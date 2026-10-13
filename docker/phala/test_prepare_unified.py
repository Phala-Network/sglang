"""Git/artifact validation with a replay-service fixture, not runtime tests."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "prepare_unified_test", HERE / "prepare_unified.py"
)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class UnifiedPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.engine, self.patches = self.root / "engine", self.root / "patches"
        for repo in (self.engine, self.patches):
            repo.mkdir()
            self.git(repo, "init", "-q")
            self.git(repo, "config", "user.name", "Fixture")
            self.git(repo, "config", "user.email", "fixture@example.invalid")
        files = {
            PREPARE.WRAPPER: "old = True\n",
            "python/sglang/kernels/jit/fixture.cuh": "int old_code;\n",
            "rust/Cargo.toml": '[workspace]\nmembers = ["core"]\n[workspace.metadata.sglang]\nextension-manifests = ["tree/Cargo.toml"]\n',
            "rust/core/Cargo.toml": '[package]\nname = "core"\n[package.metadata.sglang]\npython-module = "sglang.native.core"\n',
            "rust/tree/Cargo.toml": '[package]\nname = "tree"\n[package.metadata.sglang]\npython-module = "sglang.native.tree"\n',
            "python/sglang/kernels/kvcache_incremental/pyproject.toml": '[project]\nname = "phala-kvcache-incremental"\nversion = "0.2.0"\n',
        }
        for name, text in files.items():
            path = self.engine / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.commit(self.engine)
        base = self.git(self.engine, "rev-parse", "HEAD")
        base_tree = self.git(self.engine, "rev-parse", "HEAD^{tree}")
        (self.engine / PREPARE.WRAPPER).write_text("new = True\n", encoding="utf-8")
        (self.engine / "python/sglang/kernels/jit/fixture.cuh").write_text(
            "int new_code;\n", encoding="utf-8"
        )
        self.commit(self.engine)
        engine = self.git(self.engine, "rev-parse", "HEAD")
        tree = self.git(self.engine, "rev-parse", "HEAD^{tree}")
        patch = subprocess.check_output(
            [
                "git",
                "-C",
                str(self.engine),
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-renames",
                base,
                engine,
            ]
        )
        (self.patches / "engine.patch").write_bytes(patch)
        (self.patches / "dependency.patch").write_bytes(b"native source fixture\n")
        self.write_json(
            self.patches / "manifest.json",
            {
                "upstream": {"commit": base},
                "engine": {"commit": base, "tree": base_tree},
                "patches": [],
            },
        )
        self.write_json(
            self.patches / "selectors.json",
            {
                "upstream": base,
                "active_selector": "unified",
                "complete_engine_reference": {"commit": base, "tree": base_tree},
                "selectors": {
                    "unified": {
                        "source_parent": base,
                        "fork_commit": engine,
                        "fork_tree": tree,
                        "replay_mode": "exact",
                        "patches": [
                            {
                                "id": "one",
                                "path": "engine.patch",
                                "source_commit": engine,
                                "source_parent": base,
                                "sha256": PREPARE.sha256(patch),
                            }
                        ],
                    }
                },
            },
        )
        self.write_json(
            self.patches / "external-dependencies.json",
            {
                "schema": "phala.external-source.v1",
                "xgrammar": {
                    "version": "0.2.6+phala.union1",
                    "patch": "dependency.patch",
                    "patch_sha256": PREPARE.sha256(b"native source fixture\n"),
                },
            },
        )
        (self.patches / "scripts").mkdir()
        # The patch repository owns replay tests. This fixture exercises its
        # response contract; real preparation separately executes the full stack.
        verified = {
            "status": "source-verified",
            "upstream": base,
            "engine_commit": engine,
            "engine_tree": tree,
        }
        (self.patches / "scripts/stack.py").write_text(
            "print(" + repr(json.dumps(verified)) + ")\n", encoding="utf-8"
        )
        self.commit(self.patches)
        self.spec = {
            "schema": "phala.sglang.unified-image-inputs.v1",
            "engine": {"commit": engine, "version": "0.5.20+phala." + engine[:10]},
            "patches_commit": self.git(self.patches, "rev-parse", "HEAD"),
            "base_image": "lmsysorg/sglang:v0.5.20-cu130@sha256:" + "1" * 64,
            "kernel": {"version": "0.4.7", "wrapper_preimage_sha256": "2" * 64},
            "wheels": [],
        }
        asset_dir = self.root / "assets"
        asset_dir.mkdir()
        self.spec["runtime_assets"] = {}
        for name, (filename, _) in PREPARE.RUNTIME_ASSETS.items():
            path = asset_dir / filename
            path.write_bytes((name + " template\n").encode())
            self.spec["runtime_assets"][name] = {
                "path": "assets/" + filename,
                "sha256": PREPARE.sha256(path.read_bytes()),
            }
        self.wheel(
            "sglang",
            self.spec["engine"]["version"],
            ("sglang.native.core", "sglang.native.tree"),
        )
        self.wheel("xgrammar", "0.2.6+phala.union1")
        self.wheel("phala-kvcache-incremental", "0.2.0")
        self.inputs = self.root / "inputs.json"
        self.output = self.root / "context"

    def git(self, repo, *args):
        return subprocess.check_output(
            ["git", "-c", "core.autocrlf=false", "-C", str(repo), *args],
            stderr=subprocess.PIPE,
            text=True,
        ).strip()

    def commit(self, repo):
        self.git(repo, "add", ".")
        self.git(repo, "commit", "-qm", "fixture")

    def write_json(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def wheel(
        self, name, version, modules=(), metadata_version=None, corrupt_source=False
    ):
        normalized = name.replace("-", "_")
        path = self.root / f"{normalized}-{version}-cp312-cp312-linux_x86_64.whl"
        with zipfile.ZipFile(path, "w") as wheel:
            wheel.writestr(
                f"{normalized}-{version}.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: {name}\nVersion: {metadata_version or version}\n",
            )
            for module in modules:
                wheel.writestr(
                    module.replace(".", "/") + ".cpython-312-x86_64-linux-gnu.so",
                    b"fixture, not an executable library",
                )
            if name == "sglang":
                wheel.writestr(
                    "sglang/kernels/jit/fixture.cuh",
                    b"wrong code"
                    if corrupt_source
                    else (
                        self.engine / "python/sglang/kernels/jit/fixture.cuh"
                    ).read_bytes(),
                )
        item = {
            "name": name,
            "path": path.name,
            "sha256": PREPARE.sha256(path.read_bytes()),
        }
        self.spec["wheels"] = [
            entry for entry in self.spec["wheels"] if entry["name"] != name
        ] + [item]

    def prepare(self):
        self.write_json(self.inputs, self.spec)
        return PREPARE.prepare(
            self.engine,
            self.patches,
            self.inputs,
            self.output,
            getattr(self, "governor", None),
        )

    def governor_fixture(self):
        self.governor = self.root / "governor"
        self.governor.mkdir()
        self.git(self.governor, "init", "-q")
        self.git(self.governor, "config", "user.name", "Fixture")
        self.git(self.governor, "config", "user.email", "fixture@example.invalid")
        files = {
            "pyproject.toml": "[project]\nname='pig-governor'\nversion='0.2.4'\n",
            "rust/Cargo.toml": "[package]\nname='pig_governor_core'\n",
            "rust/Cargo.lock": "version = 4\n",
            "rust/src/lib.rs": "pub const ABI_VERSION: u32 = 4;\n",
            "python/pig_governor/__init__.py": "__all__ = []\n",
        }
        for name, text in files.items():
            path = self.governor / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.commit(self.governor)
        commit = self.git(self.governor, "rev-parse", "HEAD")
        tree = self.git(self.governor, "rev-parse", "HEAD^{tree}")
        archive = subprocess.check_output(
            ["git", "-C", str(self.governor), "archive", "--format=tar", commit]
        )
        wheel = self.root / "pig_governor-0.2.4-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive_file:
            archive_file.writestr(
                "pig_governor-0.2.4.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: pig-governor\nVersion: 0.2.4\n",
            )
            archive_file.writestr("pig_governor/__init__.py", "__all__ = []\n")
        library = self.root / PREPARE.GOVERNOR_LIBRARY_NAME
        library.write_bytes(b"ELF fixture for hash validation\n")
        self.spec["governor"] = {
            "enabled": True,
            "default_enabled": False,
            "source_repository": PREPARE.GOVERNOR_REPOSITORY,
            "commit": commit,
            "tree": tree,
            "archive_sha256": PREPARE.sha256(archive),
            "version": "0.2.4",
            "abi_version": 4,
            "wheel": {
                "name": "pig-governor",
                "version": "0.2.4",
                "source_commit": commit,
                "path": wheel.name,
                "sha256": PREPARE.sha256(wheel.read_bytes()),
            },
            "library": {
                "source_commit": commit,
                "abi_version": 4,
                "path": library.name,
                "sha256": PREPARE.sha256(library.read_bytes()),
                "destination": PREPARE.GOVERNOR_LIBRARY_DESTINATION,
            },
        }

    def test_complete_stack_and_all_native_members_prepare_one_context(self):
        before = self.git(self.engine, "status", "--porcelain")
        result = self.prepare()
        self.assertEqual(result["status"], "context-prepared")
        self.assertTrue(result["source_verified"])
        self.assertFalse(result["image_qualified"])
        self.assertFalse(result["governor_default_enabled"])
        self.assertEqual(
            result["rust_modules"], ["sglang.native.core", "sglang.native.tree"]
        )
        self.assertEqual(len(list((self.output / "wheels").iterdir())), 3)
        for name, (filename, destination) in PREPARE.RUNTIME_ASSETS.items():
            self.assertEqual(
                (self.output / "runtime-assets" / filename).read_bytes(),
                (self.root / "assets" / filename).read_bytes(),
            )
            self.assertEqual(result["runtime_assets"][name]["destination"], destination)
        self.assertEqual(result["changed_runtime_files_checked"], 1)
        self.assertEqual(
            (self.output / "kernel-wrapper.py").read_bytes(),
            (self.engine / PREPARE.WRAPPER).read_bytes(),
        )
        self.assertEqual(self.git(self.engine, "status", "--porcelain"), before)

    def test_missing_rust_extension_is_not_a_pure_python_success(self):
        self.wheel("sglang", self.spec["engine"]["version"], ("sglang.native.core",))
        with self.assertRaisesRegex(ValueError, "Missing or ambiguous"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_wheel_hash_mismatch_is_rejected_before_context_creation(self):
        self.spec["wheels"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Wheel bytes"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_missing_or_changed_runtime_asset_is_rejected(self):
        item = self.spec["runtime_assets"].pop("nemotron_chat_template")
        with self.assertRaisesRegex(ValueError, "Both frozen runtime chat templates"):
            self.prepare()
        self.spec["runtime_assets"]["nemotron_chat_template"] = item
        (self.root / item["path"]).write_bytes(b"changed template\n")
        with self.assertRaisesRegex(ValueError, "Runtime asset bytes"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_metadata_version_mismatch_cannot_hide_behind_a_valid_hash(self):
        self.wheel("xgrammar", "0.2.6+phala.union1", metadata_version="0.2.1")
        with self.assertRaisesRegex(ValueError, "name/version"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_missing_common_dependency_is_rejected(self):
        self.spec["wheels"].pop()
        with self.assertRaisesRegex(ValueError, "Exactly"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_mutable_base_and_wrong_commit_are_rejected(self):
        self.spec["base_image"] = "lmsysorg/sglang:latest"
        with self.assertRaisesRegex(ValueError, "immutable registry"):
            self.prepare()
        self.spec["base_image"] += "@sha256:" + "1" * 64
        self.spec["engine"]["commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "clean at"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_dirty_source_is_preserved_and_rejected(self):
        marker = self.engine / "user-work"
        marker.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "clean at"):
            self.prepare()
        self.assertEqual(marker.read_text(), "preserve")
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / "user-work"
        marker.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.prepare()
        self.assertEqual(marker.read_text(), "preserve")

    def test_common_override_is_explicit_and_frozen(self):
        self.spec["shared_dependency_overrides"] = {
            "protobuf": {
                "version": "6.33.6",
                "reason": "Match generated proto and shared native dependency bounds",
            }
        }
        self.wheel("protobuf", "6.33.6")
        result = self.prepare()
        self.assertEqual(
            result["shared_dependency_overrides"],
            self.spec["shared_dependency_overrides"],
        )
        self.assertEqual(len(result["wheels"]), 4)

    def test_glm_vision_dependencies_are_frozen_with_the_shared_image(self):
        self.spec["shared_dependency_overrides"] = {
            name: {"version": version, "reason": "GLM-5.3 visual processing"}
            for name, version in (
                ("transformers", "5.17.0"),
                ("tokenizers", "0.23.1"),
                ("safetensors", "0.8.0"),
            )
        }
        for name, item in self.spec["shared_dependency_overrides"].items():
            self.wheel(name, item["version"])
        result = self.prepare()
        self.assertEqual(len(result["wheels"]), 6)
        self.assertEqual(
            result["shared_dependency_overrides"],
            self.spec["shared_dependency_overrides"],
        )

    def test_abi_overrides_cannot_bypass_the_pinned_base(self):
        self.spec["shared_dependency_overrides"] = {
            "torch": {"version": "2.11.0", "reason": "Unapproved ABI replacement"}
        }
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_jit_source_cannot_be_replaced_by_an_unrelated_wheel(self):
        self.wheel(
            "sglang",
            self.spec["engine"]["version"],
            ("sglang.native.core", "sglang.native.tree"),
            corrupt_source=True,
        )
        with self.assertRaisesRegex(ValueError, "Runtime source differs"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_governor_source_and_artifacts_are_frozen(self):
        self.governor_fixture()
        result = self.prepare()
        governor = result["governor"]
        self.assertEqual(governor["commit"], self.spec["governor"]["commit"])
        self.assertEqual(governor["abi_version"], 4)
        self.assertFalse(governor["default_enabled"])
        self.assertEqual(
            (self.output / "wheels" / "pig_governor-0.2.4-py3-none-any.whl").read_bytes(),
            (self.root / "pig_governor-0.2.4-py3-none-any.whl").read_bytes(),
        )
        self.assertEqual(
            (self.output / "governor" / PREPARE.GOVERNOR_LIBRARY_NAME).read_bytes(),
            (self.root / PREPARE.GOVERNOR_LIBRARY_NAME).read_bytes(),
        )

    def test_governor_native_hash_mismatch_is_rejected(self):
        self.governor_fixture()
        self.spec["governor"]["library"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "native library bytes"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_governor_must_be_disabled_by_default(self):
        self.governor_fixture()
        self.spec["governor"]["default_enabled"] = True
        with self.assertRaisesRegex(ValueError, "disabled by default"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_governor_output_cannot_be_inside_source_checkout(self):
        self.governor_fixture()
        self.output = self.governor / "context"
        with self.assertRaisesRegex(ValueError, "inside either source"):
            self.prepare()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
