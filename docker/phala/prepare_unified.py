"""Prepare one source-verified, wheel-based runtime context; never deploy."""

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


HERE = Path(__file__).resolve().parent
WRAPPER = "python/sglang/kernels/aot/python/sgl_kernel/kvcacheio.py"
ALLOWED_SHARED_OVERRIDES = {
    "protobuf",
    "grpcio-tools",
    "transformers",
    "tokenizers",
    "safetensors",
}
RUNTIME_ASSETS = {
    "muse_chat_template": (
        "muse-glimmer-chat-template.jinja",
        "/opt/muse-glimmer-chat-template.jinja",
    ),
    "nemotron_chat_template": (
        "nemotron-lightning-chat-template.jinja",
        "/opt/phala/nemotron-lightning-chat-template.jinja",
    ),
}
GOVERNOR_REPOSITORY = "https://github.com/Phala-Network/phala-inference-governor"
GOVERNOR_WHEEL_NAME = "pig-governor"
GOVERNOR_LIBRARY_NAME = "libpig_governor_core.so"
GOVERNOR_LIBRARY_DESTINATION = "/opt/phala/governor/libpig_governor_core.so"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.PIPE
    )


def _commit(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{field} must be an immutable source commit")
    return value


def _digest(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def _relative_path(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
        raise ValueError(f"{field} must be a safe relative path")
    return path.as_posix()


def identity(root, expected, tree=None, archive_sha256=None, field="Repository"):
    _commit(expected, f"{field} commit")
    actual = git(root, "rev-parse", "HEAD").decode().strip()
    if actual != expected or git(root, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError(f"{field} must be clean at the requested commit")
    if tree is not None:
        _commit(tree, f"{field} tree")
        actual_tree = git(root, "rev-parse", f"{expected}^{{tree}}").decode().strip()
        if actual_tree != tree:
            raise ValueError(f"{field} tree differs from the frozen input")
    if archive_sha256 is not None:
        archive_sha256 = _digest(archive_sha256, f"{field} archive_sha256")
        archive = git(root, "archive", "--format=tar", expected)
        if sha256(archive) != archive_sha256:
            raise ValueError(f"{field} archive differs from the frozen input")


def _artifact_path(root, value, field):
    relative = _relative_path(value, field)
    path = (root / Path(relative)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{field} leaves the input directory")
    return relative, path


def rust_modules(engine, commit):
    def read(name):
        return tomllib.loads(git(engine, "show", f"{commit}:rust/{name}").decode())

    workspace = read("Cargo.toml")["workspace"]
    manifests = [f"{member}/Cargo.toml" for member in workspace["members"]]
    manifests += (
        workspace.get("metadata", {}).get("sglang", {}).get("extension-manifests", [])
    )
    result = []
    for path in manifests:
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise ValueError("Rust manifest leaves its source root")
        module = (
            read(path)["package"]
            .get("metadata", {})
            .get("sglang", {})
            .get("python-module")
        )
        if module:
            result.append(module)
    if not result or len(result) != len(set(result)):
        raise ValueError("Rust extension discovery is empty or duplicated")
    return sorted(result)


def changed_runtime_sources(engine, upstream, commit):
    prefix = "python/"
    changes = (
        git(
            engine,
            "diff",
            "--name-status",
            "--no-renames",
            upstream,
            commit,
            "--",
            "python/sglang",
        )
        .decode()
        .splitlines()
    )
    result = {}
    for line in changes:
        status, path = line.split("\t", 1)
        if path.startswith(
            ("python/sglang/kernels/aot/", "python/sglang/kernels/kvcache_incremental/")
        ):
            continue
        if path == "python/sglang/_version.py":
            continue
        if "prelaunch" in path.lower():
            raise ValueError("Protected source is outside the runtime packaging scope")
        result[path.removeprefix(prefix)] = (
            None if status == "D" else git(engine, "show", f"{commit}:{path}")
        )
    return result


def inspect_wheel(path, expected_name, expected_version, modules=(), sources=None):
    name, version, _, _ = parse_wheel_filename(path.name)
    if name != canonicalize_name(expected_name) or version != Version(expected_version):
        raise ValueError("Wheel filename differs from the frozen input")
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate wheel members")
        for name in names:
            if (
                PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
                or "\\" in name
            ):
                raise ValueError("Unsafe wheel member")
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Expected one wheel metadata record")
        message = BytesParser().parsebytes(wheel.read(metadata[0]))
        if (
            canonicalize_name(message["Name"]) != canonicalize_name(expected_name)
            or message["Version"] != expected_version
        ):
            raise ValueError("Wheel name/version differs from the frozen input")
        for module in modules:
            prefix = module.replace(".", "/") + "."
            if (
                len(
                    [
                        name
                        for name in names
                        if name.startswith(prefix) and name.endswith(".so")
                    ]
                )
                != 1
            ):
                raise ValueError("Missing or ambiguous native extension: " + module)
        for name, expected in (sources or {}).items():
            if expected is None:
                if name in names:
                    raise ValueError("Deleted runtime source remains in wheel: " + name)
            elif name not in names or wheel.read(name) != expected:
                raise ValueError(
                    "Runtime source differs or is missing from wheel: " + name
                )


def inspect_governor(inputs, spec, governor_repo):
    if not isinstance(spec, dict) or spec.get("enabled") is not True:
        raise ValueError("Governor-enabled inputs require governor.enabled=true")
    if governor_repo is None:
        raise ValueError("Governor-enabled inputs require --governor-repo")
    if spec.get("source_repository") != GOVERNOR_REPOSITORY:
        raise ValueError("Governor source repository differs from the frozen input")
    if spec.get("default_enabled") is not False:
        raise ValueError("Governor must remain disabled by default")
    source_commit = _commit(spec.get("commit"), "Governor commit")
    source_tree = _commit(spec.get("tree"), "Governor tree")
    source_archive = _digest(spec.get("archive_sha256"), "Governor archive_sha256")
    identity(
        governor_repo,
        source_commit,
        source_tree,
        source_archive,
        "Governor repository",
    )
    version = spec.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Governor version is required")
    Version(version)
    abi = spec.get("abi_version")
    if not isinstance(abi, int) or isinstance(abi, bool) or abi <= 0:
        raise ValueError("Governor ABI version must be a positive integer")

    wheel_spec = spec.get("wheel")
    if not isinstance(wheel_spec, dict):
        raise ValueError("Governor wheel metadata is required")
    wheel_relative, wheel_path = _artifact_path(
        inputs.parent,
        wheel_spec.get("path"),
        "Governor wheel path",
    )
    if not wheel_path.is_file() or wheel_path.suffix != ".whl":
        raise ValueError("Governor wheel must be an existing .whl file")
    wheel_hash = _digest(wheel_spec.get("sha256"), "Governor wheel SHA256")
    if sha256(wheel_path.read_bytes()) != wheel_hash:
        raise ValueError("Governor wheel bytes do not match the frozen input")
    if wheel_spec.get("source_commit") != source_commit:
        raise ValueError("Governor wheel source commit differs from the repository")
    if wheel_spec.get("name") != GOVERNOR_WHEEL_NAME:
        raise ValueError("Governor wheel name differs from the frozen input")
    if wheel_spec.get("version") != version:
        raise ValueError("Governor wheel version differs from the frozen input")
    inspect_wheel(wheel_path, GOVERNOR_WHEEL_NAME, version)

    library_spec = spec.get("library")
    if not isinstance(library_spec, dict):
        raise ValueError("Governor native library metadata is required")
    library_relative, library_path = _artifact_path(
        inputs.parent,
        library_spec.get("path"),
        "Governor native library path",
    )
    if not library_path.is_file() or library_path.name != GOVERNOR_LIBRARY_NAME:
        raise ValueError("Governor native library must be libpig_governor_core.so")
    library_hash = _digest(
        library_spec.get("sha256"), "Governor native library SHA256"
    )
    if sha256(library_path.read_bytes()) != library_hash:
        raise ValueError("Governor native library bytes do not match the frozen input")
    if library_spec.get("source_commit") != source_commit:
        raise ValueError("Governor native source commit differs from the repository")
    if library_spec.get("abi_version") != abi:
        raise ValueError("Governor native ABI differs from the frozen input")
    if library_spec.get("destination") != GOVERNOR_LIBRARY_DESTINATION:
        raise ValueError("Governor native destination differs from the recipe")

    return {
        "source_repository": GOVERNOR_REPOSITORY,
        "commit": source_commit,
        "tree": source_tree,
        "archive_sha256": source_archive,
        "version": version,
        "abi_version": abi,
        "default_enabled": False,
        "wheel": {
            "name": GOVERNOR_WHEEL_NAME,
            "version": version,
            "source_commit": source_commit,
            "path": wheel_relative,
            "sha256": wheel_hash,
        },
        "library": {
            "source_commit": source_commit,
            "abi_version": abi,
            "path": library_relative,
            "sha256": library_hash,
            "destination": GOVERNOR_LIBRARY_DESTINATION,
        },
        "wheel_path": wheel_path,
        "library_path": library_path,
    }


def prepare(engine, patches, inputs, output, governor_repo=None):
    engine, patches, inputs = engine.resolve(), patches.resolve(), inputs.resolve()
    if governor_repo is not None:
        governor_repo = governor_repo.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("Output must be a fresh directory")
    output = output.resolve()
    source_roots = [engine, patches]
    if governor_repo is not None:
        source_roots.append(governor_repo)
    if any(output.is_relative_to(root) for root in source_roots):
        raise ValueError("Output cannot be inside either source checkout")
    spec = json.loads(inputs.read_text(encoding="utf-8"))
    if spec.get("schema") != "phala.sglang.unified-image-inputs.v1":
        raise ValueError("Unsupported image input schema")
    engine_spec = spec["engine"]
    identity(
        engine,
        engine_spec["commit"],
        engine_spec.get("tree"),
        engine_spec.get("archive_sha256"),
        "Engine repository",
    )
    identity(
        patches,
        spec["patches_commit"],
        spec.get("patches_tree"),
        spec.get("patches_archive_sha256"),
        "Serving-patches repository",
    )
    governor = None
    if "governor" in spec:
        governor = inspect_governor(inputs, spec["governor"], governor_repo)
    elif governor_repo is not None:
        raise ValueError("--governor-repo requires a governor object in the inputs")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", spec["base_image"]
    ):
        raise ValueError("Base image must include an immutable registry digest")
    for field in ("wrapper_preimage_sha256",):
        if not re.fullmatch(r"[0-9a-f]{64}", spec["kernel"][field]):
            raise ValueError("Kernel preimage hash is required")
    result = subprocess.run(
        [
            sys.executable,
            str(patches / "scripts/stack.py"),
            "verify",
            "--source",
            str(engine),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    verified = json.loads(result.stdout)
    if (
        verified.get("status") != "source-verified"
        or verified["engine_commit"] != spec["engine"]["commit"]
    ):
        raise ValueError("Active patch stack does not verify the chosen engine")
    version = spec["engine"]["version"]
    if not version.endswith("+phala." + spec["engine"]["commit"][:10]):
        raise ValueError("Engine wheel version must identify the source commit")
    native_spec = tomllib.loads(
        git(
            engine,
            "show",
            spec["engine"]["commit"]
            + ":python/sglang/kernels/kvcache_incremental/pyproject.toml",
        ).decode()
    )["project"]
    dependencies = json.loads((patches / "external-dependencies.json").read_text())
    versions = {
        "sglang": version,
        "xgrammar": dependencies["xgrammar"]["version"],
        native_spec["name"]: native_spec["version"],
    }
    overrides = spec.get("shared_dependency_overrides", {})
    if (
        not isinstance(overrides, dict)
        or not set(overrides) <= ALLOWED_SHARED_OVERRIDES
    ):
        raise ValueError("Unreviewed shared dependency override")
    for name, override in overrides.items():
        if (
            not isinstance(override.get("reason"), str)
            or not override["reason"].strip()
        ):
            raise ValueError("Shared dependency overrides require a reason")
        Version(override["version"])
        versions[name] = override["version"]
    wheels = spec["wheels"]
    if len(wheels) != len(versions) or {item["name"] for item in wheels} != set(
        versions
    ):
        raise ValueError(
            "Exactly the common engine/dependency wheels and declared overrides are required"
        )
    modules = rust_modules(engine, spec["engine"]["commit"])
    sources = changed_runtime_sources(
        engine, verified["upstream"], spec["engine"]["commit"]
    )
    checked = []
    for item in wheels:
        path = (inputs.parent / item["path"]).resolve()
        if (
            not path.is_file()
            or path.suffix != ".whl"
            or sha256(path.read_bytes()) != item["sha256"]
        ):
            raise ValueError("Wheel bytes do not match the frozen input")
        inspect_wheel(
            path,
            item["name"],
            versions[item["name"]],
            modules if item["name"] == "sglang" else (),
            sources if item["name"] == "sglang" else None,
        )
        checked.append((item, path))
    if len({path.name for _, path in checked}) != len(checked):
        raise ValueError("Duplicate artifact filenames")
    assets = spec.get("runtime_assets")
    if not isinstance(assets, dict) or set(assets) != set(RUNTIME_ASSETS):
        raise ValueError("Both frozen runtime chat templates are required")
    checked_assets = []
    for name, (filename, destination) in RUNTIME_ASSETS.items():
        item = assets[name]
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("Runtime asset requires path and SHA256: " + name)
        relative = Path(item["path"])
        path = (inputs.parent / relative).resolve()
        if (
            relative.is_absolute()
            or not path.is_relative_to(inputs.parent)
            or not path.is_file()
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            or sha256(path.read_bytes()) != item["sha256"]
        ):
            raise ValueError("Runtime asset bytes do not match frozen input: " + name)
        checked_assets.append((name, filename, destination, path, item["sha256"]))
    if governor is not None and governor["wheel_path"].name in {
        path.name for _, path in checked
    }:
        raise ValueError("Duplicate artifact filenames")
    wrapper = git(engine, "show", spec["engine"]["commit"] + ":" + WRAPPER)
    governor_manifest = None
    if governor is not None:
        governor_manifest = {
            "source_repository": governor["source_repository"],
            "commit": governor["commit"],
            "tree": governor["tree"],
            "archive_sha256": governor["archive_sha256"],
            "version": governor["version"],
            "abi_version": governor["abi_version"],
            "wheel": {
                **governor["wheel"],
                "path": "wheels/" + governor["wheel_path"].name,
            },
            "library": {
                **governor["library"],
                "path": "governor/" + governor["library_path"].name,
            },
            "default_enabled": False,
        }
    manifest = {
        "schema": "phala.sglang.unified-runtime-context.v1",
        "source_repository": "https://github.com/Phala-Network/sglang",
        "output_repository": "ghcr.io/phala-network/sglang",
        "engine": {**spec["engine"], "tree": verified["engine_tree"]},
        "patches_commit": spec["patches_commit"],
        "base_image": spec["base_image"],
        "kernel": {**spec["kernel"], "wrapper_sha256": sha256(wrapper)},
        "rust_modules": modules,
        "changed_runtime_files_checked": len(sources),
        "shared_dependency_overrides": overrides,
        "wheels": [
            {**item, "version": versions[item["name"]], "path": "wheels/" + path.name}
            for item, path in checked
        ],
        "runtime_assets": {
            name: {
                "path": "runtime-assets/" + filename,
                "destination": destination,
                "sha256": digest,
            }
            for name, filename, destination, _, digest in checked_assets
        },
        "governor": governor_manifest,
        "governor_default_enabled": False,
        "source_verified": True,
        "image_qualified": False,
        "boundary": "Wheel hashes, metadata and native-member presence checked; wheel execution, final-image dependencies, GPU, publication and production remain separate gates.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="unified-image-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        (staging / "wheels").mkdir()
        for _, path in checked:
            shutil.copyfile(path, staging / "wheels" / path.name)
        (staging / "runtime-assets").mkdir()
        for _, filename, _, path, _ in checked_assets:
            shutil.copyfile(path, staging / "runtime-assets" / filename)
        (staging / "governor").mkdir()
        (staging / "governor" / "README").write_text(
            "Governor artifacts are optional for this context.\n",
            encoding="utf-8",
        )
        if governor is not None:
            shutil.copyfile(
                governor["wheel_path"],
                staging / "wheels" / governor["wheel_path"].name,
            )
            shutil.copyfile(
                governor["library_path"],
                staging / "governor" / governor["library_path"].name,
            )
        (staging / "kernel-wrapper.py").write_bytes(wrapper)
        for name in (
            "Dockerfile.unified",
            "install_unified_kernel_wrapper.py",
            "install_unified_runtime.py",
        ):
            shutil.copyfile(HERE / name, staging / name)
        (staging / "runtime-context.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "source-verification.json").write_text(
            json.dumps(verified, indent=2) + "\n", encoding="utf-8"
        )
        shutil.copytree(staging, output)
    return {"status": "context-prepared", "output": str(output), **manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-repo", type=Path, required=True)
    parser.add_argument("--patches-repo", type=Path, required=True)
    parser.add_argument("--governor-repo", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.engine_repo,
                args.patches_repo,
                args.inputs,
                args.output,
                args.governor_repo,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
