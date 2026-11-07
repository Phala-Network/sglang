"""Export the reviewed frozen Python delta over image3; do not build or publish."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

BASE_REVISION = "204565cce1d82988c4f7235c6113fe1883c0f467"
BASE_IMAGE = "ghcr.io/phala-network/sglang:phala-dev-kimi-k31-3@sha256:08acd79a162875bc9bfded2ff42cd526a8b0b4e1c4f96e32c1997b61246f226b"
TASK_ROOT = Path("/opt/testdata/kimi-k31-video-20261107")
RECIPE = "docker/phala/k31/video-20261107"
RUNTIME_ALLOWLIST = {
    "python/sglang/srt/environ.py",
    "python/sglang/srt/entrypoints/openai/protocol.py",
    "python/sglang/srt/entrypoints/openai/serving_chat.py",
    "python/sglang/srt/mem_cache/memory_pool.py",
    "python/sglang/srt/mem_cache/unified_radix_cache.py",
    "python/sglang/srt/multimodal/kimi_k3_image_processing.py",
    "python/sglang/srt/multimodal/mm_utils.py",
    "python/sglang/srt/multimodal/processors/kimi_k3.py",
    "python/sglang/srt/models/kimi_k3.py",
    "python/sglang/srt/models/kimi_k3_vl.py",
}
CPU_TEST_SELECTION = {
    "test/registered/unit/entrypoints/openai/test_serving_chat.py": ["test/registered/unit/entrypoints/openai/test_serving_chat.py"],
    "test/registered/unit/models/test_kimi_k25.py": ["test/registered/unit/models/test_kimi_k25.py"],
    "test/registered/unit/models/test_kimi_k3_vision.py": ["test/registered/unit/models/test_kimi_k3_vision.py"],
    "test/registered/unit/models/test_kimi_k3_video.py": ["test/registered/unit/models/test_kimi_k3_video.py"],
    "test/registered/unit/mem_cache/test_mamba_unittest.py": ["test/registered/unit/mem_cache/test_mamba_unittest.py::TestMambaTemporalCopy"],
    "test/registered/unit/mem_cache/unified_radix_cache/test_unified_radix_cache_storage.py": ["test/registered/unit/mem_cache/unified_radix_cache/test_unified_radix_cache_storage.py::TestPrefetchCommitOrdering"],
}
TEST_ALLOWLIST = set(CPU_TEST_SELECTION)


def git(repo, *args, binary=False):
    data = subprocess.check_output(["git", "-C", str(repo), *args])
    return data if binary else data.decode("utf-8").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--revision", required=True, help="Exact root-frozen complete source commit")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo, output, task = args.repo.resolve(), args.output.resolve(), TASK_ROOT.resolve()
    assert all(task in p.parents for p in [repo, output]), "201 paths must stay in the exact task root"
    assert re.fullmatch(r"[0-9a-f]{40}", args.revision), "full frozen commit required"
    assert git(repo, "rev-parse", "HEAD") == args.revision
    assert not git(repo, "status", "--porcelain"), "a clean complete-source freeze is required"
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", BASE_REVISION, args.revision], check=True)
    changed = git(repo, "diff", "--name-only", BASE_REVISION, args.revision).splitlines()
    assert not git(repo, "diff", "--name-only", "--diff-filter=D", BASE_REVISION, args.revision), "deletions need separate packaging review"
    runtime = sorted(p for p in changed if p.startswith("python/"))
    tests = sorted(p for p in changed if p.startswith("test/"))
    assert runtime and set(runtime) <= RUNTIME_ALLOWLIST, "review unexpected runtime paths: " + repr(runtime)
    assert tests and set(tests) <= TEST_ALLOWLIST, "review focused regressions: " + repr(tests)
    allowed = set(runtime + tests)
    unexpected = [p for p in changed if p not in allowed and not p.startswith(RECIPE + "/") and not p.startswith("docs/")]
    assert not unexpected, "review unexpected source/build/native/dependency paths: " + repr(unexpected)
    assert not output.exists(), "use a new attempt directory"
    rootfs = output / "rootfs"
    entries = []
    for relative in runtime + tests:
        data = git(repo, "show", args.revision + ":" + relative, binary=True)
        compile(data, relative, "exec")
        destination = rootfs / "sgl-workspace/sglang" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        entries.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "kind": "runtime" if relative in runtime else "test"})
    marker = rootfs / "sgl-workspace/sglang/.k31-source-commit"
    marker.write_text(args.revision + "\n", encoding="utf-8")
    installed = rootfs / "opt/phala/k31-video"
    installed.mkdir(parents=True)
    for name in ["verify_installed.py", "qualify_installed.py"]:
        data = git(repo, "show", args.revision + ":" + RECIPE + "/" + name, binary=True)
        compile(data, name, "exec")
        (installed / name).write_bytes(data)
    selection = {path: CPU_TEST_SELECTION[path] for path in tests}
    focused_tests = [selector for selectors in selection.values() for selector in selectors]
    manifest = {"base_revision": BASE_REVISION, "base_image": BASE_IMAGE, "source_revision": args.revision,
                "version": "phala-dev-kimi-k31-4", "changed_runtime_paths": runtime, "changed_test_paths": tests,
                "focused_test_selection": selection, "focused_tests": focused_tests,
                "files": entries, "native_or_dependency_changes": False, "source_mounts": False}
    for destination in [installed / "inputs.json", output / "inputs.json"]:
        destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "Dockerfile").write_bytes(git(repo, "show", args.revision + ":" + RECIPE + "/Dockerfile", binary=True))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
