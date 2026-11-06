"""Export a frozen, committed Python-only delta; never build or publish an image."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

BASE_REVISION = "254af1d9de5eb6baf68d72e961dcb3116969acfa"
BASE_IMAGE = "ghcr.io/phala-network/sglang:phala-dev-kimi-k31-2@sha256:edc2bb25b3c7f7d67a902e0b7e3a99f1ef8a1371e8395f950ecbb06cb9f3db62"
TASK_ROOT = Path("/opt/testdata/kimi-k31-usw3-repair-20261106")
RECIPE = "docker/phala/k31/repair-20261106"
RUNTIME_ALLOWLIST = {
    "python/sglang/srt/managers/scheduler.py",
    "python/sglang/srt/managers/schedule_policy.py",
    "python/sglang/srt/managers/schedule_batch.py",
    "python/sglang/srt/mem_cache/base_prefix_cache.py",
    "python/sglang/srt/mem_cache/unified_cache/components/mamba.py",
}
TEST_PREFIX = "test/registered/unit/managers/test_scheduler_hybrid_ssm_admission_"


def git(repo, *args, binary=False):
    data = subprocess.check_output(["git", "-C", str(repo), *args])
    return data if binary else data.decode("utf-8").strip()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--revision", required=True, help="Root-frozen full source commit")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    task = TASK_ROOT.resolve()
    assert task in output.parents and task in repo.parents, "201 paths must stay in the exact task root"
    assert re.fullmatch(r"[0-9a-f]{40}", args.revision), "full frozen commit required"
    assert git(repo, "rev-parse", "HEAD") == args.revision, "checkout must match the frozen commit"
    assert not git(repo, "status", "--porcelain"), "freeze a clean source checkout first"
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", BASE_REVISION, args.revision], check=True)
    changed = git(repo, "diff", "--name-only", BASE_REVISION, args.revision).splitlines()
    runtime = sorted(p for p in changed if p.startswith("python/"))
    tests = sorted(p for p in changed if p.startswith(TEST_PREFIX) and p.endswith(".py"))
    assert runtime and set(runtime) <= RUNTIME_ALLOWLIST, runtime
    assert tests, "the frozen candidate must contain its focused CPU regressions"
    allowed = set(runtime + tests)
    unexpected = [p for p in changed if p not in allowed and not p.startswith(RECIPE + "/") and not p.startswith("docs/")]
    assert not unexpected, "review unexpected source/build/native changes: " + repr(unexpected)
    assert not output.exists(), "use a fresh context; do not overwrite a previous attempt"
    rootfs = output / "rootfs"
    entries = []
    for relative in runtime + tests:
        data = git(repo, "show", args.revision + ":" + relative, binary=True)
        compile(data, relative, "exec")
        destination = rootfs / "sgl-workspace/sglang" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        entries.append({"path": relative, "sha256": sha(data), "bytes": len(data), "kind": "runtime" if relative in runtime else "test"})
    marker = rootfs / "sgl-workspace/sglang/.k31-source-commit"
    marker.write_text(args.revision + "\n", encoding="utf-8")
    installed = rootfs / "opt/phala/k31-repair"
    installed.mkdir(parents=True)
    for name in ["verify_installed.py", "qualify_installed.py"]:
        (installed / name).write_bytes(git(repo, "show", args.revision + ":" + RECIPE + "/" + name, binary=True))
    manifest = {
        "base_revision": BASE_REVISION,
        "base_image": BASE_IMAGE,
        "source_revision": args.revision,
        "version": "phala-dev-kimi-k31-3",
        "changed_runtime_paths": runtime,
        "focused_tests": tests,
        "files": entries,
        "native_or_dependency_changes": False,
        "source_mounts": False,
    }
    (installed / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "Dockerfile").write_bytes(git(repo, "show", args.revision + ":" + RECIPE + "/Dockerfile", binary=True))
    (output / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
