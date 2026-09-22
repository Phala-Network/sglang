#!/usr/bin/env python3
"""Materialize the single exact native dependency from its deterministic export.

No published Phala wheel or native Git remote is assumed. The source delta and
manifest come from the pinned serving-patches checkout used for this engine.
The destination must not exist. --verify-only performs no wheel build.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def git(root, *args):
    return (
        subprocess.check_output(
            ["git", "-c", "core.autocrlf=false", "-C", str(root), *args]
        )
        .decode()
        .strip()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-repository", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(
        (args.patch_repository / "external-dependencies.json").read_text()
    )["xgrammar"]
    if manifest["version"] != "0.2.6+phala.union1":
        raise ValueError(
            "external manifest does not match this engine's dependency pin"
        )
    if manifest["upstream_repository"] != "https://github.com/mlc-ai/xgrammar":
        raise ValueError("unexpected external upstream")
    patch = args.patch_repository / manifest["patch"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise ValueError("external source delta hash mismatch")
    if not args.verify_only and args.wheel_dir is None:
        parser.error("--wheel-dir is required for a native wheel build")
    args.source.mkdir(parents=True, exist_ok=False)
    git(args.source, "init", "--quiet")
    git(
        args.source,
        "fetch",
        "--no-tags",
        "--depth=1",
        manifest["upstream_repository"],
        manifest["upstream_commit"],
    )
    git(args.source, "checkout", "--detach", manifest["upstream_commit"])
    git(args.source, "apply", "--index", str(patch.resolve()))
    tree = git(args.source, "write-tree")
    if tree != manifest["candidate_tree"]:
        raise ValueError(f"external clean replay mismatch: {tree}")
    license_hash = hashlib.sha256(
        (args.source / manifest["license_file"]).read_bytes()
    ).hexdigest()
    if license_hash != manifest["license_sha256"]:
        raise ValueError("external license hash mismatch")
    for path, dependency in manifest["submodules"].items():
        repository = git(
            args.source, "config", "-f", ".gitmodules", "--get", f"submodule.{path}.url"
        )
        if repository != dependency["repository"]:
            raise ValueError(f"native submodule repository differs: {path}")
        entry = git(args.source, "ls-files", "--stage", "--", path).split()
        if entry[:2] != ["160000", dependency["commit"]]:
            raise ValueError(f"native submodule gitlink differs: {path}")
    print(f"Verified native source tree {tree}")
    if args.verify_only:
        return
    for path, dependency in manifest["submodules"].items():
        git(args.source, "submodule", "update", "--init", "--", path)
        if git(args.source / path, "rev-parse", "HEAD") != dependency["commit"]:
            raise ValueError(f"native submodule identity differs: {path}")
    args.wheel_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(args.source.resolve()),
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(args.wheel_dir.resolve()),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
