#!/usr/bin/env python3
"""Replay pinned p6 + budget830; only --wheel-dir explicitly builds a CPU wheel."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import zipfile
from pathlib import Path


def git(root, *args):
    return (
        subprocess.check_output(
            ["git", "-c", "core.autocrlf=false", "-C", str(root), *args]
        )
        .decode()
        .strip()
    )


def check_hash(data, expected):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("pinned input SHA256 mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-repository", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--upstream-repository", help="Optional local upstream object repository"
    )
    artifact = parser.add_mutually_exclusive_group()
    artifact.add_argument(
        "--wheel", type=Path, help="Verify the existing qualified wheel; no build"
    )
    artifact.add_argument(
        "--wheel-dir", type=Path, help="Explicitly build a new candidate"
    )
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    inputs = json.loads((here / "release-inputs.json").read_text())
    source = args.source.resolve()
    source.mkdir(parents=True, exist_ok=False)
    git(source, "init", "--quiet")
    git(
        source,
        "fetch",
        "--no-tags",
        "--depth=1",
        args.upstream_repository or inputs["upstream_repository"],
        inputs["upstream_commit"],
    )
    git(source, "checkout", "--detach", inputs["upstream_commit"])
    for patch in inputs["patches"]:
        if "local_path" in patch:
            data = (here / patch["local_path"]).read_bytes()
        else:
            data = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(args.release_repository),
                    "show",
                    inputs["release_commit"] + ":" + patch["repository_path"],
                ]
            )
        check_hash(data, patch["sha256"])
        subprocess.run(
            [
                "git",
                "-c",
                "core.autocrlf=false",
                "-C",
                str(source),
                "apply",
                "--index",
                *(["--unidiff-zero"] if patch.get("unified_context") == 0 else []),
                "-",
            ],
            input=data,
            check=True,
        )
        if git(source, "write-tree") != patch["tree"]:
            raise ValueError("native source tree differs after patch replay")
    submodule = inputs["submodule"]
    if git(source, "ls-files", "--stage", "--", submodule["path"]).split()[:2] != [
        "160000",
        submodule["commit"],
    ]:
        raise ValueError("DLPack gitlink differs")
    if (
        git(
            source,
            "config",
            "-f",
            ".gitmodules",
            "--get",
            f"submodule.{submodule['path']}.url",
        )
        != submodule["repository"]
    ):
        raise ValueError("DLPack repository differs")
    print("Verified source tree", inputs["patches"][-1]["tree"])
    if args.wheel:
        wheel = inputs["wheel"]
        check_hash(args.wheel.read_bytes(), wheel["sha256"])
        with zipfile.ZipFile(args.wheel) as archive:
            check_hash(archive.read(wheel["native_member"]), wheel["native_sha256"])
        print("Verified existing wheel and native hashes; no build performed")
    if not args.wheel_dir:
        return
    build = inputs["build"]
    if platform.python_version() != build["python"]:
        raise ValueError("build Python differs from the recorded recipe")
    for name, version in build["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"build dependency differs: {name}")
    commands = {
        "cmake": ["cmake", "--version"],
        "ninja": ["ninja", "--version"],
        "compiler": ["c++", "--version"],
    }
    for name, command in commands.items():
        line = subprocess.check_output(command, text=True).splitlines()[0]
        expected = "cmake version " + build[name] if name == "cmake" else build[name]
        if line != expected:
            raise ValueError(f"build tool differs: {name}")
    git(source, "submodule", "update", "--init", "--", submodule["path"])
    if git(source / submodule["path"], "rev-parse", "HEAD") != submodule["commit"]:
        raise ValueError("DLPack checkout differs")
    os.environ.update(
        SOURCE_DATE_EPOCH=build["source_date_epoch"],
        CMAKE_BUILD_PARALLEL_LEVEL=str(build["jobs"]),
        CXXFLAGS=build["cxxflags"].format(source=source),
    )
    destination = args.wheel_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    os.chdir(source)
    from scikit_build_core.build import build_wheel

    filename = build_wheel(str(destination), config_settings=build["settings"])
    print(
        "New build requires qualification:",
        filename,
        hashlib.sha256((destination / filename).read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    main()
