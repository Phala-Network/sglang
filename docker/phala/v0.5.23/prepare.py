"""Prepare the exact v0.5.23 source context; never build or publish an image."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = "297e54fd105e974cdf401e5b4d20253a22c16f90"
BASE = "lmsysorg/sglang:v0.5.23-cu130@sha256:f7876bef696ccf9cf75a2bf7ccb82970f5fac577f082fb8ca95c9a59d2160eb5"
RECIPE = "docker/phala/v0.5.23"


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "-C", str(repo), *args]
    )


def commit(repo, value):
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Use a complete immutable commit")
    if git(repo, "rev-parse", value + "^{commit}").decode().strip() != value:
        raise ValueError("Commit did not resolve exactly")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-repo", type=Path, required=True)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--patches-repo", type=Path, required=True)
    parser.add_argument("--patches-commit", required=True)
    parser.add_argument("--recipe-commit", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine = commit(args.engine_repo, args.engine_commit)
    patches = commit(args.patches_repo, args.patches_commit)
    recipe = commit(ROOT, args.recipe_commit)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.version):
        raise ValueError("Use the intended public version tag")
    stack = json.loads(
        git(args.patches_repo, "show", patches + ":versions/v0.5.23/stack.json")
    )
    tree = git(args.engine_repo, "rev-parse", engine + "^{tree}").decode().strip()
    if (
        stack["upstream"]["commit"] != OFFICIAL
        or stack["engine"]["commit"] != engine
        or stack["engine"]["tree"] != tree
    ):
        raise ValueError("Patch export does not bind this complete engine")
    if stack.get("governor", {}).get("selected") or stack.get("governor", {}).get(
        "embedded"
    ):
        raise ValueError("This recipe does not select Governor")
    paths = (
        git(
            args.engine_repo,
            "diff",
            "--name-only",
            OFFICIAL,
            engine,
            "--",
            "python/sglang",
        )
        .decode()
        .splitlines()
    )
    if any(not path.startswith("python/sglang/srt/") for path in paths):
        raise ValueError(
            "Package changes outside srt require an updated installation recipe"
        )
    if git(
        args.engine_repo,
        "diff",
        "--name-only",
        "--diff-filter=D",
        OFFICIAL,
        engine,
        "--",
        "python/sglang/srt",
    ).strip():
        raise ValueError("Deleted source files require an explicit packaging update")
    dockerfile = git(ROOT, "show", recipe + ":" + RECIPE + "/Dockerfile")
    if BASE.encode() not in dockerfile:
        raise ValueError("Recipe does not declare the selected official base")
    source = git(
        args.engine_repo, "archive", "--format=tar", engine, "python/sglang/srt"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(source), mode="r:") as archive:
        archive.extractall(args.output, filter="data")
    (args.output / "Dockerfile").write_bytes(dockerfile)
    build_args = {
        "SGLANG_BASE": BASE,
        "ENGINE_COMMIT": engine,
        "ENGINE_VERSION": args.version,
        "PATCHES_COMMIT": patches,
        "RECIPE_COMMIT": recipe,
    }
    context = {
        "engine_repository": "https://github.com/Phala-Network/sglang",
        "engine_commit": engine,
        "engine_tree": tree,
        "version": args.version,
        "official_commit": OFFICIAL,
        "base_image": BASE,
        "patches_commit": patches,
        "recipe_commit": recipe,
        "governor": {"selected": False, "embedded": False},
        "build_args": build_args,
        "parser_sha256": hashlib.sha256(
            git(
                args.engine_repo,
                "show",
                engine + ":python/sglang/srt/parser/reasoning_parser.py",
            )
        ).hexdigest(),
        "installation": "Complete committed srt at the official image's installed editable-package path; unchanged official CLI/native dependencies retained.",
        "qualified_or_published": False,
    }
    (args.output / "context.json").write_text(
        json.dumps(context, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "context": str(args.output),
                "engine_commit": engine,
                "engine_tree": tree,
                "build_args": build_args,
            }
        )
    )


if __name__ == "__main__":
    main()
