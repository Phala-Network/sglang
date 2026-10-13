#!/usr/bin/env python3
"""Stage the published offload repair image context from a pinned source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


SOURCE_COMMIT = "53c8f4e63f11887a398b727dd6bc30dbe4aad73c"
REGRESSION_TEST_SHA256 = (
    "d8b0011acaac22b9b368141844896b4e3d117e4472fe9f66c0484b33b0c52ba3"
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_blob(source_repo: Path, revision: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(source_repo), "cat-file", "blob", f"{revision}:{path}"],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot read {revision}:{path}: {detail}")
    return result.stdout


def verify_revision(source_repo: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(source_repo), "rev-parse", "--verify", f"{SOURCE_COMMIT}^{{commit}}"],
        check=False,
        capture_output=True,
    )
    resolved = result.stdout.decode("ascii", errors="replace").strip()
    if result.returncode or resolved != SOURCE_COMMIT:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"source repository does not contain pinned commit {SOURCE_COMMIT}: {detail}"
        )


def load_manifest(recipe: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(
        (recipe / "image-inputs.json").read_text(encoding="utf-8")
    )
    if manifest.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("image-inputs.json source_commit does not match the pinned source")
    entries = manifest.get("runtime_files")
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("image-inputs.json must declare exactly three runtime files")
    return manifest, entries


def stage(recipe: Path, source_repo: Path, output: Path) -> dict:
    verify_revision(source_repo)
    manifest, entries = load_manifest(recipe)
    regression_test = manifest.get("regression_test")
    if not isinstance(regression_test, str):
        raise ValueError("image-inputs.json must declare regression_test")

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"staging output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(recipe / "Dockerfile", output / "Dockerfile")
    shutil.copyfile(recipe / "image-inputs.json", output / "image-inputs.json")
    overlay = output / "overlay"
    overlay.mkdir()

    staged = []
    for entry in entries:
        source_path = entry["runtime_path"]
        staged_name = entry["staged_name"]
        payload = git_blob(source_repo, SOURCE_COMMIT, source_path)
        actual = sha256(payload)
        expected = entry["after_sha256"]
        if actual != expected:
            raise ValueError(f"source hash mismatch for {source_path}: {actual}")
        target = overlay / staged_name
        target.write_bytes(payload)
        if sha256(target.read_bytes()) != expected:
            raise ValueError(f"staged hash mismatch for {source_path}")
        staged.append({"path": source_path, "staged_name": staged_name, "sha256": actual})

    test_payload = git_blob(source_repo, SOURCE_COMMIT, regression_test)
    test_hash = sha256(test_payload)
    if test_hash != REGRESSION_TEST_SHA256:
        raise ValueError(f"regression test hash mismatch: {test_hash}")

    return {
        "source_commit": SOURCE_COMMIT,
        "overlay": staged,
        "regression_test": {"path": regression_test, "sha256": test_hash},
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe = Path(__file__).resolve().parent
    report = stage(recipe, args.source_repo.resolve(), args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
