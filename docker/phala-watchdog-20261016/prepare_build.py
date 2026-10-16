#!/usr/bin/env python3
"""Stage a small immutable overlay on the accepted 0.5.21 runtime image."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BASE_IMAGE = "ghcr.io/phala-network/sglang:0.5.21-phala-offload-fix-20261012@sha256:9619a8004cbeda15ce999ab89d96b1e89b5476bd0507936253876621cc616f9e"
BASE_SOURCE = "53c8f4e63f11887a398b727dd6bc30dbe4aad73c"
VERSION = "0.5.21-phala-watchdog-dspark-fix-20261016"
RUNTIME_FILES = (
    "python/sglang/srt/entrypoints/engine.py",
    "python/sglang/srt/entrypoints/http_server.py",
    "python/sglang/srt/managers/scheduler.py",
    "python/sglang/srt/managers/scheduler_components/invariant_checker.py",
    "python/sglang/srt/managers/tokenizer_manager.py",
    "python/sglang/srt/server_args.py",
    "python/sglang/srt/speculative/dspark_components/dspark_verify.py",
    "python/sglang/srt/utils/engine_failure.py",
    "python/sglang/srt/utils/uvicorn_supervisor.py",
    "python/sglang/srt/utils/watchdog.py",
)


def git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=check, capture_output=True
    )


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--patches-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.source_repo
    if git(repo, "status", "--porcelain").stdout:
        raise SystemExit("source worktree must be clean")
    revision = git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    git(repo, "merge-base", "--is-ancestor", BASE_SOURCE, revision)
    if len(args.patches_commit) != 40 or any(
        char not in "0123456789abcdef" for char in args.patches_commit
    ):
        raise SystemExit("patches commit must be a complete SHA")
    args.output.mkdir(parents=True, exist_ok=False)
    entries = []
    for path in RUNTIME_FILES:
        blob = git(repo, "show", f"{revision}:{path}").stdout
        original = git(repo, "show", f"{BASE_SOURCE}:{path}", check=False)
        entry = {
            "path": path,
            "before_sha256": sha(original.stdout) if original.returncode == 0 else None,
            "after_sha256": sha(blob),
        }
        target = args.output / "overlay" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        entries.append(entry)
    manifest = dict(
        source_repository="https://github.com/Phala-Network/sglang",
        source_commit=revision,
        source_tag="v" + VERSION,
        patches_repository="https://github.com/Phala-Network/sglang-serving-patches",
        patches_commit=args.patches_commit,
        base_image=BASE_IMAGE,
        base_source_commit=BASE_SOURCE,
        version=VERSION,
        embedded_governor="not selected; external PIG unchanged",
        runtime_files=entries,
        builder="CPU201 / 10.80.10.201",
    )
    (args.output / "image-inputs.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    dockerfile = f'''# syntax=docker/dockerfile:1
FROM {BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \\
      org.opencontainers.image.revision="{revision}" \\
      org.opencontainers.image.version="{VERSION}" \\
      org.phala.patches.commit="{args.patches_commit}" \\
      org.phala.base.image="{BASE_IMAGE}"
ENV SGLANG_BUILD_COMMIT={revision} \\
    SGLANG_IMAGE_TAG={VERSION} \\
    SGLANG_BUILD_URL=https://github.com/Phala-Network/sglang/releases/tag/v{VERSION}
COPY overlay/ /tmp/phala-watchdog-overlay/
RUN python3 - <<'PY'
from pathlib import Path
import hashlib, json, py_compile, shutil
entries = json.loads({json.dumps(json.dumps(entries))})
root = Path('/sgl-workspace/sglang')
staging = Path('/tmp/phala-watchdog-overlay')
for entry in entries:
    target = root / entry['path']
    if entry['before_sha256'] is None:
        assert not target.exists(), entry['path']
    else:
        assert hashlib.sha256(target.read_bytes()).hexdigest() == entry['before_sha256'], entry['path']
    blob = (staging / entry['path']).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == entry['after_sha256'], entry['path']
    target.write_bytes(blob)
    py_compile.compile(str(target), doraise=True)
shutil.rmtree(staging)
print('PASS_WATCHDOG_DSPARK_OVERLAY', len(entries))
PY
'''
    (args.output / "Dockerfile").write_text(dockerfile)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
