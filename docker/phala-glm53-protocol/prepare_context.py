"""Export the small GLM build context from a complete immutable engine commit."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

BASE_COMMIT = "c2ce5382dad45fd85e39f5d4e1d93165225eb34d"
BASE_IMAGE = "ghcr.io/phala-network/sglang@sha256:6a74c66ab1c44d32d07d8f227fc5074a43a330251ea3e2b37ebe731a6178e5ee"
RUNTIME = "python/sglang/srt/entrypoints/openai/serving_chat.py"
FILES = (
    RUNTIME,
    "test/registered/unit/entrypoints/openai/test_serving_chat.py",
    "docker/phala-glm53-protocol/Dockerfile",
    "docker/phala-glm53-protocol/install.py",
)


def git(*args):
    return subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), *args],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


parser = argparse.ArgumentParser()
parser.add_argument("--engine-ref", required=True)
parser.add_argument("--patches-commit", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if not re.fullmatch(r"[0-9a-f]{40}", args.patches_commit):
    raise ValueError("An immutable complete patch-set commit is required")
engine = (
    git("rev-parse", "--verify", "--end-of-options", args.engine_ref + "^{commit}")
    .decode()
    .strip()
)
git("merge-base", "--is-ancestor", BASE_COMMIT, engine)
if args.output.exists() and any(args.output.iterdir()):
    raise ValueError("Build context output must be new or empty")
args.output.mkdir(parents=True, exist_ok=True)
hashes = {}
for name in FILES:
    payload = git("show", f"{engine}:{name}")
    target = args.output / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    hashes[name] = hashlib.sha256(payload).hexdigest()
manifest = {
    "engine_commit": engine,
    "engine_tree": git("rev-parse", engine + "^{tree}").decode().strip(),
    "patches_commit": args.patches_commit,
    "base_engine_commit": BASE_COMMIT,
    "base_image": BASE_IMAGE,
    "base_runtime_sha256": hashlib.sha256(
        git("show", f"{BASE_COMMIT}:{RUNTIME}")
    ).hexdigest(),
    "source_sha256": hashes,
}
(args.output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
