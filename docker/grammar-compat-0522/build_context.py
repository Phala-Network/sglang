"""Export the two common repairs from one exact engine commit for a CPU build."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("revision")
parser.add_argument("output", type=Path)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]


def git(*command):
    return subprocess.check_output(["git", "-C", str(repo), *command])


revision = git("rev-parse", args.revision + "^{commit}").decode().strip()
base_engine = "bd1ecb0fc6e056b22e7d41ecbf2c81c40ff19403"
git("merge-base", "--is-ancestor", base_engine, revision)
files = [
    "python/sglang/srt/constrained/reasoner_grammar_backend.py",
    "python/sglang/srt/entrypoints/openai/serving_base.py",
    "python/sglang/srt/entrypoints/openai/serving_chat.py",
    "python/sglang/srt/entrypoints/openai/serving_completions.py",
    "test/registered/unit/constrained/test_reasoner_backend_mask_compat.py",
    "test/registered/unit/entrypoints/openai/test_preferred_sampling_defaults.py",
]
changed_python = git("diff", "--name-only", base_engine, revision, "--", "python/")
assert set(changed_python.decode().splitlines()) == set(files[:4])
args.output.mkdir(parents=True, exist_ok=False)
hashes = {}
for name in files:
    content = git("show", revision + ":" + name)
    destination = args.output / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    hashes[name] = hashlib.sha256(content).hexdigest()
for name in ("Dockerfile", "verify.py"):
    (args.output / name).write_bytes(
        git("show", revision + ":docker/grammar-compat-0522/" + name)
    )
manifest = {
    "engine_repository": "https://github.com/Phala-Network/sglang",
    "engine_commit": revision,
    "base_engine_commit": base_engine,
    "upstream_base": "baa9befa381511cd4d56f83980951e5a45fffc6d",
    "base_image": "ghcr.io/phala-network/sglang@sha256:b07063aca12994e5dd7c1db64416cfe607c439244cb2eada2688525db6d6b1be",
    "recipe": "docker/grammar-compat-0522/Dockerfile",
    "files": hashes,
    "inheritance": "All other runtime source, dependencies and native components are inherited unchanged from the pinned base. Governor is not added or selected; external PIG is separate.",
}
(args.output / "candidate-source.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
)
print(
    json.dumps({"engine_commit": revision, "context": str(args.output.resolve())})
)
