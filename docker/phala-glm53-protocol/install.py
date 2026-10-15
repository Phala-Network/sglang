"""Install one reviewed GLM protocol module into the pinned Phala7 wheel."""

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import shutil

import sglang

RUNTIME = "python/sglang/srt/entrypoints/openai/serving_chat.py"
TEST = "test/registered/unit/entrypoints/openai/test_serving_chat.py"
BASE_COMMIT = "c2ce5382dad45fd85e39f5d4e1d93165225eb34d"
BASE_IMAGE = "ghcr.io/phala-network/sglang@sha256:6a74c66ab1c44d32d07d8f227fc5074a43a330251ea3e2b37ebe731a6178e5ee"
CONTEXT_FILES = {
    RUNTIME,
    TEST,
    "docker/phala-glm53-protocol/Dockerfile",
    "docker/phala-glm53-protocol/install.py",
}


def digest(path, *, normalize=False):
    payload = path.read_bytes()
    if normalize:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--engine-commit", required=True)
parser.add_argument("--patches-commit", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
manifest = json.loads((root / "source-manifest.json").read_text(encoding="utf8"))
for field, expected in (
    ("engine_commit", args.engine_commit),
    ("patches_commit", args.patches_commit),
):
    if not re.fullmatch(r"[0-9a-f]{40}", expected) or manifest[field] != expected:
        raise RuntimeError(f"Source identity mismatch: {field}")
if (
    manifest["base_engine_commit"] != BASE_COMMIT
    or manifest["base_image"] != BASE_IMAGE
):
    raise RuntimeError("The immutable Phala7 base identity differs")
if set(manifest["source_sha256"]) != CONTEXT_FILES:
    raise RuntimeError("Expected exactly the reviewed runtime, test and build files")
for name, expected in manifest["source_sha256"].items():
    if digest(root / name) != expected:
        raise RuntimeError(f"Source context hash mismatch: {name}")

installed = Path(sglang.__file__).resolve().parent
if installed == (root / "python/sglang").resolve():
    raise RuntimeError("Source context must not shadow the installed wheel")
relative = Path(RUNTIME).relative_to("python/sglang")
target = installed / relative
if digest(target, normalize=True) != manifest["base_runtime_sha256"]:
    raise RuntimeError("Installed serving_chat.py does not match the Phala7 preimage")
shutil.copyfile(root / RUNTIME, target)
if digest(target) != manifest["source_sha256"][RUNTIME]:
    raise RuntimeError("Installed serving_chat.py differs from the reviewed source")

distribution = importlib.metadata.distribution("sglang")
record = next(
    distribution.locate_file(file)
    for file in distribution.files
    if file.name == "RECORD" and ".dist-info" in str(file)
)
rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf8"))))
name = f"sglang/{relative.as_posix()}"
encoded = base64.urlsafe_b64encode(hashlib.sha256(target.read_bytes()).digest())
replacement = [
    name,
    "sha256=" + encoded.decode().rstrip("="),
    str(target.stat().st_size),
]
matches = [row for row in rows if row[0] == name]
if len(matches) != 1:
    raise RuntimeError("Expected one serving_chat.py entry in the wheel RECORD")
matches[0][:] = replacement
with record.open("w", encoding="utf8", newline="") as stream:
    csv.writer(stream, lineterminator="\n").writerows(rows)

receipt = {
    "engine_commit": manifest["engine_commit"],
    "patches_commit": manifest["patches_commit"],
    "base_engine_commit": BASE_COMMIT,
    "base_image": BASE_IMAGE,
    "installed_module": str(target),
    "runtime_sha256": digest(target),
    "test_sha256": manifest["source_sha256"][TEST],
    "wheel_record_updated": True,
    "native_dependencies": "unchanged from immutable Phala7 base",
}
(root / "installed-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, sort_keys=True), flush=True)
