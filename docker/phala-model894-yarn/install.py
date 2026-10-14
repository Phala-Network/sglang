"""Bake verified GGUF deltas into the installed, immutable YaRN release."""
import base64
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil

import sglang


def digest(path, *, normalize=False):
    payload = path.read_bytes()
    if normalize:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


root = Path(__file__).resolve().parents[2]
manifest = json.loads((root / "source-manifest.json").read_text(encoding="utf8"))
installed = Path(sglang.__file__).resolve().parent
if installed == (root / "python/sglang").resolve():
    raise RuntimeError("Source context must not shadow the installed release")
if len(manifest["engine_commit"]) != 40 or len(manifest["patches_commit"]) != 40:
    raise RuntimeError("Complete engine and patch commit identities are required")
if len(manifest["runtime_sha256"]) != 13:
    raise RuntimeError("Expected exactly 13 reviewed GGUF runtime modules")

# Prove the published dynamic implementation, including rejected-request
# retirement, is present before adding the GGUF loader/vision changes.
for name, expected in manifest["base_yarn_sha256"].items():
    path = installed / Path(name).relative_to("python/sglang")
    if digest(path, normalize=True) != expected:
        raise RuntimeError(f"Published YaRN source mismatch: {name}")

copied = {}
for name, expected in manifest["runtime_sha256"].items():
    relative = Path(name).relative_to("python/sglang")
    source, target = root / name, installed / relative
    prior = manifest["base_runtime_sha256"][name]
    if prior is None:
        if target.exists():
            raise RuntimeError(f"Unexpected pre-existing GGUF module: {name}")
    elif not target.exists() or digest(target, normalize=True) != prior:
        raise RuntimeError(f"Published GGUF preimage mismatch: {name}")
    if digest(source) != expected:
        raise RuntimeError(f"Candidate source mismatch: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if digest(target) != expected:
        raise RuntimeError(f"Installed candidate mismatch: {name}")
    copied[f"sglang/{relative.as_posix()}"] = target

# Keep the wheel's standard RECORD truthful for changed and new modules.
distribution = importlib.metadata.distribution("sglang")
record = next(
    distribution.locate_file(file)
    for file in distribution.files
    if file.name == "RECORD" and ".dist-info" in str(file)
)
rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf8"))))
existing = {row[0]: row for row in rows}
for name, target in copied.items():
    encoded = base64.urlsafe_b64encode(hashlib.sha256(target.read_bytes()).digest())
    row = [name, "sha256=" + encoded.decode().rstrip("="), str(target.stat().st_size)]
    if name in existing:
        existing[name][:] = row
    else:
        rows.append(row)
with record.open("w", encoding="utf8", newline="") as stream:
    csv.writer(stream, lineterminator="\n").writerows(rows)

importlib.invalidate_caches()
for name, expected in manifest["candidate_yarn_sha256"].items():
    target = installed / Path(name).relative_to("python/sglang")
    if digest(target, normalize=True) != expected:
        raise RuntimeError(f"Candidate YaRN source mismatch after GGUF install: {name}")
policy = importlib.import_module("sglang.srt.configs.request_static_yarn")
if Path(policy.__file__).resolve() != (installed / "srt/configs/request_static_yarn.py").resolve():
    raise RuntimeError("YaRN policy is not imported from the installed package")
receipt = {
    "engine_commit": manifest["engine_commit"],
    "patches_commit": manifest["patches_commit"],
    "base_engine_commit": manifest["base_engine_commit"],
    "installed_sglang": str(installed),
    "runtime_module_count": len(copied),
    "runtime_sha256": manifest["runtime_sha256"],
    "candidate_yarn_sha256": manifest["candidate_yarn_sha256"],
    "wheel_record_updated": True,
    "native_dependencies": "unchanged from immutable phala-v0.5.21-6 base",
}
(root / "installed-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, sort_keys=True), flush=True)
