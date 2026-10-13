"""Install a source-bound runtime delta while retaining verified native dependencies."""

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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path(__file__).resolve().parents[2]
manifest = json.loads((root / "source-manifest.json").read_text(encoding="utf8"))
installed = Path(sglang.__file__).resolve().parent
source_root = root / "python/sglang"
if installed == source_root.resolve():
    raise RuntimeError("sglang resolves to build source rather than installed package")
if len(manifest["engine_commit"]) != 40:
    raise RuntimeError("A complete engine source commit is required")

copied = {}
for name, expected in manifest["runtime_sha256"].items():
    relative = Path(name).relative_to("python/sglang")
    source, target = source_root / relative, installed / relative
    if digest(source) != expected:
        raise RuntimeError(f"Source hash mismatch: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if digest(target) != expected:
        raise RuntimeError(f"Installed hash mismatch: {target}")
    copied[f"sglang/{relative.as_posix()}"] = target
if len(copied) != 11:
    raise RuntimeError("The qualified runtime delta must contain exactly 11 modules")

# Keep wheel metadata truthful after the immutable build-time source delta.
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
module = importlib.import_module("sglang.srt.configs.request_static_yarn")
if Path(module.__file__).resolve() != (installed / "srt/configs/request_static_yarn.py").resolve():
    raise RuntimeError("YaRN module resolves outside the installed package")
receipt = {
    "engine_commit": manifest["engine_commit"],
    "tested_candidate_identity": manifest["tested_candidate_identity"],
    "installed_sglang": str(installed),
    "runtime_module_count": len(copied),
    "runtime_sha256": manifest["runtime_sha256"],
    "wheel_record_updated": True,
    "native_dependencies": "unchanged from immutable phala-v0.5.21-5 base",
}
(root / "installed-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, sort_keys=True), flush=True)
