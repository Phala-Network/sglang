"""Install the frozen engine delta at build time and preserve wheel RECORD."""

import base64
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil

import sglang


root = Path(__file__).resolve().parents[2]
manifest = json.loads((root / "source-manifest.json").read_text())
installed = Path(sglang.__file__).resolve().parent
if len(manifest["engine_commit"]) != 40:
    raise RuntimeError("A full engine commit is required")
if installed == (root / "python/sglang").resolve():
    raise RuntimeError("Build source masks the installed package")

copied = {}
for name, expected in manifest["runtime_sha256"].items():
    source = root / name
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Source mismatch: {name}")
    relative = Path(name).relative_to("python/sglang")
    target = installed / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Installed mismatch: {name}")
    copied[f"sglang/{relative.as_posix()}"] = target

if not copied:
    raise RuntimeError("The frozen engine delta is empty")
distribution = importlib.metadata.distribution("sglang")
record = next(distribution.locate_file(file) for file in distribution.files
              if file.name == "RECORD" and ".dist-info" in str(file))
rows = list(csv.reader(io.StringIO(record.read_text())))
existing = {row[0]: row for row in rows}
for name, target in copied.items():
    checksum = base64.urlsafe_b64encode(hashlib.sha256(target.read_bytes()).digest())
    row = [name, "sha256=" + checksum.decode().rstrip("="), str(target.stat().st_size)]
    if name in existing:
        existing[name][:] = row
    else:
        rows.append(row)
with record.open("w", newline="") as stream:
    csv.writer(stream, lineterminator="\n").writerows(rows)

(root / "installed-source.json").write_text(json.dumps({
    "engine_commit": manifest["engine_commit"],
    "base_engine_commit": manifest["base_engine_commit"],
    "installed_package": str(installed),
    "runtime_sha256": manifest["runtime_sha256"],
    "source_overlay": False,
}, indent=2) + "\n")
