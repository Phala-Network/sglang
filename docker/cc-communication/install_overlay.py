"""Apply only the reviewed pinned-base overlay; fail closed on source drift."""

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "overlay-manifest.json").read_text())
roots = {name: Path(importlib.util.find_spec(name).origin).parent
         for name in ("sglang", "flashinfer")}

for name, expected in manifest["versions"].items():
    assert importlib.metadata.version(name) == expected, (name, expected)
for item in manifest["files"]:
    target = roots[item["package"]] / item["path"]
    if item["before"] is None:
        assert not target.exists(), f"Unexpected pre-existing source: {target}"
    else:
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual == item["before"], f"Base source drift: {target}: {actual}"

subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p1"],
    input=(HERE / "flashinfer-0.6.18-ipc-guard.patch").read_bytes(),
    cwd=roots["flashinfer"].parent, check=True)
for item in manifest["files"]:
    if item.get("copy"):
        shutil.copyfile(HERE / item["copy"], roots[item["package"]] / item["path"])
for item in manifest["files"]:
    target = roots[item["package"]] / item["path"]
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    assert actual == item["after"], f"Overlay hash mismatch: {target}: {actual}"
print(json.dumps({"overlay_verified": True, "files": manifest["files"]}, indent=2))
