"""Verify installed runtime bytes and the complete-source identity without mounts."""

import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    root = Path("/sgl-workspace/sglang")
    manifest = json.loads(Path("/opt/phala/k31-repair/inputs.json").read_text())
    assert (root / ".k31-source-commit").read_text().strip() == manifest["source_revision"]
    assert Path(importlib.util.find_spec("sglang").origin).resolve().is_relative_to(root / "python/sglang")
    for entry in manifest["files"]:
        data = (root / entry["path"]).read_bytes()
        assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
        compile(data, entry["path"], "exec")
    print(json.dumps({"passed": True, "source_revision": manifest["source_revision"], "installed_files": len(manifest["files"]), "runtime_paths": manifest["changed_runtime_paths"]}))


if __name__ == "__main__":
    main()
