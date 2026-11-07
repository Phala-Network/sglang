"""Verify final installed delta and complete-source identity without mounts."""

import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    root = Path("/sgl-workspace/sglang")
    manifest = json.loads(Path("/opt/phala/k31-video/inputs.json").read_text())
    assert (root / ".k31-source-commit").read_text().strip() == manifest["source_revision"]
    assert Path(importlib.util.find_spec("sglang").origin).resolve().is_relative_to(root / "python/sglang")
    for row in manifest["files"]:
        data = (root / row["path"]).read_bytes()
        assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], row["path"]
        compile(data, row["path"], "exec")
    print(json.dumps({"passed": True, "source_revision": manifest["source_revision"], "installed_files": len(manifest["files"]), "runtime_paths": manifest["changed_runtime_paths"]}))


if __name__ == "__main__":
    main()
