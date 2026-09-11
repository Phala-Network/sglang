"""Build-time packaging only: restore the installed distribution's native HF CLI."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import sysconfig

from distlib.scripts import ScriptMaker


def restore():
    distribution = importlib.metadata.distribution("huggingface_hub")
    assert distribution.version == "1.30.0", "Unexpected pinned-base HF version"
    entries = [
        entry for entry in distribution.entry_points
        if entry.group == "console_scripts" and entry.name == "hf"
    ]
    assert len(entries) == 1, "Need exactly one official HF CLI entrypoint"
    entry = entries[0]
    assert entry.value == "huggingface_hub.cli.hf:main", "Unexpected HF entrypoint"
    scripts = Path(sysconfig.get_path("scripts"))
    assert str(scripts) == "/opt/sglang/bin", "Wrong runtime environment"
    target = scripts / "hf"
    assert not target.exists(), "Do not silently overwrite an existing executable"
    maker = ScriptMaker(None, str(scripts))
    maker.executable = sys.executable
    maker.variants = {""}
    maker.set_mode = True
    made = maker.make(f"hf = {entry.value}")
    assert made == [str(target)] and os.access(target, os.X_OK)
    print(json.dumps({
        "action": "restore-console-script-at-build-time",
        "package": distribution.metadata["Name"], "version": distribution.version,
        "entrypoint": entry.value, "executable": str(target),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "dependency_changes": False,
    }, sort_keys=True))


if __name__ == "__main__":
    restore()
