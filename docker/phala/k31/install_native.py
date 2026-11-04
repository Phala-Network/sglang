"""Install the frozen Mooncake client artifact without changing other modules."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import sysconfig
import tarfile
from pathlib import Path, PurePosixPath

archive = Path(sys.argv[1])
inputs = json.loads(Path(__file__).with_name("native-client.json").read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == inputs["artifact_sha256"]
site = Path(sysconfig.get_path("purelib")) / "mooncake"
assert (site / "store.so").is_file() and (site / "engine.so").is_file()
original = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in site.glob("*.so")
    if path.name not in ("store.so", "engine.so")
}
with tarfile.open(archive, "r:gz") as handle:
    manifest = json.load(handle.extractfile("native-manifest.json"))
    assert manifest["source_commit"] == inputs["source_revision"]
    assert manifest["replaced_modules"] == ["mooncake/store.so", "mooncake/engine.so"]
    for member in handle.getmembers():
        if member.isdir() or member.name == "native-manifest.json":
            continue
        path = PurePosixPath(member.name)
        assert (
            member.isfile() and path.parts[0] == "mooncake" and ".." not in path.parts
        )
        relative = PurePosixPath(*path.parts[1:])
        assert str(relative) in ("store.so", "engine.so") or (
            len(relative.parts) == 2 and relative.parts[0] == ".phala_k31_native"
        ), member.name
        target = site / str(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with handle.extractfile(member) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        target.chmod(0o755)
for name in ("store", "engine"):
    expected = manifest["elf"]["mooncake/" + name + ".so"]["sha256"]
    assert hashlib.sha256((site / (name + ".so")).read_bytes()).hexdigest() == expected
after = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in site.glob("*.so")
    if path.name not in ("store.so", "engine.so")
}
assert after == original
receipt = Path("/opt/phala/mooncake-k31")
receipt.mkdir(parents=True, exist_ok=True)
harness = receipt / "cpu_smoke.py"
assert hashlib.sha256(harness.read_bytes()).hexdigest() == inputs["cpu_smoke_sha256"]
(receipt / "native-manifest.json").write_text(json.dumps(manifest, indent=2))
(receipt / "preserved-official-native.json").write_text(json.dumps(original, indent=2))
print(
    json.dumps(
        {
            "source": inputs["source_revision"],
            "replaced": ["store.so", "engine.so"],
            "preserved_official_native": len(original),
        }
    )
)
