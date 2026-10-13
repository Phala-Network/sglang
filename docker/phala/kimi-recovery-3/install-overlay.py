"""Install the frozen three-file repair only after all preimages match."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
lock = json.loads((here / "source-lock.json").read_text())
root = Path("/sgl-workspace/sglang")
pending = []
for name, hashes in lock["files"].items():
    relative = Path(name)
    assert not relative.is_absolute() and ".." not in relative.parts
    target = root / relative
    payload = (here / "overlay" / relative).read_bytes()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == hashes["before"], name
    assert hashlib.sha256(payload).hexdigest() == hashes["after"], name
    compile(payload, name, "exec")
    pending.append((target, payload))
for target, payload in pending:
    target.write_bytes(payload)
print("Verified and installed", len(pending), "frozen SGLang source files")
