"""Check the installed repair bytes and inherited packages during image build."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import xgrammar

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--revision", required=True)
args = parser.parse_args()
manifest = json.loads(Path("/opt/phala/candidate-source.json").read_text())
assert manifest["engine_commit"] == args.revision
for name, expected in manifest["files"].items():
    actual = Path("/sgl-workspace/sglang") / name
    assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected, name
packages = {
    "sglang": "0.5.22",
    "torch": "2.14.1+cu130",
    "llguidance": "1.9.1",
    "xgrammar": "0.2.7",
    "transformers": "5.19.0",
}
for name, expected in packages.items():
    assert metadata.version(name) == expected, name
native = Path(xgrammar.__file__).parent / "libxgrammar_bindings.so"
native_sha256 = hashlib.sha256(native.read_bytes()).hexdigest()
assert native_sha256 == "d09f3f5f702946bbf7c0ece4e0283c6b03e210e6fb7811a4544e5b55f2e2249c"
print(
    json.dumps(
        {
            "engine_commit": args.revision,
            "packages": packages,
            "verified_files": len(manifest["files"]),
            "xgrammar_native_sha256": native_sha256,
        }
    )
)
