"""Offline install checks; run once during image build, never at service startup."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument("--revision",required=True)
args=parser.parse_args()
manifest=json.loads(Path("/opt/phala/candidate-source.json").read_text())
assert manifest["engine_commit"] == args.revision
assert metadata.version("sglang") == "0.5.22"
assert metadata.version("flashinfer-python") == "0.7.1"
assert metadata.version("torch") == "2.14.1+cu130"
import sglang
assert Path(sglang.__file__).resolve().parent == Path("/sgl-workspace/sglang/python/sglang")
for name, expected in manifest["python_files"].items():
    actual=Path("/sgl-workspace/sglang")/name
    assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected, name
wheel=Path("/tmp/xgrammar-0.2.7-cp312-cp312-linux_x86_64.whl")
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == "05cbf9fdbd294310190bc6509e655a43d8a6b374c97fc882fc681cb10facc611"
subprocess.run([sys.executable,"-m","pip","install","--no-index","--no-deps","--force-reinstall",str(wheel)],check=True)
wheel.unlink()
import xgrammar
library=Path(xgrammar.__file__).parent/"libxgrammar_bindings.so"
assert metadata.version("xgrammar") == "0.2.7"
assert hashlib.sha256(library.read_bytes()).hexdigest() == "d09f3f5f702946bbf7c0ece4e0283c6b03e210e6fb7811a4544e5b55f2e2249c"
print(json.dumps({"engine_commit":args.revision,"verified_python_files":len(manifest["python_files"]),"xgrammar_native_sha256":hashlib.sha256(library.read_bytes()).hexdigest()}))
