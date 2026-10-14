"""Export committed engine bytes for the source-bound Xing image recipe."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = "c2ce5382dad45fd85e39f5d4e1d93165225eb34d"
repo = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1]).resolve()
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
subprocess.run(["git", "merge-base", "--is-ancestor", BASE, commit], cwd=repo, check=True)
changed = subprocess.check_output(["git", "diff", "--name-only", BASE, commit],
                                  cwd=repo, text=True).splitlines()
runtime = [name for name in changed if name.startswith("python/sglang/")]
inputs = set(runtime + ["LICENSE", "docker/xing4/Dockerfile",
                        "docker/xing4/install_source.py",
                        "docker/xing4/verify_cpu.sh", "docker/xing4/README.md"])
inputs.update(name for name in changed if name.startswith("test/registered/unit/"))
# Retained Qwen coverage shares the public policy and provides the real
# Scheduler/Req fixture used by Xing's lifecycle tests.
inputs.update([
    "test/registered/unit/configs/test_request_static_yarn_policy.py",
    "test/registered/unit/model_executor/test_request_static_yarn_torch.py",
    "test/registered/unit/managers/test_request_static_yarn_admission_reject.py",
])
manifest = {"engine_commit": commit, "base_engine_commit": BASE, "runtime_sha256": {}}
for name in sorted(inputs):
    blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    if name in runtime:
        manifest["runtime_sha256"][name] = hashlib.sha256(blob).hexdigest()
out.mkdir(parents=True, exist_ok=True)
(out / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"engine_commit": commit, "context": str(out), "modules": len(runtime)}))
