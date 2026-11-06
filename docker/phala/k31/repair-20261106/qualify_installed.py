"""Reuse image2's installed CPU/HF/native/CLI gates and add the frozen repair tests."""

import json
from pathlib import Path

from verify_installed import main as verify


def main():
    verify()
    root = Path("/sgl-workspace/sglang")
    manifest = json.loads(Path("/opt/phala/k31-repair/inputs.json").read_text())
    # Execute the inherited qualification harness with the installed candidate.
    # Its CUDA-stub environment, pinned data-only HF fixture and native smoke stay intact.
    harness = root / "docker/phala/k31/qualify.py"
    namespace = {"__file__": str(harness), "__name__": "image2_qualification"}
    exec(compile(harness.read_bytes(), str(harness), "exec"), namespace)
    namespace["SUITES"] = list(manifest["focused_tests"]) + namespace["SUITES"]
    return namespace["main"]()


if __name__ == "__main__":
    raise SystemExit(main())
