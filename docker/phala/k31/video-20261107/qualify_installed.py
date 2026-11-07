"""Reuse image3 CPU/native/HF/serve gates and add frozen video/cache regressions."""

import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

from verify_installed import main as verify


def main():
    verify()
    # Confirm image3's existing decoder packages; this recipe installs nothing.
    versions = {name: importlib.metadata.version(name) for name in ["av", "torchcodec"]}
    assert versions == {"av": "16.1.0", "torchcodec": "0.17.0+cu130"}, versions
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=15)
    assert ffmpeg.returncode == 0
    print("VIDEO_DEPENDENCIES=" + json.dumps({"packages": versions, "ffmpeg": ffmpeg.stdout.splitlines()[0]}), flush=True)
    root = Path("/sgl-workspace/sglang")
    manifest = json.loads(Path("/opt/phala/k31-video/inputs.json").read_text())
    inherited = json.loads(Path("/opt/phala/k31-repair/inputs.json").read_text())
    assert inherited["source_revision"] == manifest["base_revision"]
    harness = root / "docker/phala/k31/qualify.py"
    namespace = {"__file__": str(harness), "__name__": "inherited_image3_qualification"}
    exec(compile(harness.read_bytes(), str(harness), "exec"), namespace)
    # The inherited harness passes each SUITES item as one pytest argv element,
    # so committed file::Class selectors retain their intended CPU-only scope.
    selected = [selector for selectors in manifest["focused_test_selection"].values() for selector in selectors]
    assert selected == manifest["focused_tests"]
    assert "test/registered/unit/mem_cache/test_mamba_unittest.py" not in selected
    assert "test/registered/unit/mem_cache/unified_radix_cache/test_unified_radix_cache_storage.py" not in selected
    namespace["SUITES"] = list(dict.fromkeys(selected + inherited["focused_tests"] + namespace["SUITES"]))
    # test_utils parses a present CUDA_VISIBLE_DEVICES as a nonempty GPU list.
    # GPU isolation is enforced by runc/no DeviceRequests/NVIDIA_VISIBLE_DEVICES.
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        os.environ.pop("CUDA_VISIBLE_DEVICES")
    return namespace["main"]()


if __name__ == "__main__":
    raise SystemExit(main())
