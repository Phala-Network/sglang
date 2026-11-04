"""Focused installed-image CPU qualification; no model or source mounts."""

import json
import os
import subprocess
import time
from pathlib import Path

SUITES = [
    "test/registered/unit/mem_cache/test_mooncake_read_completeness.py",
    "test/registered/unit/mem_cache/test_mooncake_group_semantics.py",
    "test/registered/unit/mem_cache/test_mooncake_store_config.py",
    "test/registered/unit/parser/test_kimik3_reasoning_parser.py",
    "test/registered/unit/function_call/test_kimik3_structural_tag.py",
    "test/registered/unit/spec/test_dflash_logits.py",
    "test/registered/unit/spec/test_dflash_model_config.py",
]
HF_SUITE = "test/registered/unit/multimodal/test_kimi_k31_native_hf.py"


def main():
    results = []
    fixture = Path(os.environ["KIMI_K31_TEST_DATA"])
    assert (fixture / "manifest.json").is_file(), "pinned HF data fixture is required"
    root = Path("/opt/testdata/k31-qualification")
    stub = root / "cpu-stubs"
    stub.mkdir(parents=True, exist_ok=True)
    (root / "tmp").mkdir(exist_ok=True)
    (stub / "libcuda.so.1").symlink_to("/usr/local/cuda/lib64/stubs/libcuda.so")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(stub) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["TMPDIR"] = str(root / "tmp")
    env["XDG_CACHE_HOME"] = str(root / "cache")
    env["HF_MODULES_CACHE"] = str(root / "hf-modules")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["GLOG_logtostderr"] = "1"
    commands = [
        ["python3", "/opt/k31-image/verify_image.py"],
        ["python3", "-m", "pytest", "-q", "--tb=short", *SUITES],
        ["python3", "-m", "pytest", "-q", "-s", "--tb=short", HF_SUITE],
        ["python3", "-B", "/opt/phala/mooncake-k31/cpu_smoke.py", str(root)],
        ["python3", "-B", "docker/phala/k31/check_fastsafetensors.py"],
        ["sglang", "serve", "--help"],
    ]
    for command in commands:
        started = time.monotonic()
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            env=env,
        )
        print(proc.stdout, flush=True)
        results.append(
            {
                "command": command,
                "exit": proc.returncode,
                "seconds": round(time.monotonic() - started, 3),
            }
        )
    print("QUALIFICATION=" + json.dumps(results), flush=True)
    return int(any(row["exit"] for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
