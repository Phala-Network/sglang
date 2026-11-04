"""Focused installed-image CPU qualification; no model or source mounts."""

import json
import subprocess
import time

SUITES = [
    "test/registered/unit/mem_cache/test_mooncake_read_completeness.py",
    "test/registered/unit/mem_cache/test_mooncake_group_semantics.py",
    "test/registered/unit/mem_cache/test_mooncake_store_config.py",
    "test/registered/unit/parser/test_kimik3_reasoning_parser.py",
    "test/registered/unit/function_call/test_kimik3_structural_tag.py",
    "test/registered/unit/spec/test_dflash_logits.py",
    "test/registered/unit/spec/test_dflash_model_config.py",
]


def main():
    results = []
    commands = [
        ["python3", "/opt/k31-image/verify_image.py"],
        ["python3", "-m", "pytest", "-q", "--tb=short", *SUITES],
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
