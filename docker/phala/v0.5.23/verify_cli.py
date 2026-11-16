"""Check the actual immutable image's labels, direct CLI and parser import."""

import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--nvidia-runtime",
        action="store_true",
        help="Inject host driver libraries with GPUs hidden",
    )
    args = parser.parse_args()
    if not re.fullmatch(
        r"(?:sha256:[0-9a-f]{64}|[^@\s]+@sha256:[0-9a-f]{64})", args.image
    ):
        raise ValueError("Use the exact local image ID or an immutable digest")
    expected = json.loads(args.context.read_text(encoding="utf-8"))
    inspected = json.loads(
        subprocess.check_output(["docker", "image", "inspect", args.image])
    )[0]
    labels = inspected["Config"].get("Labels") or {}
    required = {
        "org.opencontainers.image.source": expected["engine_repository"],
        "org.opencontainers.image.revision": expected["engine_commit"],
        "org.opencontainers.image.version": expected["version"],
        "org.phala.serving-patches.revision": expected["patches_commit"],
        "org.phala.recipe.revision": expected["recipe_commit"],
        "org.phala.governor.selected": "false",
    }
    if any(labels.get(key) != value for key, value in required.items()):
        raise ValueError("Final image labels differ from the frozen source context")
    args.output.mkdir(parents=True, exist_ok=False)
    common = ["docker", "run", "--rm", "--pull=never", "--network=none"]
    if args.nvidia_runtime:
        common += [
            "--runtime=nvidia",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=none",
            "-e",
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        ]
    probe = """import hashlib, json, os
from pathlib import Path
import sglang.srt.parser.reasoning_parser as parser
path = Path(parser.__file__)
assert str(path) == '/sgl-workspace/sglang/python/sglang/srt/parser/reasoning_parser.py'
assert hashlib.sha256(path.read_bytes()).hexdigest() == PARSER_SHA
assert os.environ.get('PIG_GOVERNOR_ENABLE', '').lower() not in ('1', 'true', 'yes', 'on')
print(json.dumps({'actual_parser_module': str(path), 'parser_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'governor_enabled': False}))
""".replace("PARSER_SHA", repr(expected["parser_sha256"]))
    checks = [
        ("hf-help", "hf", ["--help"]),
        ("hf-download-help", "hf", ["download", "--help"]),
        ("sglang-help", "sglang", ["serve", "--help"]),
        ("installed-parser", "python", ["-c", probe]),
    ]
    results = []
    for name, entrypoint, argv in checks:
        result = subprocess.run(
            common + ["--entrypoint", entrypoint, args.image] + argv,
            capture_output=True,
            text=True,
            timeout=90,
        )
        (args.output / (name + ".log")).write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        results.append({"check": name, "exit_code": result.returncode})
    passed = all(item["exit_code"] == 0 for item in results)
    receipt = {
        "image": args.image,
        "actual_image_id": inspected["Id"],
        "context": expected,
        "labels_match": True,
        "checks": results,
        "passed": passed,
        "source_or_executable_mounts": False,
        "runtime_packages_installed": False,
        "scope": "Final-image CLI, source import and unselected-Governor state only; not model/GPU/performance/production acceptance.",
    }
    (args.output / "RESULT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": passed, "image": inspected["Id"], "checks": results}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
