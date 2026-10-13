"""Install the fixed wheel set inside a Docker build RUN, not at startup."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


GOVERNOR_WHEEL_NAME = "pig-governor"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--patches-commit", required=True)
    parser.add_argument("--governor-commit", default="")
    parser.add_argument("--governor-version", default="")
    parser.add_argument("--governor-abi", default="")
    args = parser.parse_args()
    context = json.loads(args.context.read_text())
    if (
        context["schema"] != "phala.sglang.unified-runtime-context.v1"
        or context["engine"]["commit"] != args.engine_commit
        or context["patches_commit"] != args.patches_commit
        or context["base_image"] != args.base_image
        or context["governor_default_enabled"] is not False
    ):
        raise ValueError("Docker arguments differ from the prepared source context")
    governor = context.get("governor")
    supplied_governor = (
        args.governor_commit,
        args.governor_version,
        args.governor_abi,
    )
    if governor is None:
        if any(supplied_governor):
            raise ValueError("Docker Governor arguments were supplied without Governor artifacts")
    elif (
        not all(supplied_governor)
        or governor["commit"] != args.governor_commit
        or governor["version"] != args.governor_version
        or str(governor["abi_version"]) != args.governor_abi
        or governor["default_enabled"] is not False
    ):
        raise ValueError("Docker arguments differ from the prepared Governor context")
    wheels = []
    for item in context["wheels"]:
        path = args.wheel_dir / Path(item["path"]).name
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Build-context wheel changed")
        wheels.append(str(path))
    governor_wheel = None
    if governor is not None:
        item = governor["wheel"]
        governor_wheel = args.wheel_dir / Path(item["path"]).name
        if (
            not governor_wheel.is_file()
            or hashlib.sha256(governor_wheel.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError("Governor wheel changed")
        wheels.append(str(governor_wheel))
    env = dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    if governor is not None:
        library = Path(governor["library"]["destination"])
        if (
            not library.is_file()
            or hashlib.sha256(library.read_bytes()).hexdigest()
            != governor["library"]["sha256"]
        ):
            raise ValueError("Governor native library changed")
        env["PIG_GOVERNOR_LIBRARY"] = str(library)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            *wheels,
        ],
        env=env,
        check=True,
    )
    for item in context["wheels"]:
        if importlib.metadata.version(item["name"]) != item["version"]:
            raise RuntimeError("Installed distribution differs from the frozen wheel")
    if governor is not None:
        if importlib.metadata.version(GOVERNOR_WHEEL_NAME) != governor["version"]:
            raise RuntimeError("Installed Governor distribution differs from the frozen wheel")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from pig_governor import Governor; "
                "g=Governor(1.0, max_running_requests=1); "
                "assert g._lib.pig_governor_abi_version() == %d; g.close()"
                % governor["abi_version"],
            ],
            env=env,
            check=True,
        )
    kernel = context["kernel"]
    subprocess.run(
        [
            sys.executable,
            "install_unified_kernel_wrapper.py",
            "--wrapper",
            "kernel-wrapper.py",
            "--expected-sha256",
            kernel["wrapper_sha256"],
            "--base-sha256",
            kernel["wrapper_preimage_sha256"],
            "--base-version",
            kernel["version"],
            "--receipt",
            "kernel-wrapper-install.json",
        ],
        env=env,
        check=True,
    )
    for command in (
        ["hf", "--help"],
        ["hf", "download", "--help"],
        ["sglang", "serve", "--help"],
    ):
        subprocess.run(command, env=env, check=True)
    # Inherited base conflicts are evidence, never silently converted to a pass.
    check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], env=env, capture_output=True, text=True
    )
    Path("dependency-check.txt").write_text(check.stdout + check.stderr)
    Path("installation.json").write_text(
        json.dumps(
            {
                "engine": context["engine"],
                "governor": governor,
                "wheel_installation": "passed",
                "cli_build_layer_checks": "passed",
                "dependency_check_exit": check.returncode,
                "release_qualified": False,
                "boundary": "Repeat final image CLI/native/functional gates; dependency findings require explicit review.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
