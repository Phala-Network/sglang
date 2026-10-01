"""Fail-closed CPU gate for candidate-overlay and final-image installed modules.

Run with the image Python -I. No source/sys.modules stubs, GPU, model download,
container management, network access or dependency installation in this runner.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import signal
import sys
import sysconfig
import unittest
from pathlib import Path

MODULES = {
    "sglang.srt.entrypoints.openai.protocol": "python/sglang/srt/entrypoints/openai/protocol.py",
    "sglang.srt.entrypoints.openai.serving_chat": "python/sglang/srt/entrypoints/openai/serving_chat.py",
    "sglang.srt.entrypoints.openai.serving_base": "python/sglang/srt/entrypoints/openai/serving_base.py",
    "sglang.srt.constrained.xgrammar_schema": "python/sglang/srt/constrained/xgrammar_schema.py",
    "sglang.srt.constrained.xgrammar_backend": "python/sglang/srt/constrained/xgrammar_backend.py",
    "sglang.srt.function_call.function_call_parser": "python/sglang/srt/function_call/function_call_parser.py",
}
SUITES = {
    "test_kimi_schema_precheck_cpu": 6,
    "test_xgrammar_prevalidation_cpu": 16,
    "test_initial_stream_error_cpu": 7,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("candidate-overlay", "final-image"), required=True
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--native-sha256",
        default="778e9ac15a67a2bf731057de0834e71874e490f9fa777a627f0b4527f76ee7c4",
    )
    args = parser.parse_args()
    report = {
        "stage": args.stage,
        "image": args.image,
        "passed": False,
        "scope": "CPU actual installed modules/native matcher; not GPU or HTTP service acceptance",
        "modules": {},
        "suites": {},
    }
    try:
        if not sys.flags.isolated:
            raise RuntimeError(
                "Run with python -I; source import paths are not allowed"
            )
        if not __debug__:
            raise RuntimeError("Do not disable assertions for an acceptance gate")
        os.environ["PHALA_KIMI_INSTALLED"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        signal.alarm(210)
        expected = {
            row["path"]: row["sha256"]
            for row in json.loads(args.manifest.read_text())["files"]
        }
        purelib = Path(sysconfig.get_path("purelib")).resolve()
        # Import before adding the test directory, so a test fixture cannot
        # replace the serving modules being accepted.
        installed_paths = []
        for name, source in MODULES.items():
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(purelib):
                raise RuntimeError(
                    "Module did not import from installed site-packages: " + name
                )
            actual = sha(path)
            if actual != expected[source]:
                raise RuntimeError("Installed module hash mismatch: " + name)
            installed_paths.append(path)
            report["modules"][name] = {"path": str(path), "sha256": actual}
        xgr = importlib.import_module("xgrammar")
        report["xgrammar"] = {
            "version": importlib.metadata.version("xgrammar"),
            "path": xgr.__file__,
        }
        if report["xgrammar"]["version"] != "0.2.6+phala.union1":
            raise RuntimeError("Unexpected native XGrammar distribution")
        libs = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            if "libxgrammar_bindings.so" in line:
                libs.add(Path(line.split(maxsplit=5)[5]))
        if len(libs) != 1:
            raise RuntimeError(
                "Expected exactly one actually loaded XGrammar native library"
            )
        lib = libs.pop()
        report["xgrammar"]["native_library"] = str(lib)
        report["xgrammar"]["native_sha256"] = sha(lib)
        if report["xgrammar"]["native_sha256"] != args.native_sha256:
            raise RuntimeError("Loaded native library hash mismatch")
        if args.stage == "final-image":
            mounts = [
                Path(line.split()[4].replace("\\040", " ")).resolve()
                for line in Path("/proc/self/mountinfo").read_text().splitlines()
            ]
            for path in [*installed_paths, lib.resolve()]:
                if any(
                    mount != Path("/") and path.is_relative_to(mount)
                    for mount in mounts
                ):
                    raise RuntimeError(
                        "Final-image gate forbids runtime source/library mount overlays: "
                        + str(path)
                    )
            report["runtime_overlays"] = "none under accepted installed paths"
        else:
            report["runtime_overlays"] = (
                "candidate-overlay explicitly allowed; not final-image acceptance"
            )
        torch = importlib.import_module("torch")
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was initialized during CPU imports")
        test_dir = Path(__file__).resolve().parent
        for name in [*SUITES, Path(__file__).stem]:
            source = "test/manual/phala/" + name + ".py"
            if sha(test_dir / (name + ".py")) != expected[source]:
                raise RuntimeError("Test input hash mismatch: " + name)
        sys.path.insert(0, str(test_dir))
        for name, count in SUITES.items():
            module = importlib.import_module(name)
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            if suite.countTestCases() != count:
                raise RuntimeError("Unexpected test count in " + name)
            signal.alarm(210)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            report["suites"][name] = {
                "tests": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped),
                "passed": result.wasSuccessful() and not result.skipped,
            }
            if not result.wasSuccessful() or result.skipped:
                raise RuntimeError(
                    "Installed regression failed or skipped cases: " + name
                )
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized during CPU regression")
        report["tests"] = sum(row["tests"] for row in report["suites"].values())
        report["cuda_initialized"] = False
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
