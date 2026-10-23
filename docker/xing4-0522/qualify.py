"""Qualify the installed candidate image on CPU, without mounts or installs."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path("/sgl-workspace/sglang")
MANIFEST = Path("/opt/phala/candidate-source.json")
TESTS = [
    "models/test_xing4_native.py",
    "models/test_xing4_mhc_sinkhorn.py",
    "models/test_xing4_runtime_auth.py",
    "models/test_xing4_tokenizer.py",
    "managers/test_xing_native_eagle_public_budget.py",
    "utils/test_token_env_auth.py",
    "function_call/test_allowed_tools_choice.py",
    "function_call/test_named_tool_reference_scope.py",
    "entrypoints/openai/test_allowed_tools_output.py",
    "test_openai_usage_processor.py",
    "constrained/test_xing4_serving_contract.py",
]


def main():
    manifest = json.loads(MANIFEST.read_text())
    assert not os.environ.get("PYTHONPATH"), "Use installed package, without source path override"
    import sglang
    import xgrammar
    assert Path(sglang.__file__).resolve().parent == ROOT / "python/sglang"
    for name, expected in {**manifest["python_files"], **manifest["test_files"]}.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    versions = {name: metadata.version(name) for name in (
        "sglang", "flashinfer-python", "torch", "xgrammar", "huggingface-hub")}
    assert versions["sglang"] == "0.5.22"
    assert versions["flashinfer-python"] == "0.7.1"
    assert versions["torch"] == "2.14.1+cu130"
    assert versions["xgrammar"] == "0.2.7"
    native_sha = hashlib.sha256((Path(xgrammar.__file__).parent / "libxgrammar_bindings.so").read_bytes()).hexdigest()
    assert native_sha == "d09f3f5f702946bbf7c0ece4e0283c6b03e210e6fb7811a4544e5b55f2e2249c"
    help_checks = []
    for argv, expected in [
        (["hf", "--help"], ["download"]),
        (["hf", "download", "--help"], ["--local-dir"]),
        (["sglang", "serve", "--help"], ["hrrn", "--enable-metrics", "--enable-cache-report", "--uvicorn-access-log-exclude-prefixes"]),
    ]:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr[-2000:]
        assert all(word in result.stdout for word in expected), argv
        help_checks.append({"argv": argv, "returncode": result.returncode,
            "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest()})
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.runtime_context import reset_context
    from sglang.srt.arg_groups.overrides import resolving_view
    os.environ["PIG_AUTH_FROM_TOKEN"] = "1"
    token = secrets.token_hex(24)
    os.environ["TOKEN"] = token
    args = prepare_server_args(["--model-path", "dummy", "--schedule-policy", "hrrn",
        "--enable-metrics", "--enable-cache-report", "--disable-overlap-schedule",
        "--uvicorn-access-log-exclude-prefixes", "/health", "/metrics"])
    args.resolve_once()
    view = resolving_view(args)
    assert view.schedule_policy == "hrrn" and view.enable_metrics and view.enable_cache_report
    assert view.uvicorn_access_log_exclude_prefixes == ["/health", "/metrics"]
    assert secrets.compare_digest(view.api_key, token) and secrets.compare_digest(view.admin_api_key, token)
    assert token not in json.dumps(args.diagnostic_dict()) and token not in args.launch_command
    os.environ.pop("TOKEN")
    os.environ.pop("PIG_AUTH_FROM_TOKEN")
    reset_context()
    junit = Path("/tmp/xing4-0522-cpu.xml")
    argv = [sys.executable, "-m", "pytest", "-q", "--disable-warnings",
        "-o", "cache_dir=/tmp/xing4-0522-pytest-cache", "--junitxml=" + str(junit)]
    argv += [str(ROOT / "test/registered/unit" / path) for path in TESTS]
    result = subprocess.run(argv, cwd=ROOT, check=False)
    counts = {}
    if junit.exists():
        xml = ET.parse(junit).getroot()
        suites = [xml] if xml.tag == "testsuite" else list(xml)
        counts = {k: sum(int(x.attrib.get(k, "0")) for x in suites)
                  for k in ("tests", "failures", "errors", "skipped")}
    receipt = {"engine_commit": manifest["engine_commit"], "engine_tree": manifest["engine_tree"],
        "installed_import_path": str(Path(sglang.__file__).resolve()), "versions": versions,
        "xgrammar_native_sha256": native_sha, "help_checks": help_checks,
        "official_cli_resolution": "pass", "pytest_returncode": result.returncode, "junit": counts,
        "scope": "CPU installed-image gates only; GPU numerical/model/protocol/context gates remain"}
    print("XING4_FINAL_CPU_RECEIPT=" + json.dumps(receipt, sort_keys=True), flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
