"""CPU checks of an exact built image, with no source or executable mounts."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


PROBE = r"""
from array import array
import hashlib
import importlib
import importlib.metadata as metadata
import io
import json
import logging
import warnings
import os
from pathlib import Path
import numpy as np
import torch
import sglang
assert not torch.cuda.is_available(), "CPU gate must not claim GPU qualification"
actual = json.loads(Path("/opt/phala/runtime/runtime-context.json").read_text())
assert actual == EXPECTED
assert "/site-packages/" in sglang.__file__, sglang.__file__
for item in EXPECTED["wheels"]:
    assert metadata.version(item["name"]) == item["version"], item["name"]
for name, item in EXPECTED["runtime_assets"].items():
    path = Path(item["destination"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], name
    from jinja2 import Environment
    Environment().parse(path.read_text(encoding="utf-8"))
if EXPECTED.get("governor") is not None:
    governor = EXPECTED["governor"]
    assert metadata.version("pig-governor") == governor["version"]
    library = Path(governor["library"]["destination"])
    assert hashlib.sha256(library.read_bytes()).hexdigest() == governor["library"]["sha256"]
    assert os.environ.get("PIG_GOVERNOR_LIBRARY") == str(library)
    from pig_governor import Governor
    handle = Governor(1.0, max_running_requests=1)
    assert handle._lib.pig_governor_abi_version() == governor["abi_version"]
    handle.close()
wrapper = metadata.distribution("sglang-kernel").locate_file("sgl_kernel/kvcacheio.py")
assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == EXPECTED["kernel"]["wrapper_sha256"]
receipt = json.loads(Path("/opt/phala/runtime/kernel-wrapper-install.json").read_text())
assert receipt["source_owned_wrapper_modified"]
modules = {}
for name in EXPECTED["rust_modules"]:
    module = importlib.import_module(name)
    assert "/site-packages/" in module.__file__, module.__file__
    modules[name] = module.__file__
import phala_kvcache_incremental as kv
kv.ensure_loaded()
assert len(kv._REQUIRED_OPS) == 15
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
assert DeepseekV4ForCausalLM.__name__ == "DeepseekV4ForCausalLM"
if "transformers" in EXPECTED["shared_dependency_overrides"]:
    from transformers.models.glm5_next.processing_glm5_next import Glm5NextProcessor
    assert Glm5NextProcessor.__name__ == "Glm5NextProcessor"
mm = importlib.import_module("sglang.srt.rust_extensions._multimodal")
bits = mm.dsv41.resize_patchify(np.full((2, 2, 3), 255, dtype=np.uint8), (2, 2), (2, 2), (0, 0), 1)
assert bits.dtype == np.uint16 and bits.shape == (12,)
assert np.all(bits == 0x3f80)
try:
    mm.dsv41.resize_patchify(np.zeros((2, 2, 3), dtype=np.uint8), (2, 2), (2, 2), (0, 0), 0)
except ValueError:
    pass
else:
    raise AssertionError("Invalid native image geometry was accepted")
tree = importlib.import_module("sglang.srt.mem_cache.rust_tree_core.mem_cache")
core = tree.RustUnifiedTreeCoreBinding(tree.TreeCoreInitParamsBinding(device="cpu"), [0])
core.insert(tree.InsertParamsBinding(array("q", [1, 2, 3]), torch.tensor([10, 11, 12], dtype=torch.int64)))
matched = core.match_prefix(tree.MatchParamsBinding(array("q", [1, 2, 3])))
assert matched.device_indices.tolist() == [10, 11, 12]
import xgrammar as xg
info = xg.TokenizerInfo([bytes([i]) for i in range(256)] + [b"<eos>"], xg.VocabType.RAW, stop_token_ids=[256])
compiler = xg.GrammarCompiler(info, max_threads=2)
grammar = compiler.compile_json_schema(schema='{"type":"array","items":{"type":"integer"}}', any_whitespace=True, max_whitespace_cnt=2)
matcher = xg.GrammarMatcher(grammar)
assert matcher.accept_string("[1,  2]") and matcher.accept_token(256)
matcher = xg.GrammarMatcher(grammar)
assert not matcher.accept_string("[1,   2]")
for name in (
    "smg_grpc_proto.generated.sglang_scheduler_pb2",
    "smg_grpc_proto.generated.sglang_encoder_pb2",
    "grpc_health.v1.health_pb2",
    "grpc_reflection.v1alpha.reflection_pb2",
):
    importlib.import_module(name)
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.entrypoints.openai import protocol
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.function_call import mimo_detector
for reasoning_tokens in (0, 4):
    usage = UsageProcessor.calculate_token_usage(
        prompt_tokens=11, completion_tokens=9, reasoning_tokens=reasoning_tokens
    )
    assert isinstance(usage.completion_tokens_details, protocol.CompletionTokensDetails)
    responses = (
        protocol.ChatCompletionResponse(
            id="usage-probe", created=1, model="usage-probe", choices=[], usage=usage
        ),
        protocol.ChatCompletionStreamResponse(
            id="usage-probe", created=1, model="usage-probe", choices=[], usage=usage
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for response in responses:
            assert response.model_dump()["usage"]["completion_tokens_details"] == {
                "reasoning_tokens": reasoning_tokens
            }
marker = "SYNTHETIC_UNIFIED_IMAGE_PRIVACY_CANARY"
tool = Tool.model_validate({"type": "function", "function": {
    "name": marker, "parameters": {"type": "object", "properties": {marker: {"type": "integer"}}}
}})
captured = io.StringIO()
handler = logging.StreamHandler(captured)
logger = logging.getLogger()
logger.addHandler(handler)
try:
    mimo_detector._convert_param_value(marker, marker, marker, [tool])
finally:
    logger.removeHandler(handler)
    handler.close()
assert captured.getvalue().strip() and marker not in captured.getvalue()
print(json.dumps({
    "rust_modules": modules, "cuda_available": False, "used_sdk_driver_stub": False,
    "native_kv_registered_ops": 15, "rust_multimodal_cpu": True,
    "deepseek_model_import": True,
    "glm5_next_processor_import": "transformers" in EXPECTED["shared_dependency_overrides"],
    "governor_component": EXPECTED.get("governor") is not None,
    "rust_tree_torch_roundtrip": True, "xgrammar_bounded_whitespace": True,
    "generated_proto_imports": True, "parser_error_privacy": True,
    "typed_reasoning_usage_serialization": True,
    "installed_wrapper_sha256": EXPECTED["kernel"]["wrapper_sha256"],
    "runtime_assets_checked": sorted(EXPECTED["runtime_assets"]),
    "scope": "final-image-cpu-gates-not-gpu-or-production-acceptance",
}, indent=2))
"""


def verify(image, context_path, output):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Use an exact builder image ID, not a moving tag")
    if output.exists():
        raise ValueError("Evidence output must be fresh")
    expected = json.loads(context_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    rows = []
    commands = [
        ("hf-help", ["hf", "--help"], None),
        ("hf-download-help", ["hf", "download", "--help"], None),
        ("sglang-help", ["sglang", "serve", "--help"], None),
        (
            "installed-cpu",
            ["python", "-"],
            "EXPECTED = " + repr(expected) + "\n" + PROBE,
        ),
        ("dependencies", ["python", "-m", "pip", "check"], None),
    ]
    for name, command, program in commands:
        argv = [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--runtime",
            "runc",
            "--network",
            "none",
            "--cpus",
            "4",
            "--memory",
            "12g",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
            "-e",
            "PYTHONPATH=",
        ]
        if program is not None:
            argv.append("-i")
        argv += ["--entrypoint", command[0], image, *command[1:]]
        with (output / f"{name}.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(
                argv,
                input=program,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        rows.append({"name": name, "exit_code": result.returncode})
    passed = all(row["exit_code"] == 0 for row in rows if row["name"] != "dependencies")
    report = {
        "image_id": image,
        "engine": expected["engine"],
        "patches_commit": expected["patches_commit"],
        "cpu_gates_passed": passed,
        "checks": rows,
        "dependency_findings_require_review": rows[-1]["exit_code"] != 0,
        "source_or_executable_mounts": False,
        "runtime_installation": False,
        "used_sdk_driver_stub": False,
        "gpu_or_production_qualified": False,
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.image, args.context, args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["cpu_gates_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
