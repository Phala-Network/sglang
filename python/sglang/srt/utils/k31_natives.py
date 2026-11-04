"""Refuse to serve K3.1 when a compiled part it relies on fell back.

Without flashmla_ops or the rebuilt FlashInfer DynBlock modules K3.1 still answers
correctly, only slower, and every call site falls back without a word
(docker/k31.Dockerfile). A build can check this with docker/k31-image/verify_image.py;
this does the same at model load, for an image that skipped or predates it.
"""

import json
import re
import sysconfig
from pathlib import Path
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform

_FLASHMLA_OPS = ("sparse_decode_num_sm_parts", "flashmla_token_pack_prepare")
_K31_NUM_EXPERTS = 896
_DYNBLOCK_MODULES = ("fused_moe_trtllm_sm100", "trtllm_gen_routing")
_DYNBLOCK_HEADER = (
    "flashinfer/data/include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
)


def _flashmla_problem() -> Optional[str]:
    try:
        from sgl_kernel import flash_mla
    except Exception as e:
        return f"sgl_kernel.flash_mla does not import: {e}"
    if flash_mla._flashmla_import_error is not None:
        return f"flashmla_ops does not load: {flash_mla._flashmla_import_error}"
    missing = [op for op in _FLASHMLA_OPS if not hasattr(torch.ops.sgl_kernel, op)]
    if missing:
        return f"flashmla_ops lacks {missing}: the PyPI build is installed"
    return None


def _dynblock_problem(site: Path) -> Optional[str]:
    header = site / _DYNBLOCK_HEADER
    if not header.exists():
        return f"{header} is missing"
    match = re.search(r"DynBlockKernelMaxNumExperts = (\d+);", header.read_text())
    if match is None or int(match.group(1)) < _K31_NUM_EXPERTS:
        return (
            "FlashInfer DynBlock routing does not cover "
            f"{_K31_NUM_EXPERTS} experts ({match and match.group(1)})"
        )
    modules = [
        path
        for name in _DYNBLOCK_MODULES
        for path in site.glob(f"flashinfer*/**/{name}.so")
    ]
    if not modules:
        return "no FlashInfer TRT-LLM routing module is installed"
    for path in modules:
        marker = path.parent / "k31-dynblock.json"
        rebuilt = marker.exists() and (
            json.loads(marker.read_text()).get("DynBlockKernelMaxNumExperts", 0)
            >= _K31_NUM_EXPERTS
        )
        if not rebuilt:
            return f"{path} was not rebuilt for {_K31_NUM_EXPERTS} experts"
    return None


def missing_k31_natives(site: Optional[Path] = None) -> list[str]:
    site = site or Path(sysconfig.get_paths()["purelib"])
    problems = (_flashmla_problem(), _dynblock_problem(site))
    return [problem for problem in problems if problem is not None]


def require_k31_natives() -> None:
    """Raise when K3.1 would silently fall back on the GPUs it was built for."""
    if not torch.cuda.is_available() or not get_platform().is_sm100:
        return
    if envs.SGLANG_K31_ALLOW_NATIVE_FALLBACK.get():
        return
    problems = missing_k31_natives()
    if problems:
        raise RuntimeError(
            "Kimi-K3.1 would fall back to slower paths: "
            + "; ".join(problems)
            + ". Build the image with docker/k31.Dockerfile, or set "
            "SGLANG_K31_ALLOW_NATIVE_FALLBACK=1 to accept the fallback."
        )
