"""Correctness test for the UNICAST build of the K3 fused all-reduce kernels.

The unicast build (``register_comm(..., multicast=False)``) serves a
CustomAllReduceV2 push plane whose symmetric memory has no multicast mapping
(GPU confidential computing). It never touches the multicast VA, so this test
runs on any 8-GPU NVLink box, with or without multicast, and under CC.

Checks, each bit-exact against NCCL on small-int bf16 inputs unless noted:

* ``all_reduce_push_res`` with / without the fused residual, full grid and
  the low-SM grids (1, 2, 8 work blocks), sizes up to the push window;
* ``all_reduce_push_norm`` and ``finalize_all_reduce_push_norm`` (fp32
  RMSNorm epilogue: bf16 tolerances on the normed rows);
* the K3 gemm_ag GEMV producer + spin consumer and the gather path, and the
  moe_front fp32 all-gather, in their unicast builds;
* a mixed stress loop (every kernel sharing the push plane's phase counters,
  with the low-SM bumper grids interleaved) and a CUDA-graph capture/replay.

Usage::

    python test/registered/kernels/ops/communication/test_ar_fusion_unicast.py   # relaunches under torchrun (8 GPUs)

To A/B the two builds on a multicast box, run test_ar_fusion.py next to it.
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once, get_ci_test_range
from sglang.kernels.ops.communication import all_reduce_residual as all_reduce
from sglang.kernels.ops.communication import gemm_ag, moe_front_ag
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(
    est_time=240,
    stage="extra-b",
    runner_config="8-gpu-h200",
)

H = 7168  # Kimi-K3 hidden size
NORM_DIM = 3584  # latent width
MB = 1024 * 1024

PUSH_BS = [1, 2, 8, 32, 128]
PUSH_BS = get_ci_test_range(PUSH_BS, [1, 32, 128])
# work blocks of the unicast push: None = the whole plane
GRIDS = [None, 1, 2, 8]


def _precompile(num_gpus):
    for ws in num_gpus:
        all_reduce._jit_module(ws, False)
        gemm_ag._jit_module(ws, False)


@cache_once
def _init_world():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    return coord.cpu_group


@cache_once
def _init_nccl_group():
    _init_world()
    local_rank = int(os.environ["LOCAL_RANK"])
    group = dist.new_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    assert isinstance(group, dist.ProcessGroup)
    return group


@cache_once
def _init_comm() -> CustomAllReduceV2:
    cpu_group = _init_world()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    # push-only: the unicast build has no pull family to barrier for
    comm = CustomAllReduceV2(cpu_group, device, max_pull_size=0, max_push_size=2 * MB)
    if comm.disabled:
        raise RuntimeError("the unicast ar_fusion test needs CustomAllReduceV2")
    all_reduce.register_comm(comm.obj, multicast=False)
    register_comm_cleanup(comm)
    return comm


def _device() -> torch.device:
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")


def _int_input(n: int, seed: int, per_rank: bool) -> torch.Tensor:
    # small ints are exact in bf16 even after an fp32-accumulated 8-way sum,
    # so the comparison against NCCL is bit-exact
    rank = dist.get_rank() if per_rank else 0
    g = torch.Generator().manual_seed(seed * 1009 + rank)
    return torch.randint(0, 16, (n,), dtype=torch.bfloat16, generator=g).to(_device())


def _nccl_ref(x: torch.Tensor, residual):
    ref = x.clone()
    dist.all_reduce(ref, group=_init_nccl_group())
    return ref if residual is None else ref + residual


def _norm_ref(x_reduced: torch.Tensor, num_norm_rows: int, weight, eps: float):
    out = x_reduced.clone()
    normed_part = out[: num_norm_rows * NORM_DIM].view(num_norm_rows, NORM_DIM).float()
    factor = torch.rsqrt(normed_part.pow(2).mean(-1, keepdim=True) + eps)
    normed = (normed_part * factor * weight.float()).to(torch.bfloat16)
    out[: num_norm_rows * NORM_DIM] = normed.view(-1)
    return out


def _assert_norm_close(x: torch.Tensor, ref: torch.Tensor, num_norm_rows: int):
    torch.testing.assert_close(
        x[num_norm_rows * NORM_DIM :], ref[num_norm_rows * NORM_DIM :], atol=0, rtol=0
    )
    torch.testing.assert_close(
        x[: num_norm_rows * NORM_DIM], ref[: num_norm_rows * NORM_DIM]
    )


def test_registered_as_unicast():
    comm = _init_comm()
    assert all_reduce.is_multicast(comm.world_size) is False


@pytest.mark.parametrize("bs", PUSH_BS)
@pytest.mark.parametrize("use_residual", [False, True])
@pytest.mark.parametrize("num_blocks", GRIDS)
@torch.inference_mode()
def test_unicast_push(bs: int, use_residual: bool, num_blocks):
    comm = _init_comm()
    world = comm.world_size
    n = bs * H
    x = _int_input(n, bs, per_rank=True)
    residual = _int_input(n, bs + 7, per_rank=False) if use_residual else None
    ref = _nccl_ref(x, residual)
    all_reduce.all_reduce_push_res(world, x, residual, num_blocks=num_blocks)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, ref, atol=0, rtol=0)


@torch.inference_mode()
def test_unicast_push_grid_may_differ_across_ranks():
    """The grid only shapes this rank's counter flips, so ranks may pick
    different work-block counts for the same message."""
    comm = _init_comm()
    world = comm.world_size
    n = 8 * H
    x = _int_input(n, 321, per_rank=True)
    ref = _nccl_ref(x, None)
    all_reduce.all_reduce_push_res(world, x, None, num_blocks=1 + dist.get_rank() % 4)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, ref, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [1, 8, 24])
@pytest.mark.parametrize("rows_per_token", [3, 1])
@torch.inference_mode()
def test_unicast_push_norm(num_tokens: int, rows_per_token: int):
    comm = _init_comm()
    world = comm.world_size
    n = num_tokens * rows_per_token * NORM_DIM
    x = _int_input(n, num_tokens + 41 + rows_per_token, per_rank=True)
    weight = _int_input(NORM_DIM, 43, per_rank=False) + 1
    ref = _norm_ref(_nccl_ref(x, None), num_tokens, weight, eps=1e-6)
    all_reduce.all_reduce_push_norm(world, x, weight, 1e-6, num_norm_rows=num_tokens)
    torch.cuda.synchronize()
    _assert_norm_close(x, ref, num_tokens)


FIN_TOPK = 16


def _build_permuted_layout(
    num_tokens: int, seed: int, w_dtype: torch.dtype = torch.float32
):
    num_experts, tile = 896, 8
    gen = torch.Generator(device="cpu").manual_seed(seed)
    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, generator=gen)[:FIN_TOPK]
            for _ in range(num_tokens)
        ]
    )
    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts)
    padded = (counts + tile - 1) // tile * tile
    bases = torch.cumsum(padded, 0) - padded
    fill = torch.zeros(num_experts, dtype=torch.long)
    idx = torch.empty(num_tokens * FIN_TOPK, dtype=torch.int32)
    for i, e in enumerate(topk_ids.flatten().tolist()):
        idx[i] = bases[e] + fill[e]
        fill[e] += 1
    weights = torch.rand(num_tokens, FIN_TOPK, generator=gen).to(w_dtype)
    num_rows = int(padded.sum())
    g = torch.Generator(device="cpu").manual_seed(seed * 31 + dist.get_rank())
    gemm2 = (torch.randn(num_rows, NORM_DIM, generator=g) * 2).to(torch.bfloat16)
    dev = _device()
    return gemm2.to(dev), idx.to(dev), weights.to(dev)


def _finalize_norm_ref(gemm2, idx, weights, norm_w, eps: float) -> torch.Tensor:
    num_tokens = weights.shape[0]
    idx2 = idx.view(num_tokens, FIN_TOPK).long()
    acc = torch.zeros(num_tokens, NORM_DIM, dtype=torch.float32, device=gemm2.device)
    for k in range(FIN_TOPK):
        acc += weights[:, k, None].float() * gemm2[idx2[:, k]].float()
    local = acc.to(torch.bfloat16)
    world = dist.get_world_size()
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local, group=_init_nccl_group())
    total = torch.zeros_like(acc)
    for r in range(world):
        total += gathered[r].float()
    factor = torch.rsqrt(total.square().mean(dim=-1, keepdim=True) + eps)
    return (total * factor * norm_w.float()).to(torch.bfloat16)


@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bs", PUSH_BS)
@torch.inference_mode()
def test_unicast_finalize_push_norm(bs: int, w_dtype: torch.dtype):
    comm = _init_comm()
    world = comm.world_size
    eps = 1e-6
    gemm2, idx, weights = _build_permuted_layout(bs, seed=bs + 23, w_dtype=w_dtype)
    g = torch.Generator(device="cpu").manual_seed(77)
    norm_w = (torch.rand(NORM_DIM, generator=g) + 0.5).to(torch.bfloat16).to(_device())
    ref = _finalize_norm_ref(gemm2, idx, weights, norm_w, eps)
    out = torch.empty(bs, NORM_DIM, dtype=torch.bfloat16, device=_device())
    all_reduce.finalize_all_reduce_push_norm(
        world, out, gemm2, idx, weights, norm_w, eps
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# --- gemm_ag / moe_front_ag unicast builds -----------------------------------


def _gemm_ag_ref(x, weight, b, c):
    out = (x @ weight.t()).to(torch.bfloat16) + b
    return out if c is None else out + c


@pytest.mark.parametrize("num_tokens", get_ci_test_range([1, 3, 12], [1, 12]))
@pytest.mark.parametrize("use_c", [False, True])
@torch.inference_mode()
def test_unicast_gemm_ag_gemv(num_tokens: int, use_c: bool):
    comm = _init_comm()
    world = comm.world_size
    if world not in gemm_ag.WORLD_SIZES:
        pytest.skip("gemm_ag is built for TP2/4/8")
    g = torch.Generator(device="cpu").manual_seed(900 + num_tokens)
    # a sparse 0/1 x against a {-1, 0, 1} weight keeps every dot product a
    # small integer (|.| well under 256), exact in bf16 for the kernel's fp32
    # accumulate and for the cuBLAS reference alike: bit-exact comparison
    x = (torch.rand(num_tokens, gemm_ag.K, generator=g) < 1 / 64).to(torch.bfloat16)
    weight = torch.randint(-1, 2, (gemm_ag.N, gemm_ag.K), generator=g).to(
        torch.bfloat16
    )
    b = torch.randint(0, 8, (num_tokens, gemm_ag.N), generator=g).to(torch.bfloat16)
    c = torch.randint(0, 8, (num_tokens, gemm_ag.N), generator=g).to(torch.bfloat16)
    x, weight, b, c = (t.to(_device()) for t in (x, weight, b, c))
    c_ = c if use_c else None
    out = torch.empty_like(b)
    gemm_ag.gemm_ag_up_proj(world, x, weight, b, c_, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, _gemm_ag_ref(x, weight, b, c_), atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", get_ci_test_range([13, 64, 256], [64]))
@torch.inference_mode()
def test_unicast_gemm_ag_gather(num_tokens: int):
    comm = _init_comm()
    world = comm.world_size
    if world not in gemm_ag.WORLD_SIZES:
        pytest.skip("gemm_ag is built for TP2/4/8")
    rank = dist.get_rank()
    g = torch.Generator(device="cpu").manual_seed(1000 + num_tokens)
    full = torch.randint(0, 16, (num_tokens, gemm_ag.N), generator=g).to(torch.bfloat16)
    b = torch.randint(0, 8, (num_tokens, gemm_ag.N), generator=g).to(torch.bfloat16)
    full, b = full.to(_device()), b.to(_device())
    cols = gemm_ag.N // world
    local = full[:, rank * cols : (rank + 1) * cols].contiguous()
    out = torch.empty_like(b)
    gemm_ag.gather_add3(world, local, b, None, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, full + b, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", get_ci_test_range([18, 64, 256], [18, 256]))
@torch.inference_mode()
def test_unicast_moe_front_all_gather(num_tokens: int):
    comm = _init_comm()
    world = comm.world_size
    rank = dist.get_rank()
    n_local = 560  # (896 experts + 3584 latent) / 8, a multiple of 4
    g = torch.Generator(device="cpu").manual_seed(2000 + num_tokens)
    full = torch.randn(num_tokens, n_local * world, generator=g).to(_device())
    # a strided column slice of a wider fp32 GEMM output, as the model hands it
    local = full[:, rank * n_local : (rank + 1) * n_local]
    out = torch.empty(
        num_tokens, n_local * world, dtype=torch.float32, device=_device()
    )
    moe_front_ag.moe_front_all_gather(world, local, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, full, atol=0, rtol=0)


@torch.inference_mode()
def test_unicast_stress_mixed():
    """Every unicast kernel sharing the push plane, back to back with
    low-SM bumper grids interleaved: the phase counters must stay uniform."""
    comm = _init_comm()
    world = comm.world_size
    rank = dist.get_rank()
    g = torch.Generator(device="cpu").manual_seed(78)
    norm_w = (torch.rand(NORM_DIM, generator=g) + 0.5).to(torch.bfloat16).to(_device())
    for it in range(24):
        bs = (1, 8, 32)[it % 3]
        x = _int_input(bs * H, 3000 + it, per_rank=True)
        ref = _nccl_ref(x, None)
        all_reduce.all_reduce_push_res(world, x, None, num_blocks=(None, 1, 8)[it % 3])
        torch.testing.assert_close(x, ref, atol=0, rtol=0)
        gemm2, idx, weights = _build_permuted_layout(bs, seed=9000 + it)
        ref2 = _finalize_norm_ref(gemm2, idx, weights, norm_w, 1e-5)
        out = torch.empty(bs, NORM_DIM, dtype=torch.bfloat16, device=_device())
        all_reduce.finalize_all_reduce_push_norm(
            world, out, gemm2, idx, weights, norm_w, 1e-5
        )
        torch.testing.assert_close(out, ref2, atol=1e-2, rtol=1e-2)
        if world in gemm_ag.WORLD_SIZES:
            full = _int_input(bs * gemm_ag.N, 4000 + it, per_rank=False).view(bs, -1)
            b = _int_input(bs * gemm_ag.N, 5000 + it, per_rank=False).view(bs, -1)
            cols = gemm_ag.N // world
            local = full[:, rank * cols : (rank + 1) * cols].contiguous()
            out3 = torch.empty_like(b)
            gemm_ag.gather_add3(world, local, b, None, out3)
            torch.testing.assert_close(out3, full + b, atol=0, rtol=0)
    torch.cuda.synchronize()


@torch.inference_mode()
def test_unicast_graph_capture():
    comm = _init_comm()
    world = comm.world_size
    cpu_group = _init_world()
    n = 8 * H
    gres = _int_input(n, 99, per_rank=False)
    gx = torch.zeros(n, dtype=torch.bfloat16, device=_device())
    gy = torch.zeros(n, dtype=torch.bfloat16, device=_device())

    def _run_all():
        all_reduce.all_reduce_push_res(world, gx, gres)
        all_reduce.all_reduce_push_res(world, gy, gres, num_blocks=2)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run_all()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_all()

    for it in range(4):
        vx = _int_input(n, 5000 + it, per_rank=True)
        vy = _int_input(n, 6000 + it, per_rank=True)
        ref_x = _nccl_ref(vx, gres)
        ref_y = _nccl_ref(vy, gres)
        gx.copy_(vx)
        gy.copy_(vy)
        dist.barrier(group=cpu_group)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(gx, ref_x, atol=0, rtol=0)
        torch.testing.assert_close(gy, ref_y, atol=0, rtol=0)


if __name__ == "__main__":
    multigpu_pytest_main(
        __name__,
        __file__,
        num_gpus=(8,),
        pre_launch_fn=_precompile,
    )
