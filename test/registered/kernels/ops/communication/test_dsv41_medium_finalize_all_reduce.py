"""TP4 parity and graph replay for the DSV4.1 medium finalize plane."""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
from sglang.kernels.ops.communication import all_reduce_fusion
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN = 5120
TOP_K = 6
MB = 1024 * 1024


def _precompile(num_gpus):
    for world_size in num_gpus:
        all_reduce_fusion._jit_module(
            world_size,
            HIDDEN,
            TOP_K,
            all_reduce_fusion.default_cluster_size(HIDDEN),
        )


@cache_once
def _init_world() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    return coord.cpu_group


def _device() -> torch.device:
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")


@cache_once
def _init_comms() -> tuple[CustomAllReduceV2, CustomAllReduceV2]:
    cpu_group = _init_world()
    small = CustomAllReduceV2(
        cpu_group,
        _device(),
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=1 * MB,
        max_push_blocks=96,
    )
    medium = CustomAllReduceV2(
        cpu_group,
        _device(),
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=4 * MB,
        max_push_blocks=512,
    )
    assert not small.disabled and not medium.disabled
    small.override_algo = AllReduceAlgo.ONE_SHOT_PUSH
    medium.override_algo = AllReduceAlgo.ONE_SHOT_PUSH
    register_comm_cleanup(small)
    register_comm_cleanup(medium)
    all_reduce_fusion.register_comm(
        small.obj, comm_key=all_reduce_fusion.DEFAULT_COMM_KEY
    )
    all_reduce_fusion.register_comm(
        medium.obj, comm_key=all_reduce_fusion.DSV41_MEDIUM_COMM_KEY
    )
    return small, medium


def _make_inputs(num_tokens: int, seed: int):
    rank = dist.get_rank()
    generator = torch.Generator().manual_seed(seed * 7919 + rank)
    num_slots = num_tokens * TOP_K
    gemm2 = torch.randint(-2, 3, (num_slots + 8, HIDDEN), generator=generator).to(
        torch.bfloat16
    )
    weights = (
        torch.randint(0, 3, (num_tokens, TOP_K), generator=generator)
        .to(torch.bfloat16)
        .mul_(0.5)
    )
    shared = torch.randint(-2, 3, (num_tokens, HIDDEN), generator=generator).to(
        torch.bfloat16
    )
    indices = torch.arange(num_slots, dtype=torch.int32)
    indices[::11] = -1
    indices[:TOP_K] = -1
    device = _device()
    return (
        gemm2.to(device),
        indices.to(device),
        weights.to(device),
        shared.to(device),
    )


def _refresh_inputs(gemm2, indices, weights, shared, seed):
    rank = dist.get_rank()
    torch.manual_seed(seed * 7919 + rank)
    gemm2.copy_(torch.randint(-2, 3, gemm2.shape, device=_device()))
    weights.copy_(
        torch.randint(0, 3, weights.shape, device=_device()).to(torch.bfloat16)
    ).mul_(0.5)
    shared.copy_(torch.randint(-2, 3, shared.shape, device=_device()))
    indices.copy_(torch.arange(indices.numel(), dtype=torch.int32, device=_device()))
    indices[(seed + rank) % 13 :: 13] = -1
    indices[:TOP_K] = -1


@pytest.mark.parametrize("num_tokens", [96, 97, 288, 336, 384])
@torch.inference_mode()
def test_finalize_shared_rank_sum_graph_replay(num_tokens):
    small, medium = _init_comms()
    if num_tokens <= 96:
        comm = small
        comm_key = all_reduce_fusion.DEFAULT_COMM_KEY
    else:
        comm = medium
        comm_key = all_reduce_fusion.DSV41_MEDIUM_COMM_KEY

    gemm2, indices, weights, shared = _make_inputs(num_tokens, seed=17)

    def small_plane_reference():
        chunks = []
        for start in range(0, num_tokens, 96):
            end = min(start + 96, num_tokens)
            chunks.append(
                all_reduce_fusion.moe_finalize_all_reduce(
                    gemm2,
                    indices[start * TOP_K : end * TOP_K],
                    weights[start:end],
                    TOP_K,
                    shared[start:end],
                    world_size=4,
                    hidden_dim=HIDDEN,
                    comm_key=all_reduce_fusion.DEFAULT_COMM_KEY,
                    prefetch_metadata=False,
                )
            )
        return torch.cat(chunks)

    def chain():
        expected = small_plane_reference()
        # The reference collectives have completed on every rank. Delaying
        # rank 0 here specifically exercises the medium plane's peer wait.
        if dist.get_rank() == 0:
            torch.cuda._sleep(100000)
        actual = all_reduce_fusion.moe_finalize_all_reduce(
            gemm2,
            indices,
            weights,
            TOP_K,
            shared,
            world_size=4,
            hidden_dim=HIDDEN,
            comm_key=comm_key,
            prefetch_metadata=False,
        )
        return expected, actual

    chain()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with small.capture(), medium.capture(), torch.cuda.graph(graph):
        expected, actual = chain()

    for replay in range(3):
        _refresh_inputs(gemm2, indices, weights, shared, seed=31 + replay)
        graph.replay()
        torch.cuda.synchronize()
        error = None
        try:
            assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
            torch.testing.assert_close(
                actual.view(torch.int16),
                expected.view(torch.int16),
                rtol=0,
                atol=0,
            )
        except AssertionError as exc:
            error = f"rank={dist.get_rank()}, M={num_tokens}, replay={replay}: {exc}"
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        assert not any(errors), "\n".join(item for item in errors if item)


@torch.inference_mode()
def test_routed_rounds_before_shared_add():
    """Catch helpers that incorrectly round only after adding shared output."""
    _, medium = _init_comms()
    num_tokens = 97
    gemm2 = torch.zeros(
        num_tokens * TOP_K,
        HIDDEN,
        dtype=torch.bfloat16,
        device=_device(),
    )
    weights = torch.zeros(num_tokens, TOP_K, dtype=torch.bfloat16, device=_device())
    indices = torch.arange(num_tokens * TOP_K, dtype=torch.int32, device=_device())
    shared = torch.zeros(num_tokens, HIDDEN, dtype=torch.bfloat16, device=_device())
    # FP32 routed accumulation is 1 + 1/256. BF16 round-to-even produces 1,
    # then adding -1 must produce zero. Adding shared before the routed BF16
    # rounding would incorrectly retain 1/256.
    gemm2[0, 0] = 1
    gemm2[1, 0] = 1
    weights[0, 0] = 1
    weights[0, 1] = 1 / 256
    shared[0, 0] = -1
    indices[TOP_K:] = -1

    out = all_reduce_fusion.moe_finalize_all_reduce(
        gemm2,
        indices,
        weights,
        TOP_K,
        shared,
        world_size=4,
        hidden_dim=HIDDEN,
        comm_key=all_reduce_fusion.DSV41_MEDIUM_COMM_KEY,
        prefetch_metadata=False,
    )
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) == 0


@torch.inference_mode()
def test_default_medium_and_generic_push_interleave():
    small, medium = _init_comms()
    sequence = [
        (96, all_reduce_fusion.DEFAULT_COMM_KEY),
        (97, all_reduce_fusion.DSV41_MEDIUM_COMM_KEY),
        (384, all_reduce_fusion.DSV41_MEDIUM_COMM_KEY),
        (96, all_reduce_fusion.DEFAULT_COMM_KEY),
        (288, all_reduce_fusion.DSV41_MEDIUM_COMM_KEY),
        (336, all_reduce_fusion.DSV41_MEDIUM_COMM_KEY),
    ]

    for step, (num_tokens, comm_key) in enumerate(sequence):
        gemm2, indices, weights, shared = _make_inputs(num_tokens, seed=101 + step)
        chunks = []
        for start in range(0, num_tokens, 96):
            end = min(start + 96, num_tokens)
            chunks.append(
                all_reduce_fusion.moe_finalize_all_reduce(
                    gemm2,
                    indices[start * TOP_K : end * TOP_K],
                    weights[start:end],
                    TOP_K,
                    shared[start:end],
                    world_size=4,
                    hidden_dim=HIDDEN,
                    comm_key=all_reduce_fusion.DEFAULT_COMM_KEY,
                    prefetch_metadata=False,
                )
            )
        expected = torch.cat(chunks)

        probe = torch.full(
            (1024,),
            dist.get_rank() + 1,
            dtype=torch.bfloat16,
            device=_device(),
        )
        ordinary = small.custom_all_reduce(probe)
        torch.testing.assert_close(
            ordinary,
            torch.full_like(ordinary, 10),
            rtol=0,
            atol=0,
        )

        if dist.get_rank() == 0:
            torch.cuda._sleep(100000)
        actual = all_reduce_fusion.moe_finalize_all_reduce(
            gemm2,
            indices,
            weights,
            TOP_K,
            shared,
            world_size=4,
            hidden_dim=HIDDEN,
            comm_key=comm_key,
            prefetch_metadata=False,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            actual.view(torch.int16),
            expected.view(torch.int16),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    multigpu_pytest_main(
        __name__,
        __file__,
        num_gpus=(4,),
        pre_launch_fn=_precompile,
    )
