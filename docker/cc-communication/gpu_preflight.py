"""Eight-GPU CC IPC and D2H gates, run before model load on the drained CVM.

This is not a model throughput benchmark. It intentionally refuses non-B200,
non-CC hardware, and never changes the device's security configuration.
"""

import ctypes
import gc
import json
import os
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist


def emit(event, **data):
    print(json.dumps({"gate": "phala_cc_gpu", "rank": dist.get_rank(),
                      "event": event, **data}), flush=True)


def preflight_oneshot_modes(token_count, world_size):
    """Only enumerate modes supported by TRTLLM's token-count contract."""
    # FlashInfer explicitly rejects two-shot when token_count <= world_size.
    # Small decode shapes still receive full one-shot numeric/graph coverage.
    return (True, False) if token_count > world_size else (True,)


def valid_host_allocation(is_cpu, pytorch_pinned, cc_enabled, host_status, memory_type):
    # Under CC, cudaMallocHost may be reported as cudaMemoryTypeManaged (3).
    # Require successful cudaHostGetFlags, not just a managed-memory pointer.
    return is_cpu and (pytorch_pinned or (
        cc_enabled and host_status == 0 and memory_type in (1, 3)))


def host_allocation_metadata(tensor):
    class Attributes(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("device", ctypes.c_int),
                    ("devicePointer", ctypes.c_void_p), ("hostPointer", ctypes.c_void_p),
                    ("reserved", ctypes.c_long * 8)]  # CUDA 13 driver_types.h

    runtime = ctypes.CDLL("libcudart.so.13")
    runtime.cudaPointerGetAttributes.argtypes = [ctypes.POINTER(Attributes), ctypes.c_void_p]
    runtime.cudaPointerGetAttributes.restype = ctypes.c_int
    runtime.cudaHostGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
    runtime.cudaHostGetFlags.restype = ctypes.c_int
    attributes, flags = Attributes(), ctypes.c_uint()
    pointer_status = runtime.cudaPointerGetAttributes(ctypes.byref(attributes), tensor.data_ptr())
    host_status = runtime.cudaHostGetFlags(ctypes.byref(flags), tensor.data_ptr())
    assert pointer_status == 0, pointer_status
    return {"pytorch_pinned": tensor.is_pinned(), "host_status": host_status,
            "memory_type": attributes.type, "host_flags": flags.value}


def main():
    for key in ("SGLANG_CONFIDENTIAL_COMPUTE", "FLASHINFER_CONFIDENTIAL_COMPUTE"):
        if key in os.environ:
            raise RuntimeError("GPU preflight requires real NVML detection, no software override")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    assert torch.cuda.device_count() == 8
    assert "B200" in torch.cuda.get_device_name(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    assert dist.get_world_size() == 8
    from flashinfer import comm
    from flashinfer.comm import torch_symmetric_memory as symm
    from flashinfer.comm import trtllm_ar
    from flashinfer.utils import is_confidential_compute as fi_cc
    from sglang.srt.layers import flashinfer_comm_fusion as fusion
    from sglang.srt.managers.async_d2h_copy_worker import AsyncD2HCopyWorker
    from sglang.srt.managers.utils import _async_d2h
    from sglang.srt.runtime_context import override_platform
    from sglang.srt.utils.confidential_compute import is_confidential_compute

    modes = [None] * 8
    dist.all_gather_object(modes, (is_confidential_compute(), fi_cc()), group=cpu_group)
    assert modes == [(True, True)] * 8, modes
    model = json.loads(Path(os.environ["PHALA_CC_MODEL_CONFIG"]).read_text())
    hidden = int(model.get("text_config", model)["hidden_size"])
    assert hidden > 0 and hidden % 128 == 0
    emit("hardware", gpu=torch.cuda.get_device_name(rank), cc_modes=modes,
         hidden_size=hidden, capability=torch.cuda.get_device_capability(rank),
         torch=torch.__version__, memory=torch.cuda.mem_get_info())
    backend = fusion._TorchDistBackend(device_group=dist.group.WORLD, cpu_group=cpu_group)

    # Inject only an allocation exception, not a real OOM. Other ranks use
    # real GPU allocations; every rank must raise before any rendezvous call.
    for failed_rank in (0, 7):
        allocate = symm.symm_mem.empty

        def injected(*args, **kwargs):
            if rank == failed_rank:
                raise RuntimeError("injected GPU preflight allocation failure")
            return allocate(*args, **kwargs)

        with patch.object(symm.symm_mem, "empty", side_effect=injected), \
                patch.object(symm.symm_mem, "rendezvous") as rendezvous:
            try:
                symm._alloc_symm_buffer_bytes(8192, 8, torch.float32,
                    torch.device("cuda", rank), dist.group.WORLD.group_name,
                    comm_backend=backend)
            except RuntimeError as error:
                assert "IPC allocation failed across ranks" in str(error)
            else:
                raise AssertionError("allocation failure was not propagated")
            rendezvous.assert_not_called()
        dist.barrier(group=cpu_group)
        emit("allocation_failure_propagated", injected_rank=failed_rank)

    torch.manual_seed(20260909 + rank)
    reference_count = len(trtllm_ar._symm_workspace_refs)
    for cycle in range(3):
        manager = fusion.FlashInferWorkspaceManager()
        major = torch.cuda.get_device_capability(rank)[0]
        with override_platform(is_sm100=major == 10, is_sm90=major == 9):
            assert fusion._resolve_backend("auto") == "trtllm"
            manager.initialize(world_size=8, rank=rank, max_token_num=384,
                hidden_dim=hidden, backend="trtllm", dtype=torch.bfloat16,
                use_oneshot=True, device_group=dist.group.WORLD, cpu_group=cpu_group)
        assert manager.initialized and manager.backend == "trtllm"
        emit("workspace", cycle=cycle, backend=manager.backend,
             live_ipc_reference_sets=len(trtllm_ar._symm_workspace_refs))
        for tokens in ((1, 8, 32, 48, 288, 384) if cycle == 0 else (48,)):
            src = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
            residual = torch.full_like(src, 0.1)
            weight = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
            # Accumulate the BF16 inputs in FP32 for an independent reference.
            # NCCL's BF16 reduction can round intermediate partial sums; adding
            # the residual afterwards can amplify its error under cancellation.
            # Keep the old BF16 result as diagnostic evidence, not as an oracle.
            reduced_bf16 = src.clone()
            dist.all_reduce(reduced_bf16)
            reduced = src.float()
            dist.all_reduce(reduced)
            residual_ref = reduced + residual.float()
            norm_ref = residual_ref * torch.rsqrt(residual_ref.square().mean(-1, keepdim=True) + 1e-6)
            output, norm, residual_out = (torch.empty_like(src) for _ in range(3))
            modes = preflight_oneshot_modes(tokens, dist.get_world_size())
            if False not in modes:
                emit("twoshot_not_applicable", tokens=tokens,
                     requirement="token_count > world_size")
            for oneshot in modes:
                def operations():
                    comm.allreduce_fusion(src, manager.workspace,
                        comm.AllReduceFusionPattern.kAllReduce, output=output,
                        use_oneshot=oneshot, launch_with_pdl=True)
                    comm.allreduce_fusion(src, manager.workspace,
                        comm.AllReduceFusionPattern.kARResidualRMSNorm,
                        residual_out=residual_out, norm_out=norm,
                        residual_in=residual, rms_gamma=weight, rms_eps=1e-6,
                        use_oneshot=oneshot, launch_with_pdl=True,
                        trigger_completion_at_end=False)

                # Warmup on a non-default stream before graph capture.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        operations()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                old_residual_ref = reduced_bf16.float() + residual.float()
                old_bad = (residual_out.float() - old_residual_ref).abs() > (
                    0.01 + 0.03 * old_residual_ref.abs())
                emit("reference_precision", cycle=cycle, tokens=tokens, oneshot=oneshot,
                     fp32_reference=True,
                     nccl_bf16_max_abs_error=(reduced_bf16.float()-reduced).abs().max().item(),
                     fused_reduce_max_abs_error=(output.float()-reduced).abs().max().item(),
                     fused_residual_max_abs_error=(residual_out.float()-residual_ref).abs().max().item(),
                     old_reference_failed_elements=old_bad.sum().item())
                if old_bad.any():
                    coordinates = old_bad.nonzero()[:4]
                    emit("old_reference_mismatch", cycle=cycle, tokens=tokens, oneshot=oneshot,
                         coordinates=coordinates.tolist(),
                         old_reference=old_residual_ref[old_bad][:4].tolist(),
                         fp32_reference=residual_ref[old_bad][:4].tolist(),
                         fused_residual=residual_out[old_bad][:4].tolist())
                torch.testing.assert_close(output.float(), reduced, rtol=0.03, atol=0.01)
                torch.testing.assert_close(residual_out.float(), residual_ref, rtol=0.03, atol=0.01)
                torch.testing.assert_close(norm.float(), norm_ref, rtol=0.03, atol=0.03)
                graph = torch.cuda.CUDAGraph()
                dist.barrier(group=cpu_group)
                with torch.cuda.graph(graph, stream=stream):
                    operations()
                for _ in range(16):
                    graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(output.float(), reduced, rtol=0.03, atol=0.01)
                torch.testing.assert_close(residual_out.float(), residual_ref, rtol=0.03, atol=0.01)
                torch.testing.assert_close(norm.float(), norm_ref, rtol=0.03, atol=0.03)
                emit("numeric_graph", cycle=cycle, tokens=tokens, oneshot=oneshot,
                     graph_replays=16, max_norm_abs_error=(norm.float()-norm_ref).abs().max().item())
                del graph
            del src, residual, weight, reduced, reduced_bf16, residual_ref, norm_ref, output, norm, residual_out
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        manager.cleanup()
        del manager
        gc.collect()
        torch.cuda.empty_cache()
        assert len(trtllm_ar._symm_workspace_refs) == reference_count
        emit("workspace_released", cycle=cycle, memory=torch.cuda.mem_get_info())

    worker = AsyncD2HCopyWorker(torch.cuda)
    for elements in (8, 48, 288, 151936):
        for step in range(16):
            src = torch.randn(elements, device="cuda")
            expected = src.cpu()
            result = {}
            done = worker.submit(lambda: result.update(cpu=_async_d2h(src)))
            done.synchronize()
            host = host_allocation_metadata(result["cpu"])
            assert valid_host_allocation(result["cpu"].device.type == "cpu",
                host["pytorch_pinned"], is_confidential_compute(),
                host["host_status"], host["memory_type"]), host
            torch.testing.assert_close(result["cpu"], expected, rtol=0, atol=0)
        emit("d2h_numeric", elements=elements, iterations=16, host_allocation=host)
    assert worker.shutdown(timeout=5)
    del src, expected, result, done, worker
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    emit("PASS")
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
