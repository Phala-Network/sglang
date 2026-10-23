"""Small final-image GPU qualification, no model data or serving traffic."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import publish, get_parallel
from sglang.srt.server_args import ServerArgs

envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.set(False)  # Actual Xing model hook.
publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="scheduler")
get_parallel().override_permanently(tp_group=SimpleNamespace(world_size=1),
    tp_rank=0, attn_tp_rank=0, moe_tp_rank=0, moe_ep_rank=0)

from sglang.kernels.ops.layernorm.mhc import mhc_pre_big_fuse_tilelang
from sglang.srt.models.xing4_0 import mhc_pre, mhc_post


def sinkhorn(mixes, scale, base):
    logits = (mixes[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4).clamp(-30, 30)
    comb = torch.exp(logits - logits.amax(-1, keepdim=True))
    for _ in range(20):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    return comb


def maxabs(a, b):
    return float((a.float() - b.float()).abs().max())


def main():
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261023)
    device = "cuda"
    hidden = 3584
    scale = torch.tensor([0.123456789, 0.618033989, 1.41421356], device=device)
    base = torch.linspace(-1, 1, 24, device=device)
    checks = []
    # Isolate the changed mixing code from existing GEMM precision.
    mixes = (torch.arange(3 * 24, device=device).view(3, 24).float().sin()
        * torch.tensor([1.0, 20.0, 100.0], device=device)[:, None])
    sqr = torch.full((1, 3), 4 * hidden, device=device, dtype=torch.float32)
    residual = torch.randn(3, 4, hidden, device=device, dtype=torch.bfloat16)
    post = torch.empty(3, 4, device=device)
    comb = torch.empty(3, 16, device=device)
    layer = torch.empty(3, hidden, device=device, dtype=torch.bfloat16)
    mhc_pre_big_fuse_tilelang(mixes.unsqueeze(0), sqr, scale, base, residual,
        post, comb, layer, hidden, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 4, 24,
        xing_sinkhorn=True)
    normalized = mixes * torch.rsqrt(sqr[0, :, None] / (4 * hidden) + 1e-6)
    expected = sinkhorn(normalized, scale, base)
    direct_error = maxabs(comb.view(3, 4, 4).transpose(-1, -2), expected)
    assert direct_error <= 1e-5, direct_error
    checks.append(dict(case="direct-normal-and-clamped", comb_maxabs=direct_error,
        tolerance=1e-5, finite=bool(torch.isfinite(comb).all())))
    assert checks[-1]["finite"]

    fn = (torch.randn(24, 4 * hidden, device=device) * 0.002).float()
    for tokens in [1, 32, 2048, 2049]:
        started = time.monotonic()
        residual = torch.randn(tokens, 4, hidden, device=device, dtype=torch.bfloat16)
        result = mhc_pre(residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 8)
        torch.cuda.synchronize()
        post, comb, layer = result
        flat = residual.float().view(tokens, -1)
        # Exact fused precision contract: FP32 operands and reduction, BF16 output.
        stats = torch.nn.functional.linear(flat, fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        expected_comb = sinkhorn(stats, scale, base)
        expected_post = 2 * torch.sigmoid(stats[:, 4:8] * scale[1] + base[4:8])
        pre = torch.sigmoid(stats[:, :4] * scale[0] + base[:4]) + 1e-6
        expected_layer = (pre[..., None] * residual.float()).sum(1).bfloat16()
        errors = dict(comb=maxabs(comb.transpose(-1, -2), expected_comb),
            post=maxabs(post.squeeze(-1), expected_post), layer=maxabs(layer, expected_layer))
        assert errors["comb"] <= 1e-4, (tokens, errors)
        assert errors["post"] <= 1e-4, (tokens, errors)
        assert errors["layer"] <= 0.03125, (tokens, errors)
        assert comb.is_contiguous() and comb.shape == (tokens, 4, 4)
        assert post.shape == (tokens, 4, 1) and layer.dtype == torch.bfloat16
        x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
        actual_post = mhc_post(x, residual, post, comb)
        # mhc_post accumulates FP32 and rounds once to BF16.
        expected_output = (post.float() * x.float()[:, None, :]
            + torch.matmul(comb.transpose(-1, -2), residual.float())).bfloat16()
        post_error = maxabs(actual_post, expected_output)
        assert post_error <= 0.0625, (tokens, post_error)
        graph_equal = None
        if tokens in (1, 2049):
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    mhc_pre(residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 8)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = mhc_pre(residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 8)
            graph.replay()
            torch.cuda.synchronize()
            graph_equal = all(torch.equal(a, b) for a, b in zip(captured, result))
            assert graph_equal
        checks.append(dict(case="splitk" if tokens <= 2048 else "large-prefill",
            tokens=tokens, errors=errors, mhc_post_maxabs=post_error,
            graph_equal=graph_equal, seconds=time.monotonic()-started))
        print("XING4_MHC_GPU_PROGRESS=" + json.dumps(checks[-1]), flush=True)
    source = Path('/opt/phala/candidate-source.json')
    manifest = json.loads(source.read_text())
    print("XING4_MHC_GPU_RECEIPT=" + json.dumps(dict(engine_commit=manifest['engine_commit'],
        checks=checks, pass_all=True, scope="mHC kernel/layout/graphs; no model-quality or throughput claim")), flush=True)


if __name__ == "__main__":
    main()
