"""Execute the real TileLang mixing block on CPU; GPU lowering is separate."""
import ast
import copy
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]


class TileLangCPU:
    float32 = torch.float32
    sigmoid = staticmethod(torch.sigmoid)
    exp = staticmethod(torch.exp)
    clamp = staticmethod(torch.clamp)
    serial = staticmethod(range)

    @staticmethod
    def Parallel(*shape):
        return range(shape[0]) if len(shape) == 1 else itertools.product(*(range(x) for x in shape))

    @staticmethod
    def alloc_fragment(shape, dtype):
        return torch.empty(shape, dtype=dtype)

    @staticmethod
    def reduce_max(source, target, dim):
        target.copy_(source.amax(dim=dim))

    @staticmethod
    def reduce_sum(source, target, dim):
        target.copy_(source.sum(dim=dim))


def real_mix_block(mixes, scale, base, *, xing):
    source = ROOT / "python/sglang/kernels/ops/layernorm/mhc.py"
    function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "mhc_pre_big_fuse_tilelang")
    block = next(n for n in ast.walk(function) if isinstance(n, ast.If) and "get_thread_binding" in ast.unparse(n.test))
    body = ast.Module(body=copy.deepcopy(block.body), type_ignores=[])
    namespace = dict(T=TileLangCPU, hc_mult=4, mixes_shared=mixes, hc_scale=scale,
        hc_base=base, post_mix=torch.empty(1, 4), comb_mix=torch.empty(1, 16),
        i=0, hc_post_mult_value=2.0, hc_sinkhorn_eps=1e-6,
        sinkhorn_repeat=20, xing_sinkhorn=xing)
    exec(compile(ast.fix_missing_locations(body), str(source), "exec"), namespace)
    return namespace["post_mix"], namespace["comb_mix"].view(4, 4)


def reference(mixes, scale, base):
    # Pinned official HF c500d681 Xing4_0HyperConnection.forward.
    logits = (mixes[8:] * scale[2] + base[8:]).view(4, 4).clamp(-30, 30)
    comb = torch.exp(logits - logits.amax(-1, keepdim=True))
    for _ in range(20):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    return comb


@pytest.mark.parametrize("magnitude", [0.0, 1.0, 15.0, 100.0])
def test_actual_tilelang_xing_block_matches_official_reference(magnitude):
    generator = torch.Generator().manual_seed(20261023)
    mixes = torch.randn(24, generator=generator) * magnitude
    scale = torch.tensor([0.125, 0.375, 1.125])
    base = torch.randn(24, generator=generator) * magnitude
    post, transposed = real_mix_block(mixes, scale, base, xing=True)
    expected = reference(mixes, scale, base)
    torch.testing.assert_close(transposed.T, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(post[0], 2 * torch.sigmoid(mixes[4:8] * scale[1] + base[4:8]))
    assert torch.isfinite(transposed).all()


def test_nonxing_default_block_retains_original_math():
    mixes = torch.arange(24, dtype=torch.float32).sin() * 2
    scale, base = torch.tensor([0.1, 0.2, 0.3]), torch.arange(24).float().cos()
    _, actual = real_mix_block(mixes, scale, base, xing=False)
    logits = (mixes[8:] * scale[2] + base[8:]).view(4, 4)
    expected = torch.exp(logits - logits.amax(1, keepdim=True))
    expected = expected / expected.sum(1, keepdim=True) + 1e-6
    expected = expected / (expected.sum(0, keepdim=True) + 1e-6)
    for _ in range(19):
        expected = expected / (expected.sum(1, keepdim=True) + 1e-6)
        expected = expected / (expected.sum(0, keepdim=True) + 1e-6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [0, 1, 32, 2048, 2049])
def test_module_cuda_layout_and_empty_contract(tokens, monkeypatch):
    from sglang.srt.models import xing4_0 as model
    from sglang.srt.configs.xing4_0 import Xing4_0Config

    monkeypatch.setattr(model, "_is_cuda", True)
    module = model.mHCModule(Xing4_0Config(hidden_size=8), 0)
    module._finalized = True
    module.use_hf_reference = False
    expected = torch.arange(tokens * 16, dtype=torch.float32).view(tokens, 4, 4)
    def pre(**kwargs):
        return torch.ones(tokens, 4, 1), expected, torch.zeros(tokens, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(model, "mhc_pre", pre)
    aggregated, comb, post = module(torch.zeros(tokens, 1, 32, dtype=torch.bfloat16))
    assert aggregated.shape == (tokens, 1, 8)
    assert comb.shape == (tokens, 4, 4) and comb.is_contiguous()
    assert post.shape == (tokens, 4, 1)
    torch.testing.assert_close(comb, expected, rtol=0, atol=0)


def test_fp8_checkpoint_scale_preserves_fp32_values():
    from sglang.srt.models.xing4_0 import mHCModule, Xing4_0ForCausalLM
    from sglang.srt.configs.xing4_0 import Xing4_0Config

    class Harness(nn.Module):
        load_weights = Xing4_0ForCausalLM.load_weights
        def __init__(self):
            super().__init__()
            self.attn_hc = mHCModule(Xing4_0Config(hidden_size=8), 0)
        def do_load_weights(self, weights):
            assert not weights
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        owner = Harness()
    finally:
        torch.set_default_dtype(original_dtype)
    scale = torch.tensor([0.123456789, 0.618033989, 1.41421356], dtype=torch.float32)
    weights = [("attn_hc." + name, scale if name == "hc_scale" else torch.zeros_like(param))
        for name, param in owner.attn_hc.named_parameters()]
    owner.load_weights(weights)
    assert owner.attn_hc.hc_scale.dtype == torch.float32
    torch.testing.assert_close(owner.attn_hc.scale_fp32, scale, rtol=0, atol=0)


def test_xing_default_target_has_no_dedicated_stream():
    source = (ROOT / "python/sglang/srt/models/xing4_0.py").read_text()
    tree = ast.parse(source)
    constructor = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Xing4_0Model")
    assert "torch.cuda.Stream" not in ast.unparse(constructor)
    layer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Xing4_0DecoderLayer")
    init = next(n for n in layer.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert "moe_alt_stream" not in ast.unparse(init)
