"""Philox draws for CUDA graphs that advance their own key on the device.

A graph that consumes torch's RNG gets its seed and offset refreshed by two
eager fills before every replay; under GPU CC each eager kernel costs ~38 us
of GPU front-end time. Here the key lives in a device tensor that the graph
increments after each draw, so a replay needs no host-side setup.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _graph_rand_kernel(
    out_ptr, key_ptr, n, EXPONENTIAL: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Strictly inside (0, 1): tl.rand can return 0 and round to 1, and a zero
    # exponential noise would make its token win the noisy argmax outright.
    r = tl.randint(tl.load(key_ptr), offs) >> 9
    u = (r.to(tl.float32) + 0.5) * (1.0 / 8388608.0)
    if EXPONENTIAL:
        u = -tl.log(u)
    tl.store(out_ptr + offs, u, mask=offs < n)


class GraphRng:
    def __init__(self, device, salt: int):
        # Same key on every TP rank, as with torch's CUDA generator (one seed
        # per server): ranks that do not broadcast a draw must draw the same.
        gen = torch.Generator().manual_seed(torch.cuda.initial_seed() + salt)
        self.key = torch.randint(0, 2**62, (1,), dtype=torch.int64, generator=gen).to(
            device
        )

    def _draw(self, out: torch.Tensor, exponential: bool) -> torch.Tensor:
        assert out.dtype == torch.float32 and out.is_contiguous()
        n = out.numel()
        block = 1024
        _graph_rand_kernel[(triton.cdiv(n, block),)](
            out, self.key, n, EXPONENTIAL=exponential, BLOCK=block
        )
        self.key.add_(1)
        return out

    def rand(self, *shape: int, device) -> torch.Tensor:
        return self._draw(torch.empty(shape, dtype=torch.float32, device=device), False)

    def exponential_(self, out: torch.Tensor) -> torch.Tensor:
        return self._draw(out, True)
