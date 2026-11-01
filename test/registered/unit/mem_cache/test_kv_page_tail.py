"""KVCache.zero_page_tails: the rows of each page a forward writes that come
after its last write there are zeroed in every layer, nothing else changes, and
only layouts with one row per token location take part."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

PAGE = 64
PAGES = 16
ROWS = (PAGES + 1) * PAGE  # pool size + page_size


def _locs(*runs):
    """Locations of (page, first offset, count) runs, in forward order."""
    return [p * PAGE + o + i for p, o, n in runs for i in range(n)]


def _mla():
    return MLATokenToKVPool(
        size=PAGES * PAGE,
        page_size=PAGE,
        dtype=torch.float8_e4m3fn,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=3,
        device="cuda",
        enable_memory_saver=False,
    )


def _mha(layout=None):
    return MHATokenToKVPool(
        size=PAGES * PAGE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=128,
        layer_num=2,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
        kv_cache_layout=layout,
    )


def _buffers(pool):
    return [b for lst in pool.page_tail_buffer_lists() for b in lst]


def _fill(bufs):
    # A fill without a zero byte (0xFF is NaN in fp8 e4m3fn and bf16).
    for b in bufs:
        b.view(torch.uint8).fill_(0xFF)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestKvPageTail(unittest.TestCase):
    def _check(self, pool, locs, zeroed, dtype=torch.int64):
        bufs = _buffers(pool)
        self.assertTrue(bufs)
        _fill(bufs)
        pool.zero_page_tails(torch.tensor(locs, dtype=dtype, device="cuda"))
        torch.cuda.synchronize()
        expect = torch.zeros(ROWS, dtype=torch.bool)
        expect[zeroed] = True
        for buf in bufs:
            got = (buf.view(torch.uint8).flatten(1) == 0).all(1).cpu()
            wrong = torch.nonzero(got ^ expect).flatten().tolist()
            self.assertEqual(wrong, [], "rows zeroed or left wrongly")

    def test_tails(self):
        cases = [
            # an extend that ends inside a fresh page
            ([(3, 0, 45)], list(range(3 * PAGE + 45, 4 * PAGE))),
            # a continuing chunk, then a request that fills a page and enters the next
            (
                [(5, 10, 11), (7, 0, PAGE), (8, 0, 6)],
                list(range(5 * PAGE + 21, 6 * PAGE))
                + list(range(8 * PAGE + 6, 9 * PAGE)),
            ),
            # verify rows: one ends on a page boundary, one crosses into a new page
            (
                [(9, 60, 4), (10, 62, 2), (11, 0, 2)],
                list(range(11 * PAGE + 2, 12 * PAGE)),
            ),
            # one-token decodes, and padded rows (-1)
            ([(12, 7, 1), (13, 63, 1)], list(range(12 * PAGE + 8, 13 * PAGE))),
        ]
        pools = {"mla": _mla(), "mha": _mha()}
        pools["hybrid"] = SimpleNamespace(
            full_kv_pool=pools["mla"],
            page_tail_buffer_lists=pools["mla"].page_tail_buffer_lists,
            zero_page_tails=lambda locs: HybridLinearKVPool.zero_page_tails(
                SimpleNamespace(full_kv_pool=pools["mla"]), locs
            ),
        )
        for name, pool in pools.items():
            for runs, zeroed in cases:
                for dtype in (torch.int64, torch.int32):
                    with self.subTest(pool=name, runs=runs, dtype=dtype):
                        locs = _locs(*runs)
                        if runs[0][0] == 12:
                            locs = locs + [-1, -1]
                        self._check(pool, locs, zeroed, dtype)

    def test_noops(self):
        pool = _mla()
        bufs = _buffers(pool)
        _fill(bufs)
        pool.zero_page_tails(torch.empty(0, dtype=torch.int64, device="cuda"))
        pool.zero_page_tails(None)
        torch.cuda.synchronize()
        for b in bufs:
            self.assertTrue(bool((b.view(torch.uint8) == 0xFF).all()))

    def test_page_major_layouts_opt_out(self):
        # HND rows are whole pages: zeroing 63 "rows" would clear other
        # requests' pages. Such pools must not take part at all.
        with envs.SGLANG_USE_HND_KVCACHE.override(True):
            hnd = _mha()
        self.assertEqual(hnd.page_tail_buffer_lists(), [])
        bufs = list(hnd.k_buffer) + list(hnd.v_buffer)
        _fill(bufs)
        hnd.zero_page_tails(torch.tensor(_locs((3, 0, 5)), device="cuda"))
        torch.cuda.synchronize()
        for b in bufs:
            self.assertTrue(bool((b.view(torch.uint8) == 0xFF).all()))
        # ROCm AITER's vectorized_5d (it reports device "cuda" too): only the
        # layout label matters for opting out.
        vec = _mha()
        vec.kv_cache_layout = "vectorized_5d"
        self.assertEqual(vec.page_tail_buffer_lists(), [])


if __name__ == "__main__":
    unittest.main()
