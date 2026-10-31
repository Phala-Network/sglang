"""zero_kv_page_tails: the rows of each page a forward writes that come after
its last write there are zeroed in every layer, and nothing else changes."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv_page_tail import zero_kv_page_tails
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

PAGE = 64
PAGES = 16


def _locs(*runs):
    """Locations of (page, first offset, count) runs, in forward order."""
    return [p * PAGE + o + i for p, o, n in runs for i in range(n)]


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestKvPageTail(unittest.TestCase):
    def _pools(self):
        rows = (PAGES + 1) * PAGE
        mla = SimpleNamespace(
            kv_buffer=[
                torch.full((rows, 1, 576), 0xFF, dtype=torch.uint8, device="cuda")
                for _ in range(3)
            ]
        )
        mha = SimpleNamespace(
            k_buffer=[
                torch.full((rows, 2, 128), float("nan"), device="cuda").bfloat16()
                for _ in range(2)
            ],
            v_buffer=[
                torch.full((rows, 2, 128), float("nan"), device="cuda").bfloat16()
                for _ in range(2)
            ],
        )
        # A hybrid pool wraps the full-attention pool.
        return mla, SimpleNamespace(full_kv_pool=mha)

    def _check(self, pool, bufs, locs, zeroed, dtype=torch.int64):
        zero_kv_page_tails(pool, torch.tensor(locs, dtype=dtype, device="cuda"), PAGE)
        torch.cuda.synchronize()
        expect = torch.zeros((PAGES + 1) * PAGE, dtype=torch.bool)
        expect[zeroed] = True
        for buf in bufs:
            # The fill has no zero byte, so a zero row was zeroed here.
            got = (buf.view(torch.uint8).flatten(1) == 0).all(1).cpu()
            wrong = torch.nonzero(got ^ expect).flatten().tolist()
            self.assertEqual(wrong, [], "rows zeroed or left wrongly")

    def test_tails(self):
        # (runs, rows expected zero)
        cases = [
            # an extend that ends inside a fresh page
            ([(3, 0, 45)], list(range(3 * PAGE + 45, 4 * PAGE))),
            # a continuing chunk, then a request that fills a page and enters the next
            ([(5, 10, 11), (7, 0, PAGE), (8, 0, 6)],
             list(range(5 * PAGE + 21, 6 * PAGE)) + list(range(8 * PAGE + 6, 9 * PAGE))),
            # verify rows: one ends on a page boundary, one crosses into a new page
            ([(9, 60, 4), (10, 62, 2), (11, 0, 2)], list(range(11 * PAGE + 2, 12 * PAGE))),
            # one-token decodes
            ([(12, 7, 1), (13, 63, 1)], list(range(12 * PAGE + 8, 13 * PAGE))),
        ]
        for runs, zeroed in cases:
            for dtype in (torch.int64, torch.int32):
                with self.subTest(runs=runs, dtype=dtype):
                    for pool in self._pools():
                        inner = getattr(pool, "full_kv_pool", pool)
                        bufs = getattr(inner, "kv_buffer", None) or (
                            inner.k_buffer + inner.v_buffer
                        )
                        self._check(pool, bufs, _locs(*runs), zeroed, dtype)

    def test_padding_and_noops(self):
        mla, _ = self._pools()
        # Padded rows (-1) are skipped; an empty or missing batch does nothing.
        self._check(mla, mla.kv_buffer, [-1, -1], [])
        zero_kv_page_tails(mla, torch.empty(0, dtype=torch.int64, device="cuda"), PAGE)
        zero_kv_page_tails(mla, None, PAGE)
        zero_kv_page_tails(mla, torch.tensor([5], device="cuda"), 1)
        # A pool without row-major buffers is left alone.
        zero_kv_page_tails(SimpleNamespace(), torch.tensor([5], device="cuda"), PAGE)
        torch.cuda.synchronize()
        for buf in mla.kv_buffer:
            self.assertTrue(bool((buf == 0xFF).all()))

    def test_strided_buffer_is_skipped(self):
        rows = (PAGES + 1) * PAGE
        base = torch.full((rows, 2, 64), 0xFF, dtype=torch.uint8, device="cuda")
        pool = SimpleNamespace(kv_buffer=[base[:, 0]])  # not dense
        zero_kv_page_tails(pool, torch.tensor(_locs((3, 0, 5)), device="cuda"), PAGE)
        torch.cuda.synchronize()
        self.assertTrue(bool((base == 0xFF).all()))


if __name__ == "__main__":
    unittest.main()
