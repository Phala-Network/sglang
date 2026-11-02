"""Zero the unwritten rest of every KV page a forward writes.

A page goes back to the free list with its previous owner's bytes, and its next
owner fills it from the front. Paged decode kernels (trtllm-gen MLA, also for
the multi-token target verify) load whole KV tiles and mask the positions past
a sequence's length in the scores, and ``0 * NaN`` is NaN: a non-finite value a
NaN-poisoned request left in a page reaches whichever request gets the page
next. Zeroing the rows after the last one each forward writes keeps every
position past a sequence's end finite.
"""

import logging
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _zero_kv_page_tails_kernel(
    loc_ptr,
    num_locs,
    buf_ptrs,
    row_bytes_ptr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    j = tl.program_id(1)
    w = tl.load(loc_ptr + i).to(tl.int64)
    nxt = tl.load(loc_ptr + i + 1, mask=i + 1 < num_locs, other=-1).to(tl.int64)
    page_end = (w // PAGE_SIZE + 1) * PAGE_SIZE
    # Only after the last row the forward writes in a page: a request's
    # locations run consecutively within its page.
    last = (w >= 0) & (nxt != w + 1)
    row_bytes = tl.load(row_bytes_ptr + j)
    start = (w + 1) * row_bytes
    stop = tl.where(last, page_end * row_bytes, start)
    base = tl.load(buf_ptrs + j).to(tl.pointer_type(tl.uint8))
    offs = tl.arange(0, BLOCK)
    for off in range(start, stop, BLOCK):
        idx = off + offs
        tl.store(base + idx, tl.zeros([BLOCK], dtype=tl.uint8), mask=idx < stop)


def kv_page_buffers(pool) -> List[torch.Tensor]:
    """The buffers a pool opted in with (`KVCache.page_tail_buffer_lists`), if
    every one has one row per token location; [] otherwise, which skips the
    pool. A row is cleared as stride(0) bytes, so a layout whose leading dim is
    pages must never get here."""
    rows = pool.size + pool.page_size
    bufs = [b for lst in pool.page_tail_buffer_lists() for b in lst]
    ok = all(
        b.is_cuda and b.is_contiguous() and b.dim() >= 2 and b.shape[0] == rows
        for b in bufs
    )
    return bufs if ok else []


def _table(pool) -> Optional[Tuple[int, torch.Tensor, torch.Tensor]]:
    """(buffer count, device base addresses, device row bytes), cached on the
    pool and rebuilt when it replaces its buffer lists."""
    lists = pool.page_tail_buffer_lists()
    cached = getattr(pool, "_kv_page_tail_table", None)
    if (
        cached is not None
        and len(cached[0]) == len(lists)
        and all(a is b for a, b in zip(cached[0], lists))
    ):
        return cached[1]
    bufs = kv_page_buffers(pool)
    table = None
    if bufs:
        device = bufs[0].device
        table = (
            len(bufs),
            torch.tensor(
                [b.data_ptr() for b in bufs], dtype=torch.int64, device=device
            ),
            torch.tensor(
                [b.stride(0) * b.element_size() for b in bufs],
                dtype=torch.int64,
                device=device,
            ),
        )
    pool._kv_page_tail_table = (lists, table)
    logger.info(
        "KV page-tail zeroing: %s, %d buffers",
        type(pool).__name__,
        table[0] if table else 0,
    )
    return table


def zero_kv_page_tails(pool, out_cache_loc: Optional[torch.Tensor]) -> None:
    """Before a forward: in every layer, zero the rows of each page the forward
    writes that come after the last location it writes there. The page's owner
    writes its positions in order, so those rows hold none of its data yet. One
    launch, no host sync. Pools that did not opt in are left alone."""
    page_size = pool.page_size
    if page_size <= 1 or out_cache_loc is None or out_cache_loc.numel() == 0:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    table = _table(pool)
    if table is None:
        return
    num_bufs, ptrs, row_bytes = table
    locs = out_cache_loc
    if locs.dtype not in (torch.int32, torch.int64):
        locs = locs.to(torch.int64)
    _zero_kv_page_tails_kernel[(locs.numel(), num_bufs)](
        locs,
        locs.numel(),
        ptrs,
        row_bytes,
        PAGE_SIZE=page_size,
        BLOCK=1024,
    )
