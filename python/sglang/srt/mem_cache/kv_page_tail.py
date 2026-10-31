"""Zero the unwritten rest of every KV page a forward writes.

A page goes back to the free list with its previous owner's bytes, and its next
owner fills it from the front. Paged decode kernels (trtllm-gen MLA, also for
the multi-token target verify) load whole KV tiles and mask the positions past
a sequence's length in the scores, and ``0 * NaN`` is NaN: a non-finite value a
NaN-poisoned request left in a page reaches whichever request gets the page
next. Zeroing the rows after the last one each forward writes keeps every
position past a sequence's end finite.
"""

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl


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


def _buffer_lists(pool) -> List[list]:
    pool = getattr(pool, "full_kv_pool", pool)
    kv = getattr(pool, "kv_buffer", None)
    if kv:
        return [kv]
    lists = (getattr(pool, "k_buffer", None), getattr(pool, "v_buffer", None))
    return [x for x in lists if x]


def kv_page_buffers(pool) -> List[torch.Tensor]:
    """The per-layer KV buffers of a MLA or MHA pool, indexed by location."""
    bufs = [b for lst in _buffer_lists(pool) for b in lst]
    # A row is zeroed as stride(0) bytes, which is only that row when dense.
    return [b for b in bufs if b.is_cuda and b.is_contiguous() and b.numel() > 0]


def _table(pool) -> Optional[Tuple[int, torch.Tensor, torch.Tensor]]:
    """(buffer count, device base addresses, device row bytes), cached on the
    pool and rebuilt when it replaces its buffer lists."""
    lists = _buffer_lists(pool)
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
    return table


def zero_kv_page_tails(pool, out_cache_loc: torch.Tensor, page_size: int) -> None:
    """Before a forward: in every layer, zero the rows of each page the forward
    writes that come after the last location it writes there. The page's owner
    writes its positions in order, so those rows hold none of its data yet. One
    launch, no host sync."""
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
