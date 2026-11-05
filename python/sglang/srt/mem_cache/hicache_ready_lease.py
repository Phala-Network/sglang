"""Run the HiCache ready-counts all-reduce only while some rank has work to drain.

Stock `check_hicache_events` MIN-reduces the ack and storage queue counts on every
scheduler pass. No rank can skip that alone: its peers would block in the collective,
or pair it with a later one. So skipping is a decision of the group, taken from the
one thing every rank already shares, the result of the collective itself:

- each round also reduces "this rank has something to drain", so the result says
  whether ANY rank has;
- if none has, every rank skips the next ``passes - 1`` passes, counted from that same
  result. Passes are in lockstep, because each one enters this collective or none does.

The counts keep their MIN semantics, and acks are still drained only by the counts of
a round, at the same point of the same pass on every rank. Work that shows up during a
lease is drained at the next round: late by at most ``passes - 1`` passes, which holds
host locks and staging that much longer, and is never wrong. So the verdict decides
when the next round is, and nothing else. Unfinished backups do not hold a lease off,
or a steady stream of them would keep this collective on every pass; loads do, since
requests may be waiting for their slots.

The file sits outside `unified_radix_cache.py` because ccbridge pins that file's bytes.
"""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_device, get_disagg, get_memory, get_parallel


def lease_eligible(cache) -> bool:
    """The modes `_async_ready_counts_eligible` validated, whose decision is the same on
    every rank; an idle pass under sleep_on_idle may last until the next request."""
    if (
        type(cache).check_hicache_events is not UnifiedRadixCache.check_hicache_events
        or cache.cache_controller is None
    ):
        return False
    memory_write_policy = get_memory().hicache_write_policy
    return (
        cache.pp_size == 1
        and get_parallel().dp_size == 1
        and cache._ready_counts_group is not None
        and memory_write_policy in ("write_through", "write_through_selective")
        and cache.cache_controller.write_policy == memory_write_policy
        and not cache.is_write_back
        and get_disagg().disaggregation_mode == "null"
        and cache.host_memory_mode == "cache"
        and cache.buffer_pipeline is None
        and cache.linker is None
        and not get_device().sleep_on_idle
    )


def check_hicache_events_leased(cache, passes: int) -> None:
    """`check_hicache_events`, with its collective skipped under a group lease."""
    if not lease_eligible(cache):
        cache.check_hicache_events()
        return

    # What stock runs on every pass; neither is a collective here (pp_size == 1).
    cache._drain_async_work()
    cache.flush_pending_backups()

    left = getattr(cache, "_ready_counts_lease_left", 0)
    if left > 0:
        cache._ready_counts_lease_left = left - 1
        return

    cache._drain_pending_ready_counts()
    ready_counts, extra_pool_names, digest = cache._ready_counts_tensor()
    values = ready_counts.tolist()
    has_work = any(values[:-2]) or bool(cache.cache_controller.ack_load_queue)
    # MIN of -has_work: -1 if any rank has work.
    reduced = torch.tensor([*values, -int(has_work)], dtype=torch.int64, device="cpu")
    torch.distributed.all_reduce(
        reduced, op=torch.distributed.ReduceOp.MIN, group=cache._ready_counts_group
    )
    cache._ready_counts_lease_left = 0 if reduced[-1].item() < 0 else passes - 1
    cache._apply_ready_counts(
        *cache._parse_ready_counts(reduced[:-1], extra_pool_names, digest)
    )
    _log_storage_metrics(cache)


def _log_storage_metrics(cache) -> None:
    # Stock's tail, on rounds only: these are cumulative.
    if not cache.enable_storage_metrics or cache.storage_metrics_collector is None:
        return
    storage_metrics = cache.cache_controller.storage_backend.get_stats()
    if storage_metrics is None:
        from sglang.srt.observability.metrics_collector import StorageMetrics

        storage_metrics = StorageMetrics()
    if not hasattr(storage_metrics, "prefetch_stats"):
        storage_metrics.prefetch_stats = cache.prefetch_outcome_stats_snapshot()
    cache.storage_metrics_collector.log_storage_metrics(storage_metrics)
