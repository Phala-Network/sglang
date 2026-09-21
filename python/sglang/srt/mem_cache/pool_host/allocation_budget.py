"""Startup-only admission for the single-host DSA KV + index allocation.

Every physical worker enters exactly once, before either host buffer exists.
The common snapshot is split into disjoint per-worker credits, and every real
allocation consumes credit cumulatively. There are no per-pool collectives.
This does not arbitrate independently launched jobs sharing the same host.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from contextlib import contextmanager
from pathlib import Path

import psutil
import torch

from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import requested_hugepage_bytes
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)
_startup_lock = threading.Lock()
_active_budget = None


def active_host_allocation_budget():
    # Deliberately process-wide: allocator calls from helper threads must use
    # the same locked ledger, not an unpropagated thread-local/context variable.
    return _active_budget


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def host_slot_metadata_bytes(logical_slots: int) -> int:
    # clear() allocates three independent ordinary tensors: uint8 state,
    # int64 free slots and bool used flags. Round each separately.
    return 2 * _round_up(logical_slots, 4096) + _round_up(8 * logical_slots, 4096)


def host_memory_reserve_bytes() -> int:
    reserve_gb = envs.SGLANG_HICACHE_HOST_MEMORY_RESERVE_GB.get()
    if reserve_gb < 0:
        raise ValueError(
            "SGLANG_HICACHE_HOST_MEMORY_RESERVE_GB must be non-negative, "
            f"got {reserve_gb}"
        )
    return reserve_gb * 1024**3


def _cgroup_headroom(resource: str) -> int | None:
    """Minimum v2 headroom over this process's visible cgroup ancestors."""
    membership = Path("/proc/self/cgroup").read_text().splitlines()
    unified = next((line[3:] for line in membership if line.startswith("0::")), None)
    if unified is None:
        raise RuntimeError("DSA startup budget currently requires cgroup v2")
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        if after.split()[0] != "cgroup2":
            continue
        fields = before.split()
        mount_root, mountpoint = fields[3], Path(fields[4])
        if mount_root == "/":
            relative = unified.lstrip("/")
        elif unified == mount_root or unified.startswith(mount_root + "/"):
            relative = unified[len(mount_root):].lstrip("/")
        else:
            continue
        leaf = mountpoint / relative
        if not leaf.is_dir():
            raise RuntimeError(f"Cannot resolve process cgroup directory: {leaf}")
        limits = []
        for directory in (leaf, *leaf.parents):
            maximum = directory / f"{resource}.max"
            if maximum.exists():
                value = maximum.read_text().strip()
                if value != "max":
                    current = int((directory / f"{resource}.current").read_text())
                    limits.append(max(0, int(value) - current))
            if directory == mountpoint:
                return min(limits) if limits else None
        raise RuntimeError("Cgroup path escaped its mount")
    raise RuntimeError("Cannot locate this process's cgroup v2 mount")


def _parse_nodes(value: str) -> set[int]:
    nodes = set()
    for item in value.strip().split(","):
        bounds = [int(part) for part in item.split("-")]
        nodes.update(range(bounds[0], bounds[-1] + 1))
    return nodes


def _resource_snapshot(hugepage_bytes: int) -> tuple[int, int]:
    ordinary = int(psutil.virtual_memory().available)
    cgroup = _cgroup_headroom("memory")
    if cgroup is not None:
        ordinary = min(ordinary, cgroup)
    ordinary = max(0, ordinary - host_memory_reserve_bytes())
    if not hugepage_bytes:
        return ordinary, 0

    size_kb = hugepage_bytes // 1024
    name = f"hugepages-{size_kb}kB"
    directory = Path("/sys/kernel/mm/hugepages") / name
    free = int((directory / "free_hugepages").read_text())
    reserved = int((directory / "resv_hugepages").read_text())
    status = Path("/proc/self/status").read_text().splitlines()
    allowed = _parse_nodes(next(line.split(":", 1)[1] for line in status
                                if line.startswith("Mems_allowed_list:")))
    online = _parse_nodes(Path("/sys/devices/system/node/online").read_text())
    if allowed != online:
        # Linux exposes global reservations, not per-node reservations. Deduct
        # all of them from allowed-node free pages: conservative, never additive.
        free = min(free, sum(int((Path(f"/sys/devices/system/node/node{node}") /
                                 "hugepages" / name / "free_hugepages").read_text())
                             for node in allowed))
    huge = max(0, free - reserved) * hugepage_bytes
    size_label = "2MB" if hugepage_bytes == 2 * 1024**2 else "1GB"
    for resource in (f"hugetlb.{size_label}", f"hugetlb.{size_label}.rsvd"):
        limit = _cgroup_headroom(resource)
        if limit is not None:
            huge = min(huge, limit)
    return ordinary, huge


def _gather(world, packet):
    if world is None:
        return [packet]
    packets = [None] * world.world_size
    torch.distributed.all_gather_object(packets, packet, group=world.cpu_group)
    return packets


class HostAllocationBudget:
    def __init__(self, ordinary_bytes: int, huge_bytes: int, hugepage_bytes: int):
        self.remaining = {False: ordinary_bytes, True: huge_bytes}
        self.initial = dict(self.remaining)
        self.hugepage_bytes = hugepage_bytes
        self.lock = threading.Lock()
        self.buffers = []
        self.pools = []
        self.closed = False

    def _mapping_domain(self, allocator):
        from sglang.srt.mem_cache.pool_host.common import host_mapping_page_size

        page_size = host_mapping_page_size(allocator)
        huge = page_size in (2 * 1024**2, 1024**3)
        if (page_size if huge else 0) != self.hugepage_bytes:
            raise RuntimeError("Host page mode changed during the allocation transaction")
        return huge, page_size

    def _claim(self, nbytes: int, huge: bool) -> None:
        if nbytes < 0:
            raise ValueError("Negative host allocation")
        with self.lock:
            if self.closed:
                raise RuntimeError("Host allocation transaction is closed")
            if nbytes > self.remaining[huge]:
                raise ValueError(
                    f"Not enough {'HugeTLB' if huge else 'ordinary'} host memory: "
                    f"request={nbytes}, remaining_rank_credit={self.remaining[huge]}, "
                    f"initial_rank_credit={self.initial[huge]}"
                )
            self.remaining[huge] -= nbytes

    def register_pool(self, pool, metadata_bytes: int) -> int:
        # Metadata is always ordinary RAM, even when data uses HugeTLB.
        self._claim(_round_up(metadata_bytes, 4096), False)
        huge, _ = self._mapping_domain(pool.allocator)
        with self.lock:
            self.pools.append(pool)
            return self.remaining[huge]

    def claim_mapping(self, nbytes: int, allocator) -> None:
        huge, page_size = self._mapping_domain(allocator)
        self._claim(_round_up(nbytes, page_size), huge)

    def record_buffer(self, buffer) -> None:
        with self.lock:
            self.buffers.append(buffer)

    def rollback(self) -> None:
        from sglang.srt.mem_cache.pool_host.common import (
            _CUDA_HOST_REGISTERED_RANGES_ATTR,
            _cuda_host_unregister,
        )

        errors = []
        for buffer in reversed(self.buffers):
            if getattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, None):
                _cuda_host_unregister(buffer)
                if getattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, None):
                    errors.append("cudaHostUnregister left registered ranges")
        if errors:
            # Retain owners if CUDA still holds a registration; never quietly
            # free a mapped buffer still reachable by the driver.
            raise RuntimeError("Host allocation rollback failed: " + "; ".join(errors))
        for pool in reversed(self.pools):
            pool.destroy()
        self.pools.clear()
        self.buffers.clear()
        with self.lock:
            self.remaining = dict(self.initial)
            self.closed = True

    def commit(self) -> None:
        with self.lock:
            self.closed = True
        # Pools now own the tensors. Nothing is retained on a request hot path.
        self.pools.clear()
        self.buffers.clear()


@contextmanager
def dsa_host_allocation_budget(params, *, storage_backend=None, host_memory_mode="cache"):
    """One coordinated entry/exit for the complete DSA main/index pair."""
    global _active_budget
    if not envs.SGLANG_HICACHE_DSA_STARTUP_BUDGET.get():
        if requested_hugepage_bytes():
            raise ValueError("DSA HugeTLB requires SGLANG_HICACHE_DSA_STARTUP_BUDGET=1")
        yield None
        return
    if storage_backend is not None or host_memory_mode != "cache":
        raise NotImplementedError(
            "DSA startup budget currently covers RAM cache only; storage backends "
            "and buffer_only require a joint external L3/segment allocation plan"
        )
    parallel = get_parallel()
    if parallel.nnodes != 1 or params.pp_size != 1 or parallel.dcp_enabled:
        raise NotImplementedError(
            "DSA startup budget supports single-host PP1 without DCP; "
            "heterogeneous/multi-host initialization requires a full allocation plan"
        )
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not initialized and parallel.tp_size != 1:
        raise RuntimeError("Distributed host budget requested before rank initialization")
    world = get_world_group() if initialized else None
    count = world.world_size if world is not None else 1
    acquired = _startup_lock.acquire(blocking=False)
    if not acquired:
        # Never interleave another transaction's collectives on the same world
        # communicator. The synchronous assembler owns one process-wide scope.
        raise RuntimeError("A host allocation transaction is already active")
    budget = None
    try:
        snapshot_error = None
        ordinary = huge = hugepage_bytes = 0
        try:
            hugepage_bytes = requested_hugepage_bytes()
            ordinary, huge = _resource_snapshot(hugepage_bytes)
        except Exception as exc:
            snapshot_error = f"{type(exc).__name__}: {exc}"
        packets = _gather(world, dict(host=socket.gethostname(), pid=os.getpid(),
                                     ordinary=ordinary, huge=huge, page=hugepage_bytes,
                                     error=snapshot_error))
        errors = [p["error"] for p in packets if p["error"]]
        if errors:
            raise RuntimeError("Host budget snapshot failed: " + "; ".join(errors))
        if len({p["host"] for p in packets}) != 1 or len({p["pid"] for p in packets}) != count:
            raise RuntimeError("Host budget requires every distinct physical worker on one host")
        if len({p["page"] for p in packets}) != 1:
            raise RuntimeError("Workers disagree on host page mode")
        ordinary = min(p["ordinary"] for p in packets) // count
        huge = min(p["huge"] for p in packets)
        huge = (huge // hugepage_bytes // count) * hugepage_bytes if hugepage_bytes else 0
        budget = HostAllocationBudget(ordinary, huge, hugepage_bytes)
        _active_budget = budget
        failure = None
        try:
            yield budget
            if len(budget.buffers) != 2:
                raise RuntimeError("DSA startup transaction must allocate exactly KV and index buffers")
        except BaseException as exc:
            failure = exc
        finally:
            _active_budget = None
        try:
            statuses = _gather(world, {"error": None if failure is None else
                                      f"{type(failure).__name__}: {failure}"})
        except BaseException:
            budget.rollback()
            raise
        errors = [status["error"] for status in statuses if status["error"]]
        if errors:
            budget.rollback()
            raise RuntimeError("DSA host allocation failed: " + "; ".join(errors)) from failure
        budget.commit()
        logger.info("DSA startup host budget committed: workers=%d, ordinary/rank=%d, "
                    "hugetlb/rank=%d, hugepage=%d", count, ordinary, huge, hugepage_bytes)
    finally:
        if acquired:
            _active_budget = None
            _startup_lock.release()
