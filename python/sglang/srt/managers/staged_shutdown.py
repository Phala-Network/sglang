"""Staged release of a scheduler rank's GPU-side resources on graceful exit.

Freeing everything at process exit is one call into the NVIDIA driver per
process, and the driver holds its global API lock for the whole of it: every
other process on the host, on any GPU, waits. Under Confidential Computing
that teardown took 32-42 s per rank on an 8x B200 host, back to back for the
four ranks of a TP4 instance, and stalled the other instance serving on the
same host for ~3 minutes (no driver call of it could complete).

Released explicitly, one object at a time, every free is a short hold of that
lock that other processes can get in between. So on a graceful exit
(ShutdownReq) and with SGLANG_STAGED_SHUTDOWN set, a rank:

1. detaches the HiCache storage backend and stops the rank-consensus worker,
   in parallel with its peers (both may still talk to them);
2. then, holding a lock file shared by the instance's ranks so they take turns
   (SGLANG_STAGED_SHUTDOWN_LOCK; a path on a volume shared by several
   instances serializes those too):
   a. releases host resources, which closes an external host pool such as
      ccbridge's (its channels, tier imports and tier device file);
   b. drops the KV and req-to-token pools' device tensors and empties the
      CUDA caching allocator, one segment free at a time;
   c. aborts its NCCL communicators (local, never collective).

What is left (weights, CUDA graphs, the context) goes at process exit. Every
step is best effort: a failure is logged and the next step still runs, since
whatever a step leaves behind is reclaimed at exit anyway.
"""

from __future__ import annotations

import contextlib
import fcntl
import gc
import logging
import os
import time
from typing import Any, Iterator, Optional

import torch

from sglang.srt.distributed.parallel_state import abort_distributed_environment
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Objects whose device tensors drop_device_tensors may release: the memory
# pools and their allocators, nothing that a still-running thread might own.
_POOL_MODULE_PREFIX = "sglang.srt.mem_cache"


@contextlib.contextmanager
def teardown_turn(path: Optional[str], timeout: float, label: str) -> Iterator[None]:
    """Hold an exclusive flock on `path` for the body, waiting at most `timeout`.

    Past the timeout the body runs anyway (a peer stuck in its turn must not
    keep this rank's resources alive until it is SIGKILLed). No path: no turn.
    """
    if not path:
        yield
        return
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    except OSError as e:
        logger.warning(f"{label}: cannot open {path} ({e}); releasing without a turn")
        yield
        return
    try:
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start >= timeout:
                    logger.warning(
                        f"{label}: no turn after {timeout:.0f}s; releasing anyway"
                    )
                    break
                time.sleep(0.2)
        waited = time.monotonic() - start
        if waited >= 1:
            logger.info(f"{label}: waited {waited:.1f}s for its turn")
        yield
    finally:
        os.close(fd)  # drops the lock


def drop_device_tensors(
    *roots: Any, max_depth: int = 4, device_type: str = "cuda"
) -> int:
    """Unbind every device tensor held by the memory-pool objects reachable from `roots`.

    Walks attributes of objects defined under sglang.srt.mem_cache (pools,
    allocators, nested sub-pools), up to `max_depth` levels, and sets every
    attribute holding a tensor on `device_type`, or a list / tuple / dict of
    only such tensors, to None. The pools are unusable afterwards. Returns the
    bytes of the distinct storages unbound; they return to the caching
    allocator once nothing else references them.
    """
    seen: set[int] = set()
    storages: dict[int, int] = {}

    def on_device(v: Any) -> bool:
        return isinstance(v, torch.Tensor) and v.device.type == device_type

    def holds_only_device_tensors(v: Any) -> bool:
        if isinstance(v, (list, tuple)):
            items = v
        elif isinstance(v, dict):
            items = list(v.values())
        else:
            return False
        return len(items) > 0 and all(on_device(x) for x in items)

    def note(t: torch.Tensor) -> None:
        s = t.untyped_storage()
        storages[s.data_ptr()] = s.nbytes()

    def walk(obj: Any, depth: int) -> None:
        if depth > max_depth or id(obj) in seen:
            return
        seen.add(id(obj))
        attrs = getattr(obj, "__dict__", None)
        if attrs is None:
            return
        for name, v in list(attrs.items()):
            if on_device(v):
                note(v)
                setattr(obj, name, None)
            elif holds_only_device_tensors(v):
                for t in v.values() if isinstance(v, dict) else v:
                    note(t)
                setattr(obj, name, None)
            elif type(v).__module__.startswith(_POOL_MODULE_PREFIX):
                walk(v, depth + 1)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    if type(x).__module__.startswith(_POOL_MODULE_PREFIX):
                        walk(x, depth + 1)

    for root in roots:
        if root is not None:
            walk(root, 0)
    return sum(storages.values())


def _step(label: str, name: str, fn) -> None:
    start = time.monotonic()
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - best effort, exit reclaims the rest
        logger.warning(f"{label}: {name} failed: {e!r}")
    logger.info(f"{label}: {name} {time.monotonic() - start:.1f}s")


def staged_release(scheduler: Any, tp_rank: int) -> None:
    """The graceful-exit release of one scheduler rank (see the module doc)."""
    label = f"Staged shutdown TP{tp_rank}"
    tree_cache = getattr(scheduler, "tree_cache", None)

    # 1. With the peers: neither may wait on a rank that is waiting for its turn.
    if hasattr(tree_cache, "shutdown"):
        _step(label, "storage detach", tree_cache.shutdown)
    from sglang.srt.utils import rank_consensus_checker

    _step(label, "rank consensus", rank_consensus_checker.shutdown)

    # 2. One rank at a time.
    with teardown_turn(
        envs.SGLANG_STAGED_SHUTDOWN_LOCK.get(),
        envs.SGLANG_STAGED_SHUTDOWN_LOCK_TIMEOUT.get(),
        label,
    ):
        _step(label, "host resources", scheduler.release_host_resources)

        def release_kv() -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            before = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            dropped = drop_device_tensors(
                getattr(scheduler, "token_to_kv_pool_allocator", None),
                getattr(scheduler, "req_to_token_pool", None),
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                after = torch.cuda.memory_reserved()
                logger.info(
                    f"{label}: unbound {dropped / 2**30:.1f} GiB of pool tensors, "
                    f"reserved {before / 2**30:.1f} -> {after / 2**30:.1f} GiB"
                )

        _step(label, "KV pools", release_kv)
        _step(label, "NCCL abort", abort_distributed_environment)
