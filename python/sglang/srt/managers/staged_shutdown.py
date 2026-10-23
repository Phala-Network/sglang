"""Release a scheduler rank's driver objects while the process is alive.

When a process exits (SIGKILL or a plain exit), the NVIDIA driver frees every
RM object the process still holds in ONE exclusive hold of its global lock,
~0.25-0.3 ms per object whatever its size. Any other process on the host that
allocates or frees GPU memory (any GPU) waits for the whole of it: on b200
(8x B200, Confidential Computing on) each exiting rank of one TP4 instance
stalled the other instance for 32-42 s. Freed while the process is alive,
every free is its own short hold and nobody waits (H200 study,
ccbridge-experiments/h200-rmlock-2026-10/REPORT.md).

So on a graceful exit (ShutdownReq) and with SGLANG_STAGED_SHUTDOWN set, a
rank runs `staged_release`, one timed step after another:

1. ``inventory``: what it holds that maps to RM objects (TP0 logs it);
2. ``synchronize``: the event loop has stopped; let the GPU finish;
3. ``storage detach`` and ``rank consensus``: with its peers (both may still
   talk to them);
4. ``CUDA graphs``: reset every captured graph (decode, prefill, verify,
   draft; their graph execs, and their private memory pools go back to the
   caching allocator);
5. ``plugin <name>`` and ``host resources``: every hook registered with
   `register_release_hook` (ccbridge gives back its CE channel, tier import,
   staging VMM buffer, cctier fds and broker leases there), then SGLang's own
   host pools;
6. ``communicators``: custom all-reduce / IPC / symmetric-memory buffers,
   FlashInfer fusion workspaces, DeepEP, PyNccl communicators (ncclCommAbort:
   releases NCCL's cuMem buffers), then torch's process groups (abort, local);
7. ``tensors``: unbind the memory pools' tensors, then every CUDA tensor still
   reachable from Python (weights, buffers, graph outputs) and every pinned
   host tensor; gc; empty the caching allocators. This also unmaps and
   releases expandable-segment and symmetric-memory VMM handles, which a
   context reset would not free;
8. ``context reset``: cuDevicePrimaryCtxReset, which frees whatever is still
   context-owned one object at a time;

and then leaves with os._exit(0): nothing in Python may touch CUDA after the
reset. Steps 3-7 run under SGLANG_STAGED_SHUTDOWN_TIMEOUT (past it, the
remaining ones are skipped); every step is also bounded on its own
(SGLANG_STAGED_SHUTDOWN_STEP_TIMEOUT; a step that hangs, e.g. on a peer that
died, is left behind in its thread). The context reset always runs, with what
is left of the budget and at least a step's bound. Every
step is best effort: a failure is logged and the next one still runs.

Ranks need not take turns (every free is a short hold); the lock file of the
first version is kept as an option (SGLANG_STAGED_SHUTDOWN_LOCK, default off).
The tokenizer side waits for the ranks: SGLANG_SCHEDULER_EXIT_TIMEOUT
defaults to SGLANG_STAGED_SHUTDOWN_TIMEOUT + 60 s with the flag on.
"""

from __future__ import annotations

import contextlib
import fcntl
import gc
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Objects whose device tensors drop_device_tensors may release: the memory
# pools and their allocators, nothing that a still-running thread might own.
_POOL_MODULE_PREFIX = "sglang.srt.mem_cache"

# ---- plugin hooks ---------------------------------------------------------------------------

# name -> callable. Release hooks give back what a plugin holds on the GPU or in
# the driver (e.g. ccbridge's tier import and copy-engine channel); they run
# after the CUDA graphs are gone, before SGLang's host pools, communicators and
# tensors.
# Inventory hooks return a small JSON-able dict of what the plugin holds.
_RELEASE_HOOKS: Dict[str, Callable[[], Any]] = {}
_INVENTORY_HOOKS: Dict[str, Callable[[], Any]] = {}


def register_release_hook(name: str, fn: Callable[[], Any]) -> None:
    """Run `fn` in the ``plugin <name>`` step of a staged shutdown (same name: replaced).

    `fn` must be idempotent and must not need the other ranks. Whatever it
    returns is logged with the step.
    """
    _RELEASE_HOOKS[name] = fn


def register_inventory_hook(name: str, fn: Callable[[], Any]) -> None:
    """Add `fn()` (a JSON-able dict) to the rank's inventory under `name`."""
    _INVENTORY_HOOKS[name] = fn


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


# ---- inventory ------------------------------------------------------------------------------


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _cuda_graph_types() -> Tuple[type, ...]:
    graph = getattr(torch.cuda, "CUDAGraph", None)
    return (graph,) if isinstance(graph, type) else ()


def _mempool_types() -> Tuple[type, ...]:
    pool = getattr(torch.cuda, "MemPool", None)
    return (pool,) if isinstance(pool, type) else ()


def _expandable_segments() -> bool:
    for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        for item in os.environ.get(var, "").split(","):
            key, _, value = item.partition(":")
            if key.strip() == "expandable_segments":
                return value.strip().lower() == "true"
    return False


_TORCH_STATS = (
    "segment.all.current",
    "segment.large_pool.current",
    "segment.small_pool.current",
    "active.all.current",
    "num_device_alloc",
    "num_device_free",
    "num_alloc_retries",
)


def _torch_allocator() -> Dict[str, Any]:
    if not _cuda_available():
        return {}
    stats = torch.cuda.memory_stats()
    out: Dict[str, Any] = {k: stats[k] for k in _TORCH_STATS if k in stats}
    out["allocated_gib"] = round(stats.get("allocated_bytes.all.current", 0) / 2**30, 2)
    out["reserved_gib"] = round(stats.get("reserved_bytes.all.current", 0) / 2**30, 2)
    if _expandable_segments():
        # One cuMem handle per mapped page: 20 MiB (large pool) / 2 MiB (small).
        out["expandable_segments"] = True
        out["vmm_handles_est"] = int(
            stats.get("reserved_bytes.large_pool.current", 0) // (20 << 20)
            + stats.get("reserved_bytes.small_pool.current", 0) // (2 << 20)
        )
    host_stats = getattr(torch.cuda, "host_memory_stats", None)
    if host_stats is not None:
        try:
            h = host_stats()
            out["host"] = {
                k: h[k]
                for k in ("segment.current", "allocations.current", "num_host_alloc")
                if k in h
            }
            out["host"]["reserved_gib"] = round(
                h.get("reserved_bytes.current", 0) / 2**30, 2
            )
        except Exception:  # noqa: BLE001 - older torch: no host stats
            pass
    return out


def _heap() -> Dict[str, Any]:
    """One pass over the gc heap: graphs, memory pools, live CUDA tensors."""
    graph_types, pool_types = _cuda_graph_types(), _mempool_types()
    graphs = pools = tensors = 0
    storages: Dict[int, int] = {}
    # type(), not isinstance(): the latter reads __class__, which some module-level
    # proxies (torch.distributed.reduce_op) answer with a deprecation warning.
    for obj in gc.get_objects():
        kind = type(obj)
        if issubclass(kind, torch.Tensor):
            try:
                if obj.device.type != "cuda":
                    continue
                tensors += 1
                s = obj.untyped_storage()
                storages[s.data_ptr()] = s.nbytes()
            except Exception:  # noqa: BLE001 - storage-less subclasses
                continue
        elif graph_types and issubclass(kind, graph_types):
            graphs += 1
        elif pool_types and issubclass(kind, pool_types):
            pools += 1
    return {
        "cuda_graphs": graphs,
        "mempools": pools,
        "cuda_tensors": tensors,
        "cuda_storages": len(storages),
        "cuda_storage_gib": round(sum(storages.values()) / 2**30, 2),
    }


def _group_coordinators() -> List[Any]:
    """Every distinct GroupCoordinator parallel_state holds (WORLD, TP, EP, ...)."""
    ps = sys.modules.get("sglang.srt.distributed.parallel_state")
    if ps is None:
        return []
    cls = getattr(ps, "GroupCoordinator", None)
    seen, out = set(), []
    for value in list(vars(ps).values()):
        if cls is not None and isinstance(value, cls) and id(value) not in seen:
            seen.add(id(value))
            out.append(value)
    return out


_COMM_ATTRS = (
    "pynccl_comm",
    "ca_comm",
    "qr_comm",
    "pcie_ipc_comm",
    "pymscclpp_comm",
    "torch_symm_mem_comm",
)


def _communicators() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    groups = _group_coordinators()
    out["groups"] = len(groups)
    for attr in _COMM_ATTRS:
        live = [
            g
            for g in groups
            if getattr(g, attr, None) is not None
            and not getattr(getattr(g, attr), "disabled", False)
        ]
        if live:
            out[attr] = len(live)
    try:
        if torch.distributed.is_initialized():
            world = torch.distributed.distributed_c10d._world
            out["process_groups"] = len(world.pg_map)
    except Exception:  # noqa: BLE001
        pass
    try:
        from sglang.srt.runtime_context import get_resources

        buffers = get_resources().buffers
        out["buffers"] = sorted(k for k, v in buffers.items() if v is not None)
    except Exception:  # noqa: BLE001
        pass
    return out


def _driver_files() -> Dict[str, int]:
    """Open /dev/nvidia* files and their mappings: RM clients and mapped objects."""
    fds = maps = 0
    try:
        for fd in os.listdir("/proc/self/fd"):
            try:
                if os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/nvidia"):
                    fds += 1
            except OSError:
                pass
        with open("/proc/self/maps") as f:
            maps = sum(1 for line in f if "/dev/nvidia" in line)
    except OSError:
        pass
    return {"nvidia_fds": fds, "nvidia_mmaps": maps}


def inventory(heap: bool = True) -> Dict[str, Any]:
    """What this process holds that maps to RM objects. Every part is best effort."""
    out: Dict[str, Any] = {}
    parts: List[Tuple[str, Callable[[], Any]]] = [
        ("torch", _torch_allocator),
        ("comms", _communicators),
        ("driver", _driver_files),
    ]
    if heap:
        parts.insert(1, ("heap", _heap))
    parts += [(name, fn) for name, fn in list(_INVENTORY_HOOKS.items())]
    for name, fn in parts:
        try:
            out[name] = fn()
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": repr(e)}
    return out


def log_inventory(label: str, when: str) -> Dict[str, Any]:
    inv = inventory()
    logger.info(f"{label}: inventory at {when}: {json.dumps(inv, default=str)}")
    return inv


def log_startup_inventory(tp_rank: int) -> None:
    """At the end of scheduler start-up, TP0 logs its inventory (flag on only)."""
    if tp_rank != 0 or not envs.SGLANG_STAGED_SHUTDOWN.get():
        return
    if not envs.SGLANG_STAGED_SHUTDOWN_INVENTORY.get():
        return
    try:
        log_inventory(f"Staged shutdown TP{tp_rank}", "startup end")
    except Exception as e:  # noqa: BLE001 - never block serving on a log line
        logger.warning(f"startup inventory failed: {e!r}")


# ---- the release steps ----------------------------------------------------------------------


def destroy_cuda_graphs() -> str:
    """Reset every CUDA graph reachable from Python: decode, prefill, verify, draft.

    `CUDAGraph.reset` destroys the graph and its executable and gives its
    private memory pool back; the pool's segments are freed by the next
    empty_cache once no tensor lives in them (step ``tensors``).
    """
    types = _cuda_graph_types()
    if not types:
        return "no CUDA graph type"
    graphs = [o for o in gc.get_objects() if issubclass(type(o), types)]
    failed = 0
    for g in graphs:
        try:
            g.reset()
        except Exception:  # noqa: BLE001
            failed += 1
    return f"reset {len(graphs) - failed} graph(s)" + (
        f", {failed} failed" if failed else ""
    )


def _abort_pynccl(comm: Any) -> bool:
    """ncclCommAbort a PyNcclCommunicator: local, frees its cuMem buffers."""
    if (
        comm is None
        or getattr(comm, "disabled", True)
        or getattr(comm, "comm", None) is None
    ):
        return False
    import ctypes

    lib = comm.nccl.lib
    abort = getattr(lib, "ncclCommAbort", None)
    if abort is None:
        return False
    abort.restype = ctypes.c_int
    abort.argtypes = [ctypes.c_void_p]
    rc = abort(comm.comm)
    comm.disabled = True
    comm.comm = None
    if rc != 0:
        raise RuntimeError(f"ncclCommAbort returned {rc}")
    return True


def _close_deepep() -> bool:
    try:
        from sglang.srt.runtime_context import get_resources

        state = get_resources().buffers.get("deepep_ep_state")
    except Exception:  # noqa: BLE001
        return False
    buffer = getattr(state, "buffer", None)
    destroy = getattr(buffer, "destroy", None)
    if destroy is None:
        return False
    destroy()  # DeepEP refuses unless built with explicitly_destroy
    state.buffer = None
    return True


def close_communicators() -> str:
    """Give back every communicator's device memory, without a collective where possible.

    Custom all-reduce / IPC / symmetric-memory first (some of them barrier on
    the CPU group, which must still exist), the FlashInfer fusion workspaces,
    DeepEP, then the PyNccl communicators (ncclCommAbort) and last torch's
    process groups (abort: local).
    """
    closed: List[str] = []
    errors: List[str] = []

    def attempt(name: str, fn: Callable[[], Any]) -> None:
        try:
            if fn() is not False:
                closed.append(name)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e!r}")

    groups = _group_coordinators()
    for g in groups:
        gname = getattr(g, "unique_name", None) or "group"
        for attr, method in (
            ("ca_comm", "close"),
            ("qr_comm", "close"),
            ("pcie_ipc_comm", "destroy"),
            ("pymscclpp_comm", "destroy"),
        ):
            comm = getattr(g, attr, None)
            if comm is not None and hasattr(comm, method):
                attempt(f"{gname}.{attr}", getattr(comm, method))
                setattr(g, attr, None)
        if getattr(g, "torch_symm_mem_comm", None) is not None:
            # Its buffer is a symmetric-memory tensor: freed with the tensors.
            g.torch_symm_mem_comm = None
    try:
        from sglang.srt.layers.flashinfer_comm_fusion import (
            cleanup_flashinfer_workspace,
        )
    except Exception:  # noqa: BLE001
        cleanup_flashinfer_workspace = None
    if cleanup_flashinfer_workspace is not None:
        attempt("flashinfer workspace", cleanup_flashinfer_workspace)
    attempt("deepep", _close_deepep)
    for g in groups:
        gname = getattr(g, "unique_name", None) or "group"
        attempt(f"{gname}.pynccl", lambda g=g: _abort_pynccl(g.pynccl_comm))
    from sglang.srt.distributed.parallel_state import abort_distributed_environment

    attempt("process groups", abort_distributed_environment)
    if errors:
        logger.warning(f"communicator teardown errors: {errors}")
    return f"closed {len(closed)}: {', '.join(closed)}"


def _unbind(t: torch.Tensor) -> bool:
    """Point `t` at an empty storage on its device; its old storage loses a reference."""
    for mode in (torch.no_grad, torch.inference_mode):
        try:
            with mode():
                t.set_()
            return True
        except Exception:  # noqa: BLE001 - try the next way
            continue
    try:
        t.data = torch.empty(0, dtype=t.dtype, device=t.device)
        return True
    except Exception:  # noqa: BLE001
        return False


def _is_pinned(t: torch.Tensor) -> bool:
    try:
        return t.device.type == "cpu" and t.is_pinned()
    except Exception:  # noqa: BLE001
        return False


def sweep_tensors(device_type: str = "cuda", pinned: bool = True) -> Dict[str, int]:
    """Unbind every tensor on `device_type` (and every pinned host tensor) the gc can see.

    The process is leaving: whatever still references them gets empty tensors.
    Returns counts; the storages go back to the caching allocators once
    nothing else (C++-only owners) holds them.
    """
    unbound = pinned_unbound = failed = 0
    for obj in gc.get_objects():
        if not issubclass(type(obj), torch.Tensor):
            continue
        try:
            on_device = obj.device.type == device_type
        except Exception:  # noqa: BLE001
            continue
        is_pinned = not on_device and pinned and _is_pinned(obj)
        if not (on_device or is_pinned):
            continue
        if _unbind(obj):
            if on_device:
                unbound += 1
            else:
                pinned_unbound += 1
        else:
            failed += 1
    return {"unbound": unbound, "pinned": pinned_unbound, "failed": failed}


def _empty_caches() -> None:
    gc.collect()
    if _cuda_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        host_empty = getattr(torch._C, "_host_emptyCache", None)
        if host_empty is not None:
            host_empty()


def release_tensors(scheduler: Any) -> str:
    before = _torch_allocator()
    dropped = drop_device_tensors(
        getattr(scheduler, "token_to_kv_pool_allocator", None),
        getattr(scheduler, "req_to_token_pool", None),
    )
    swept = sweep_tensors()
    _empty_caches()
    after = _torch_allocator()
    return (
        f"pools {dropped / 2**30:.1f} GiB, swept {swept}, reserved "
        f"{before.get('reserved_gib', 0)} -> {after.get('reserved_gib', 0)} GiB, "
        f"segments {before.get('segment.all.current', 0)} -> "
        f"{after.get('segment.all.current', 0)}, allocated now "
        f"{after.get('allocated_gib', 0)} GiB"
    )


def reset_cuda_context(device: int) -> str:
    """cuDevicePrimaryCtxReset(device): free the primary context's objects one by one."""
    import ctypes

    lib = ctypes.CDLL("libcuda.so.1")
    dev = ctypes.c_int()
    rc = lib.cuDeviceGet(ctypes.byref(dev), ctypes.c_int(device))
    if rc != 0:
        raise RuntimeError(f"cuDeviceGet({device}) returned {rc}")
    reset = getattr(lib, "cuDevicePrimaryCtxReset_v2", None) or getattr(
        lib, "cuDevicePrimaryCtxReset"
    )
    rc = reset(dev)
    if rc != 0:
        raise RuntimeError(f"cuDevicePrimaryCtxReset({device}) returned {rc}")
    return f"device {device}"


class _Steps:
    """Run named steps one after another, each bounded, and time them.

    Each step runs in a helper thread (on the rank's CUDA device) and is waited
    for at most `step_timeout`: a step stuck on a dead peer is left behind and
    the next one runs. Once `deadline` has passed, steps are skipped unless
    marked `always`. TP0 (`verbose`) logs one line per step; failures and
    timeouts are logged by every rank.
    """

    def __init__(
        self,
        label: str,
        verbose: bool,
        deadline: float,
        step_timeout: float,
        device: Optional[int],
    ):
        self.label, self.verbose = label, verbose
        self.deadline, self.step_timeout, self.device = deadline, step_timeout, device
        self.record: List[Tuple[str, str, float]] = []  # (name, status, seconds)

    def run(self, name: str, fn: Callable[[], Any], always: bool = False) -> None:
        now = time.monotonic()
        if not always and now >= self.deadline:
            self.record.append((name, "skipped", 0.0))
            logger.warning(f"{self.label}: {name} skipped (past the shutdown budget)")
            return
        # An `always` step (the context reset, which may have tens of thousands of
        # objects to free) gets whatever budget is left, and at least a step's.
        limit = (
            max(self.step_timeout, self.deadline - now)
            if always
            else min(self.step_timeout, self.deadline - now)
        )
        box: Dict[str, Any] = {}

        def body() -> None:
            try:
                if self.device is not None:
                    torch.cuda.set_device(self.device)
                box["result"] = fn()
            except BaseException as e:  # noqa: BLE001 - best effort, exit reclaims
                box["error"] = e

        thread = threading.Thread(
            target=body, name=f"staged-shutdown-{name}", daemon=True
        )
        thread.start()
        thread.join(max(limit, 0.0))
        elapsed = time.monotonic() - now
        if thread.is_alive():
            self.record.append((name, "timeout", elapsed))
            logger.warning(
                f"{self.label}: {name} still running after {elapsed:.1f}s; moving on"
            )
        elif "error" in box:
            self.record.append((name, "failed", elapsed))
            logger.warning(
                f"{self.label}: {name} failed after {elapsed:.2f}s: {box['error']!r}"
            )
        else:
            self.record.append((name, "ok", elapsed))
            result = box.get("result")
            detail = f" ({result})" if isinstance(result, str) and result else ""
            if self.verbose:
                logger.info(f"{self.label}: {name} {elapsed:.2f}s{detail}")
            else:
                logger.debug(f"{self.label}: {name} {elapsed:.2f}s{detail}")


_released = threading.Event()


def _current_device() -> Optional[int]:
    return torch.cuda.current_device() if _cuda_available() else None


def _exit_now(code: int) -> None:
    for handler in logging.getLogger().handlers:
        with contextlib.suppress(Exception):
            handler.flush()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    os._exit(code)


def staged_release(
    scheduler: Any, tp_rank: int, exit_process: Optional[bool] = None
) -> List[Tuple[str, str, float]]:
    """The graceful-exit release of one scheduler rank (see the module doc).

    Idempotent: a second call returns at once. Leaves with os._exit(0) after a
    context reset (`exit_process` None), since nothing may touch CUDA after it.
    Returns the step record (name, status, seconds) when it does not exit.
    """
    if _released.is_set():
        return []
    _released.set()
    label = f"Staged shutdown TP{tp_rank}"
    started = time.monotonic()
    device = _current_device()
    steps = _Steps(
        label,
        verbose=tp_rank == 0,
        deadline=started + envs.SGLANG_STAGED_SHUTDOWN_TIMEOUT.get(),
        step_timeout=envs.SGLANG_STAGED_SHUTDOWN_STEP_TIMEOUT.get(),
        device=device,
    )
    if tp_rank == 0 and envs.SGLANG_STAGED_SHUTDOWN_INVENTORY.get():
        steps.run("inventory", lambda: log_inventory(label, "shutdown start") and None)
    if device is not None:
        steps.run("synchronize", torch.cuda.synchronize)

    # With the peers: neither may wait on a rank that has already gone further.
    tree_cache = getattr(scheduler, "tree_cache", None)
    if hasattr(tree_cache, "shutdown"):
        steps.run("storage detach", tree_cache.shutdown)

    def consensus() -> None:
        from sglang.srt.utils import rank_consensus_checker

        rank_consensus_checker.shutdown()

    steps.run("rank consensus", consensus)

    with teardown_turn(
        envs.SGLANG_STAGED_SHUTDOWN_LOCK.get(),
        envs.SGLANG_STAGED_SHUTDOWN_LOCK_TIMEOUT.get(),
        label,
    ):
        steps.run("CUDA graphs", destroy_cuda_graphs)
        # Plugins first: an external host pool's own release (ccbridge's) settles
        # its copy engine and closes its grant; SGLang's then finds nothing open.
        for name, hook in list(_RELEASE_HOOKS.items()):
            steps.run(f"plugin {name}", hook)
        steps.run("host resources", scheduler.release_host_resources)
        steps.run("communicators", close_communicators)
        steps.run("tensors", lambda: release_tensors(scheduler))
        reset = device is not None and envs.SGLANG_STAGED_SHUTDOWN_CTX_RESET.get()
        if reset:
            steps.run("context reset", lambda: reset_cuda_context(device), always=True)

    summary = ", ".join(f"{n} {s} {t:.1f}s" for n, s, t in steps.record)
    logger.info(f"{label}: done in {time.monotonic() - started:.1f}s [{summary}]")
    if exit_process is None:
        exit_process = reset
    if exit_process:
        _exit_now(0)
    return steps.record


def scheduler_exit_timeout() -> float:
    """How long the tokenizer side waits for the schedulers after ShutdownReq.

    SGLANG_SCHEDULER_EXIT_TIMEOUT when set; otherwise 15 s, or with
    SGLANG_STAGED_SHUTDOWN the whole bound of `staged_release` plus a margin,
    so that a releasing rank is not SIGKILLed (which would leave everything it
    still holds to the driver's single exit-time hold).
    """
    if (
        envs.SGLANG_SCHEDULER_EXIT_TIMEOUT.is_set()
        or not envs.SGLANG_STAGED_SHUTDOWN.get()
    ):
        return envs.SGLANG_SCHEDULER_EXIT_TIMEOUT.get()
    return (
        envs.SGLANG_STAGED_SHUTDOWN_TIMEOUT.get()
        + envs.SGLANG_STAGED_SHUTDOWN_STEP_TIMEOUT.get()
        + 20.0
    )
