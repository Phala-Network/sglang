"""Source-integrated adapter, converted from dsv41_adaptive_chunk.py.
No import hooks or external runtime code paths.
"""

import math


import os


import sys


import traceback


ENV_ENABLE = "DSV41_ADAPTIVE_CHUNK"


ENV_BUDGET = "DSV41_ADAPTIVE_CHUNK_BUDGET_BYTES"


ENV_MARGIN = "DSV41_ADAPTIVE_CHUNK_MARGIN"


ENV_MIN = "DSV41_ADAPTIVE_CHUNK_MIN"


ENV_BYTES = "DSV41_ADAPTIVE_CHUNK_BYTES"


_TARGET = "sglang.srt.managers.scheduler"


_PREFIX = "[dsv41-patches]"


DEFAULT_BYTES_PER_TOKEN_PAIR = 8.74


DEFAULT_MARGIN_FRACTION = 0.15


DEFAULT_MIN_CHUNK = 2048


BUDGET_QUANTUM_BYTES = 256 * 1024 * 1024


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "").strip().lower() in ("1", "true", "yes", "on")


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def chunk_for_history(
    history_len,
    *,
    budget_bytes,
    base_chunk_size,
    page_size,
    min_chunk=DEFAULT_MIN_CHUNK,
    bytes_per_token_pair=DEFAULT_BYTES_PER_TOKEN_PAIR,
):
    """Largest chunk whose prefill peak fits ``budget_bytes``, clamped and aligned.

    Solves ``coeff * x * (history_len + x) <= budget_bytes`` for ``x``, then
    clamps to ``[min_chunk, base_chunk_size]`` and floors to a multiple of
    ``page_size``.  Pure arithmetic: no torch, no sglang, no I/O.
    """
    base = int(base_chunk_size)
    if base <= 0:
        return base
    align = max(int(page_size), 1)
    floor = min(max(int(min_chunk), 1), base)
    history = max(int(history_len), 0)
    coeff = float(bytes_per_token_pair)
    budget = float(budget_bytes)

    if coeff <= 0 or budget <= 0 or not math.isfinite(budget):
        return base

    # coeff*x^2 + coeff*history*x - budget = 0, positive root.
    root = (-history + math.sqrt(history * history + 4.0 * budget / coeff)) / 2.0
    if not math.isfinite(root) or root <= 0:
        allowed = floor
    else:
        allowed = min(int(root), base)

    aligned = (allowed // align) * align
    if aligned < floor:
        # Never go below the floor even when the budget says we should: a chunk
        # that small is a scheduling problem, not a memory one, and the caller
        # would rather see an OOM it can diagnose than livelock on 256-token
        # chunks.  The floor is itself aligned up to the page size.
        aligned = ((floor + align - 1) // align) * align
    return min(aligned, base)


def measure_budget_bytes(margin_fraction=None):
    """Free GPU memory after graph capture, minus a margin, quantised.

    Reusable headroom is the driver's free memory plus the caching allocator's
    cached-but-unallocated blocks, which a later prefill can reuse without
    asking the driver.  Returns ``None`` if torch cannot answer.
    """
    if margin_fraction is None:
        margin_fraction = _env_float(ENV_MARGIN, DEFAULT_MARGIN_FRACTION)
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        cached_free = max(
            0, int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated())
        )
        raw = int(free) + cached_free
    except Exception:
        return None
    usable = int(raw * (1.0 - max(0.0, min(0.9, float(margin_fraction)))))
    return (usable // BUDGET_QUANTUM_BYTES) * BUDGET_QUANTUM_BYTES


def _agreed_budget_bytes():
    """One budget for the whole TP/PP world, or ``None`` to stay on the static size."""
    pinned = _env_int(ENV_BUDGET, 0)
    if pinned > 0:
        return pinned

    local = measure_budget_bytes()
    try:
        from sglang.srt.distributed.parallel_state import get_world_group
        from sglang.srt.utils import broadcast_pyobj

        world = get_world_group()
        local = broadcast_pyobj([local], world.rank, world.cpu_group, src=0)[0]
    except Exception:
        # No world group (single process, or an upstream rename): the quantised
        # local reading is what every rank would have computed anyway.
        pass
    if not local or local <= 0:
        return None
    return int(local)


class AdaptiveChunkSizer:
    """Duck-types ``DynamicChunkSizer`` for ``scheduler.py:3678``."""

    def __init__(
        self,
        *,
        budget_bytes,
        base_chunk_size,
        page_size,
        min_chunk=DEFAULT_MIN_CHUNK,
        bytes_per_token_pair=DEFAULT_BYTES_PER_TOKEN_PAIR,
    ):
        self.budget_bytes = int(budget_bytes)
        self.base_chunk_size = int(base_chunk_size)
        self.page_size = int(page_size)
        self.min_chunk = int(min_chunk)
        self.bytes_per_token_pair = float(bytes_per_token_pair)

    @classmethod
    def build(cls, scheduler):
        """Construct from a live ``Scheduler``, or ``None`` to leave it alone."""
        if not enabled():
            return None
        base = getattr(scheduler, "chunked_prefill_size", None)
        if not base or base <= 0:
            return None  # chunked prefill disabled; nothing to size
        budget = _agreed_budget_bytes()
        if not budget:
            return None
        page_size = int(getattr(scheduler, "page_size", 1) or 1)
        min_chunk = max(1, _env_int(ENV_MIN, DEFAULT_MIN_CHUNK))
        coeff = _env_float(ENV_BYTES, DEFAULT_BYTES_PER_TOKEN_PAIR)
        sizer = cls(
            budget_bytes=budget,
            base_chunk_size=int(base),
            page_size=page_size,
            min_chunk=min_chunk,
            bytes_per_token_pair=coeff,
        )
        print(
            f"{_PREFIX} adaptive chunk: budget={budget / (1 << 30):.1f} GiB "
            f"base={base} page={page_size} min={min_chunk} coeff={coeff} "
            f"(holds base to history_len={sizer.history_len_holding_base():,})",
            file=sys.stderr,
        )
        return sizer

    def history_len_holding_base(self):
        """Largest history at which the configured chunk still fits (for logging)."""
        b, c, x = float(self.budget_bytes), self.bytes_per_token_pair, self.base_chunk_size
        if c <= 0 or x <= 0:
            return 0
        return max(0, int(b / (c * x) - x))

    def predict(self, history_len):
        """``None`` keeps the static size; otherwise the chunk for this step."""
        try:
            size = chunk_for_history(
                history_len,
                budget_bytes=self.budget_bytes,
                base_chunk_size=self.base_chunk_size,
                page_size=self.page_size,
                min_chunk=self.min_chunk,
                bytes_per_token_pair=self.bytes_per_token_pair,
            )
        except Exception:
            return None
        if size >= self.base_chunk_size:
            return None  # unchanged: stay on the engine's normal path
        return size


def apply(module) -> None:
    """Install the sizer in the slot ``maybe_init_dynamic_chunk_sizer`` leaves empty."""
    scheduler_cls = getattr(module, "Scheduler", None)
    if scheduler_cls is None:
        raise AttributeError("sglang.srt.managers.scheduler has no Scheduler")
    original = getattr(scheduler_cls, "maybe_init_dynamic_chunk_sizer", None)
    if original is None:
        raise AttributeError("Scheduler has no maybe_init_dynamic_chunk_sizer")
    if getattr(original, "_dsv41_adaptive_chunk", False):
        return  # idempotent: already wrapped

    def maybe_init_dynamic_chunk_sizer(self):
        original(self)
        if getattr(self, "dynamic_chunk_sizer", None) is not None:
            return  # upstream built a PP sizer; do not displace it
        try:
            sizer = AdaptiveChunkSizer.build(self)
        except Exception:
            print(
                f"{_PREFIX} adaptive chunk: build FAILED, staying on the static "
                "chunk size",
                file=sys.stderr,
            )
            traceback.print_exc()
            return
        if sizer is not None:
            self.dynamic_chunk_sizer = sizer

    maybe_init_dynamic_chunk_sizer._dsv41_adaptive_chunk = True
    maybe_init_dynamic_chunk_sizer.__name__ = "maybe_init_dynamic_chunk_sizer"
    # Keep the original reachable so an auditor can still read upstream's body
    # (and its pp gate) after the wrapper is installed.
    maybe_init_dynamic_chunk_sizer.__wrapped__ = original
    scheduler_cls.maybe_init_dynamic_chunk_sizer = maybe_init_dynamic_chunk_sizer
