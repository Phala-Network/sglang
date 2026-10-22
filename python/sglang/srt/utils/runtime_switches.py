"""Environment switches that may be flipped while the server runs.

A reload script (SGLANG_DEBUG_HOT_RELOAD_SCRIPT, SIGUSR2) sets os.environ in
each scheduler process. The script runs at whatever loop iteration each process
happens to be in, but these switches steer decisions every rank of a scheduling
group must make identically (batch composition, prefetch termination, host
backups): ranks that disagree for even one iteration diverge and their
collectives hang. So readers go through `get()`, which returns the value the
group agreed on:

- each scheduler snapshots the switches from its environment at startup;
- once per loop iteration the rank that pulls requests (TP rank 0 of its
  group) compares its environment with the agreed values and, if they differ,
  sends a `RuntimeSwitchUpdate` on the same broadcast as its requests;
- every rank adopts the update at the same point of that request stream.

Only the request-pulling rank's environment matters after startup; setting the
variables on every rank keeps the processes consistent for anything else that
reads them. Outside a scheduler (tests, tools), `get()` reads the environment.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from sglang.srt.environ import EnvField, envs

logger = logging.getLogger(__name__)

SWITCHES: Tuple[EnvField, ...] = (
    envs.SGLANG_HICACHE_SWA_SPARSE_BACKUP,
    envs.SGLANG_HICACHE_PREFETCH_POLICY_OVERRIDE,
    envs.SGLANG_PREFILL_DECODE_INTERVAL_OVERRIDE,
    envs.SGLANG_CHUNKED_PREFILL_SHORT_RESERVE,
)
_NAMES = frozenset(field.name for field in SWITCHES)

PREFETCH_POLICIES = ("best_effort", "wait_complete", "timeout")

# Raw environment strings (None: unset) the scheduling group agreed on; None
# until a scheduler arms the sync.
_agreed: Optional[Dict[str, Optional[str]]] = None
_warned: set = set()


@dataclass(frozen=True)
class RuntimeSwitchUpdate:
    """Rides the request broadcast: the source rank's raw switch values."""

    values: Tuple[Tuple[str, Optional[str]], ...]


def _sample() -> Dict[str, Optional[str]]:
    return {name: os.environ.get(name) for name in sorted(_NAMES)}


def get(field: EnvField) -> Any:
    """The agreed value of a switch (the environment outside a scheduler)."""
    if _agreed is None or field.name not in _agreed:
        return field.get()
    return field.get_from(_agreed[field.name])


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def prefetch_policy(configured: str) -> str:
    """The storage prefetch stop policy now in effect."""
    value = (get(envs.SGLANG_HICACHE_PREFETCH_POLICY_OVERRIDE) or "").strip()
    if not value:
        return configured
    if value in PREFETCH_POLICIES:
        return value
    _warn_once(
        f"prefetch_policy={value!r}",
        "Ignoring SGLANG_HICACHE_PREFETCH_POLICY_OVERRIDE=%r: expected one of %s",
        value,
        list(PREFETCH_POLICIES),
    )
    return configured


def prefill_decode_interval(configured: int) -> int:
    """The prefill-decode interval now in effect."""
    value = (get(envs.SGLANG_PREFILL_DECODE_INTERVAL_OVERRIDE) or "").strip()
    if not value:
        return configured
    try:
        interval = int(value)
    except ValueError:
        interval = -1
    if interval >= 0:
        return interval
    _warn_once(
        f"prefill_decode_interval={value!r}",
        "Ignoring SGLANG_PREFILL_DECODE_INTERVAL_OVERRIDE=%r: expected an "
        "integer >= 0",
        value,
    )
    return configured


class RuntimeSwitchSync:
    """Per-scheduler half of the protocol in the module docstring."""

    def __init__(self) -> None:
        global _agreed
        _agreed = _sample()
        logger.info("Runtime switches at startup: %s", _format(_agreed))

    def poll(self) -> Optional[RuntimeSwitchUpdate]:
        """On the request-pulling rank: the update to broadcast, if any.

        Stateless: an update the receive path drops is sent again next
        iteration, and one sent is applied before the next poll.
        """
        current = _sample()
        if current == _agreed:
            return None
        return RuntimeSwitchUpdate(tuple(current.items()))

    def apply(self, update: RuntimeSwitchUpdate) -> None:
        global _agreed
        values = {name: raw for name, raw in update.values if name in _NAMES}
        agreed = _agreed or {}
        changed = {
            name: raw for name, raw in values.items() if agreed.get(name) != raw
        }
        _agreed = {**agreed, **values}
        if changed:
            logger.warning("Runtime switches changed: %s", _format(changed))


def _format(values: Dict[str, Optional[str]]) -> str:
    return ", ".join(
        f"{name}={'<unset>' if raw is None else repr(raw)}"
        for name, raw in sorted(values.items())
    )


def _reset_for_tests() -> None:
    global _agreed
    _agreed = None
    _warned.clear()
