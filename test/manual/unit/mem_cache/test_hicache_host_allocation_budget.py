"""CPU contracts for the configurable DSA ordinary-memory reserve."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host import allocation_budget

GIB = 1024**3


def test_host_memory_reserve_defaults_to_128_gib():
    with envs.SGLANG_HICACHE_HOST_MEMORY_RESERVE_GB.override(128):
        assert allocation_budget.host_memory_reserve_bytes() == 128 * GIB


def test_resource_snapshot_uses_configured_ordinary_reserve():
    with (
        envs.SGLANG_HICACHE_HOST_MEMORY_RESERVE_GB.override(96),
        patch.object(
            allocation_budget.psutil,
            "virtual_memory",
            return_value=SimpleNamespace(available=117 * GIB),
        ),
        patch.object(allocation_budget, "_cgroup_headroom", return_value=None),
    ):
        ordinary, huge = allocation_budget._resource_snapshot(0)

    assert ordinary == 21 * GIB
    assert huge == 0


def test_negative_host_memory_reserve_is_rejected():
    with envs.SGLANG_HICACHE_HOST_MEMORY_RESERVE_GB.override(-1):
        with pytest.raises(ValueError, match="must be non-negative"):
            allocation_budget.host_memory_reserve_bytes()
