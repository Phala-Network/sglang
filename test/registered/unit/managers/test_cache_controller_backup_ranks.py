"""Which TP ranks back up a rank-replicated (MLA / DeepSeek-V4) cache to storage.

SGLang backs such a cache up from TP rank 0 only (`backup_skip`), unless the
storage backend says it publishes from every rank (`backup_on_every_rank`), as
ccbridge does when each rank's host pools keep one share of the pages.
"""

import threading
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.cache_controller import HiCacheController, StorageOperation
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _attach(backend, *, tp_rank, controller_cls=HiCacheController):
    cc = controller_cls.__new__(controller_cls)
    cc.enable_storage = False
    cc.storage_host_pool = object()
    cc.mem_pool_host = SimpleNamespace(
        size=1024, logical_size=1024, layout_lease=nullcontext
    )
    cc.host_memory_mode = "default"
    cc.page_size = 2
    cc.storage_stop_event = threading.Event()
    cc.pp_group = None
    cc.pp_prefetch_command_group = None
    config = SimpleNamespace(
        is_mla_model=True,
        tp_rank=tp_rank,
        is_storage_writer=tp_rank == 0,
        extra_config={},
    )
    with (
        get_parallel().override(attn_dcp_size=1),
        patch.object(controller_cls, "_stop_storage_threads"),
        patch.object(controller_cls, "_generate_storage_config", return_value=config),
        patch.object(controller_cls, "_create_sync_groups", return_value=[]),
        patch.object(controller_cls, "_start_storage_threads"),
        patch(
            "sglang.srt.mem_cache.storage.StorageBackendFactory.create_backend",
            return_value=backend,
        ),
    ):
        cc.attach_storage_backend("dynamic", storage_backend_extra_config={})
    return cc


class _Backend:
    def __init__(self, **attributes):
        self.register_mem_pool_host = Mock()
        self.on_host_pools_registered = Mock()
        self.__dict__.update(attributes)


class TestBackupRanks(unittest.TestCase):
    def test_rank_replicated_cache_is_backed_up_from_rank_0_only(self):
        self.assertFalse(_attach(_Backend(), tp_rank=0).backup_skip)
        self.assertTrue(_attach(_Backend(), tp_rank=1).backup_skip)
        self.assertTrue(
            _attach(_Backend(backup_on_every_rank=False), tp_rank=1).backup_skip
        )
        # Only a literal True opts in; a mock's auto-attribute does not.
        self.assertTrue(_attach(Mock(), tp_rank=1).backup_skip)

    def test_a_backend_publishing_from_every_rank_backs_up_everywhere(self):
        backend = _Backend(backup_on_every_rank=True)
        self.assertFalse(_attach(backend, tp_rank=1).backup_skip)

    def test_every_rank_then_backs_up_the_kv_and_every_sidecar(self):
        backend = _Backend(backup_on_every_rank=True)
        backend.batch_set_v1 = Mock(
            side_effect=lambda keys, *a, **k: [True] * len(keys)
        )
        backend.batch_set_v2 = Mock(
            side_effect=lambda transfers, *a, **k: {
                t.name: [True] * len(t.keys) for t in transfers
            }
        )
        cc = _attach(backend, tp_rank=1, controller_cls=HybridCacheController)
        cc.page_set_func = cc._page_set_zero_copy
        cc.storage_backend_type = "dynamic"
        operation = StorageOperation(
            torch.arange(4), [1, 2, 3, 4], hash_value=["a", "b"]
        )
        operation.pool_transfers = [
            PoolTransfer(name=PoolName.SWA, keys=["a", "b"]),
        ]
        operation.pool_storage_result = Mock()
        with (
            patch.object(
                HybridCacheController, "_resolve_sidecar_kv_derived_pool_transfers"
            ),
            patch.object(
                HybridCacheController, "_resolve_sidecar_nonkv_derived_pool_transfers"
            ),
        ):
            cc._page_backup(operation)
        backend.batch_set_v1.assert_called_once()
        self.assertEqual(
            [t.name for t in backend.batch_set_v2.call_args.args[0]], [PoolName.SWA]
        )
        self.assertEqual(operation.completed_tokens, 4)


if __name__ == "__main__":
    unittest.main()
