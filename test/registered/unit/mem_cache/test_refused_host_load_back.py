"""CPU-only unit tests for a host load-back that the host tier refused for good."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    TreePathNodeInfo,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

# root - 1 (device, Mamba state) - 2 (host) - 3 (host, Mamba state on host only)
#                                        \- 4 (host)
ROOT, DEVICE, S, A, SIBLING = 0, 1, 2, 3, 4
PARENT = {DEVICE: ROOT, S: DEVICE, A: S, SIBLING: S}
DEVICE_MAMBA = torch.tensor([5])


class _TreeCore:
    def __init__(self):
        self.match = None
        self.removed = set()

    def match_prefix(self, params):
        return self.match

    def drive_host_eviction(self, component_type, num_tokens):
        self.removed.add(A)
        return SimpleNamespace(tracker={component_type: num_tokens})

    def collect_path_info(self, end_node_ids):
        paths = []
        for node_id in end_node_ids:
            if node_id in self.removed:
                raise KeyError(node_id)
            path = []
            while node_id != ROOT:
                path.append(TreePathNodeInfo(node_id, PARENT[node_id], 0, 1, None, 0))
                node_id = PARENT[node_id]
            paths.append(tuple(reversed(path)))
        return tuple(paths)

    def get_component_device_value(self, node_id, component_type):
        return DEVICE_MAMBA if node_id == DEVICE else None

    def build_load_back_spec(self, node_id, req=None):
        kv = PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.arange(8),
            nodes_to_load=[S, node_id],
        )
        mamba = PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([9]))
        return kv, {ComponentType.MAMBA: [mamba]}


def _host_hit(best_match_node, device_node=DEVICE, full_kv_hit_length=192):
    return MatchResult(
        device_prefix_len=64,
        last_device_node=device_node,
        last_host_node=best_match_node,
        best_match_node=best_match_node,
        host_hit_length=128,
        mamba_host_hit_length=1,
        mamba_branching_seqlen=None,
        full_kv_hit_length=full_kv_hit_length,
        cache_actions=[],
    )


def _cache():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = _TreeCore()
    cache.refused_host_nodes = set()
    cache.disable = False
    cache.linker = None
    cache.load_back_threshold = 1
    cache.token_to_kv_pool_allocator = SimpleNamespace()
    cache._component_available_size = lambda component_type: 1 << 20
    cache._build_sidecar_transfers = lambda *args: []
    cache.dec_lock_ref = mock.Mock()
    cache.dec_host_lock_ref = mock.Mock()
    cache.cache_controller = HybridCacheController.__new__(HybridCacheController)
    cache.host_memory_mode = "cache"
    cache._apply_eviction_result = mock.Mock()

    mamba = MambaComponent.__new__(MambaComponent)
    mamba.cache = cache
    mamba.tree_core = cache.tree_core
    mamba.component_type = ComponentType.MAMBA
    mamba.mamba_checkpoint_grid = 64
    cache._components_tuple = (mamba,)
    cache.components = {ComponentType.MAMBA: mamba}
    return cache


def _refuse(cache, refused=True):
    """The host tier's ``load``: None, refused for good or for lack of room."""

    def load(**kwargs):
        cache.cache_controller.load_refused = refused
        return None

    cache.cache_controller.load = load
    return cache._load_back_transfers(
        node_id=A,
        mem_quota=None,
        req=None,
        result=SimpleNamespace(delta=0),
        ancestor_lock_params=None,
        host_anchor_params=None,
    )


def _match(cache, best_match_node, **hit):
    cache.tree_core.match = _host_hit(best_match_node, **hit)
    req = SimpleNamespace(kv=SimpleNamespace(holds_mamba=True))
    req.kv.mamba_cow_src_index = None
    req.kv.mamba_needs_clear = True
    result = cache._match_tree(MatchPrefixParams(key=None, cow_mamba=True, req=req))
    return result, req


class TestRefusedHostLoadBack(unittest.TestCase):
    def test_a_refused_load_marks_the_nodes_it_would_have_read(self):
        cache = _cache()
        with self.assertLogs("sglang.srt.mem_cache.unified_radix_cache", "WARNING"):
            self.assertFalse(_refuse(cache))
        self.assertEqual(cache.refused_host_nodes, {S, A})
        cache.dec_host_lock_ref.assert_called_once()

    def test_a_load_without_room_marks_nothing(self):
        cache = _cache()
        self.assertFalse(_refuse(cache, refused=False))
        self.assertEqual(cache.refused_host_nodes, set())

    def test_a_match_through_a_refused_node_ends_at_the_device_prefix(self):
        cache = _cache()
        with self.assertLogs("sglang.srt.mem_cache.unified_radix_cache", "WARNING"):
            _refuse(cache)
        # The refused anchor, a node above it, and a branch below the refused part.
        for best in (A, S, SIBLING):
            result, req = _match(cache, best)
            self.assertEqual(result.best_match_node, DEVICE)
            self.assertEqual(result.last_host_node, DEVICE)
            self.assertEqual(result.last_device_node, DEVICE)
            self.assertEqual(result.device_prefix_len, 64)
            self.assertEqual(
                (
                    result.host_hit_length,
                    result.swa_host_hit_length,
                    result.mamba_host_hit_length,
                ),
                (0, 0, 0),
            )
            self.assertTrue(result.host_hit_refused)
            # The end of the FULL hit is checkpointed as the recompute passes it.
            self.assertEqual(result.mamba_branching_seqlen, 192)
            # The recompute starts from the device prefix's own Mamba state
            # (K3 S8b: a cleared state here answered the wrong context).
            self.assertIs(req.kv.mamba_cow_src_index, DEVICE_MAMBA)
            self.assertFalse(req.kv.mamba_needs_clear)
        result, _ = _match(cache, A, full_kv_hit_length=100)
        self.assertIsNone(
            result.mamba_branching_seqlen, "no grid point past the prefix"
        )

    def test_other_host_hits_are_kept(self):
        cache = _cache()
        cache.refused_host_nodes = {SIBLING}
        result, req = _match(cache, A)
        self.assertEqual(result, _host_hit(A))
        self.assertFalse(result.host_hit_refused)
        self.assertIsNone(req.kv.mamba_cow_src_index)

    def test_a_refused_node_the_device_holds_again_is_unmarked(self):
        # Recomputed, and possibly re-sealed from the device by the tier: a later
        # host hit through it may load it (else it is refused once more).
        cache = _cache()
        with self.assertLogs("sglang.srt.mem_cache.unified_radix_cache", "WARNING"):
            _refuse(cache)
        result, _ = _match(cache, A, device_node=S)
        self.assertEqual(cache.refused_host_nodes, {A})
        self.assertEqual(result.best_match_node, S, "A is still host-only and marked")
        result, _ = _match(cache, A, device_node=A)
        self.assertEqual(cache.refused_host_nodes, set())
        self.assertEqual(result, _host_hit(A, device_node=A))

    def test_host_eviction_forgets_the_marks_of_removed_nodes(self):
        cache = _cache()
        with self.assertLogs("sglang.srt.mem_cache.unified_radix_cache", "WARNING"):
            _refuse(cache)
        self.assertEqual(cache.evict_host(8), 8)
        self.assertEqual(cache.refused_host_nodes, {S})

    def test_the_controller_load_clears_a_previous_refusal(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        self.assertFalse(controller.load_refused)
        controller.load_refused = True
        controller.mem_pool_device_allocator = SimpleNamespace(alloc=lambda size: None)
        self.assertIsNone(controller.load(host_indices=torch.arange(4)))
        self.assertFalse(controller.load_refused)


if __name__ == "__main__":
    unittest.main()


class TestNoStorageLookupThroughRefusedNodes(unittest.TestCase):
    """An L3 copy cannot replace a refused node's host value (insert_host keeps it),
    so a match cut back for a refusal does not send the scheduler to L3."""

    def setUp(self):
        from sglang.srt.managers.scheduler import Scheduler

        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.tree_cache = mock.Mock(buffer_pipeline=None)
        self.scheduler.enable_hicache_storage = True
        self.req = SimpleNamespace(
            storage_prefetch_last_match_len=192,
            storage_prefetch_retry_attempts=0,
            prefix_len=64,
            host_hit_length=0,
            host_hit_refused=True,
            last_host_node=DEVICE,
            last_node=DEVICE,
            init_next_round_input=mock.Mock(),
        )

    def test_a_refused_shrink_is_not_a_device_hit_loss(self):
        self.assertFalse(self.scheduler._prefetch_after_device_hit_loss(self.req))
        self.assertEqual(self.req.storage_prefetch_last_match_len, 64)
        self.scheduler.tree_cache.prefetch_from_storage.assert_not_called()

    def test_no_lookup_past_a_refused_host_hit(self):
        with mock.patch("sglang.srt.managers.scheduler.get_memory"):
            self.assertIsNone(self.scheduler._prefetch_kvcache(self.req))
        self.scheduler.tree_cache.prefetch_from_storage.assert_not_called()
