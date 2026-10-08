"""CPU-only unit tests for dropping one node's host Mamba state."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from collections import defaultdict

import torch

from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedLRUList, UnifiedTreeNode

COMPONENTS = (ComponentType.FULL, ComponentType.MAMBA)


class _FakeTreeCore:
    tree_components = COMPONENTS
    enable_session_radix_cache = False
    _is_tracking_unbacked_tokens = False

    def __init__(self):
        self.root_node = UnifiedTreeNode(COMPONENTS)
        self.evictable_host_leaves = set()
        self.host_lru_lists = {
            ComponentType.MAMBA: UnifiedLRUList(
                ComponentType.MAMBA, COMPONENTS, use_host_ptr=True
            )
        }
        self.lru_lists = {
            ComponentType.MAMBA: UnifiedLRUList(ComponentType.MAMBA, COMPONENTS)
        }
        self.cascaded, self.leaves = [], []

    def _evict_component_and_detach_lru(self, node, component, *args, **kwargs):
        return UnifiedTreeCore._evict_component_and_detach_lru(
            self, node, component, *args, **kwargs
        )

    def _cascade_evict(self, node, component, tracker, **kwargs):
        self.cascaded.append(node)

    def _update_evictable_leaf_sets(self, node):
        pass

    def _evict_host_leaf(self, node, tracker, device_frees, host_frees, **kwargs):
        self.leaves.append(node)


def _mamba_with_host_state():
    core = _FakeTreeCore()
    component = object.__new__(MambaComponent)
    component.tree_core = core
    node = UnifiedTreeNode(COMPONENTS)
    node.parent = core.root_node
    core.root_node.children[0] = node
    node.component_data[ComponentType.FULL].value = torch.tensor([7])
    node.component_data[ComponentType.MAMBA].host_value = torch.tensor([3])
    core.host_lru_lists[ComponentType.MAMBA].insert_mru(node)
    return component, core, node


def _drop(component, node):
    tracker = defaultdict(int)
    device_frees, host_frees = defaultdict(list), defaultdict(list)
    component.drop_host_value(node, tracker, device_frees, host_frees, [])
    return tracker, host_frees


class TestMambaDropHostValue(unittest.TestCase):
    def test_drops_the_host_state_and_its_lru_entry(self):
        component, core, node = _mamba_with_host_state()
        tracker, host_frees = _drop(component, node)
        self.assertIsNone(node.component_data[ComponentType.MAMBA].host_value)
        self.assertEqual([t.tolist() for t in host_frees[ComponentType.MAMBA]], [[3]])
        self.assertEqual(tracker[ComponentType.MAMBA], 1)
        self.assertFalse(core.host_lru_lists[ComponentType.MAMBA].in_list(node))
        self.assertEqual(core.cascaded, [node])

    def test_a_host_leaf_goes_whole(self):
        component, core, node = _mamba_with_host_state()
        core.evictable_host_leaves.add(node)
        _drop(component, node)
        self.assertEqual(core.leaves, [node])

    def test_a_host_locked_or_absent_state_stays(self):
        component, core, node = _mamba_with_host_state()
        node.component_data[ComponentType.MAMBA].host_lock_ref = 1
        _, host_frees = _drop(component, node)
        self.assertIsNotNone(node.component_data[ComponentType.MAMBA].host_value)
        self.assertFalse(host_frees)

        component, core, node = _mamba_with_host_state()
        node.component_data[ComponentType.MAMBA].host_value = None
        _, host_frees = _drop(component, node)
        self.assertFalse(host_frees)
        self.assertEqual(core.cascaded, [])

    def test_a_state_still_on_device_stays(self):
        # Its D->H backup may still be in flight into the host slot.
        component, core, node = _mamba_with_host_state()
        node.component_data[ComponentType.MAMBA].value = torch.tensor([9])
        _, host_frees = _drop(component, node)
        self.assertIsNotNone(node.component_data[ComponentType.MAMBA].host_value)
        self.assertFalse(host_frees)


if __name__ == "__main__":
    unittest.main()
