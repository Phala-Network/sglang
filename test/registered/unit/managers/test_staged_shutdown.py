"""Staged shutdown helpers: the ranks' turns and the pool-tensor release.

A rank releases its GPU-side resources in its turn (a flock shared by the
instance's ranks), so the NVIDIA driver frees one rank's objects at a time
instead of four whole processes at exit; see staged_shutdown.py.
"""

import os
import tempfile
import threading
import time
import unittest

import torch

from sglang.srt.managers.staged_shutdown import drop_device_tensors, teardown_turn
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

POOL_MODULE = "sglang.srt.mem_cache.fake_pool"


class _SubPool:
    def __init__(self):
        self.buffers = [torch.zeros(4), torch.zeros(4)]
        self.by_layer = {0: torch.zeros(2)}
        self.mixed = [torch.zeros(1), 3]  # not all tensors: kept


class _Pool:
    def __init__(self):
        self.kv = torch.zeros(8)
        self.view = self.kv[:4]  # a view of the same storage
        self.sub = _SubPool()
        self.subs = [_SubPool()]
        self.size = 8


class _Elsewhere:  # not a memory-pool object: never walked
    def __init__(self):
        self.t = torch.zeros(16)


for cls in (_SubPool, _Pool):
    cls.__module__ = POOL_MODULE


class TestDropDeviceTensors(unittest.TestCase):
    def test_unbinds_pool_tensors_and_counts_distinct_storages(self):
        pool = _Pool()
        pool.other = _Elsewhere()
        n = drop_device_tensors(pool, None, device_type="cpu")
        self.assertIsNone(pool.kv)
        self.assertIsNone(pool.view)
        self.assertIsNone(pool.sub.buffers)
        self.assertIsNone(pool.sub.by_layer)
        self.assertIsNone(pool.subs[0].buffers)
        self.assertEqual(len(pool.sub.mixed), 2)
        self.assertEqual(pool.size, 8)
        self.assertIsNotNone(pool.other.t)
        # kv (+ its view, one storage) 32 B, per sub-pool 2x16 + 8 B.
        self.assertEqual(n, 32 + 2 * (2 * 16 + 8))

    def test_other_devices_untouched(self):
        pool = _Pool()
        self.assertEqual(drop_device_tensors(pool), 0)  # cuda: none here
        self.assertIsNotNone(pool.kv)

    def test_depth_bound(self):
        pool = _Pool()
        drop_device_tensors(pool, max_depth=0, device_type="cpu")
        self.assertIsNone(pool.kv)
        self.assertIsNotNone(pool.sub.buffers)


class TestTeardownTurn(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "turn.lock")

    def tearDown(self):
        self.dir.cleanup()

    def test_turns_do_not_overlap(self):
        inside, overlaps = [0], [0]
        guard = threading.Lock()

        def rank():
            with teardown_turn(self.path, 10, "test"):
                with guard:
                    inside[0] += 1
                    overlaps[0] = max(overlaps[0], inside[0])
                time.sleep(0.05)
                with guard:
                    inside[0] -= 1

        threads = [threading.Thread(target=rank) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(overlaps[0], 1)

    def test_timeout_runs_the_body_anyway(self):
        held = threading.Event()
        release = threading.Event()

        def holder():
            with teardown_turn(self.path, 10, "holder"):
                held.set()
                release.wait(5)

        t = threading.Thread(target=holder)
        t.start()
        held.wait(5)
        start = time.monotonic()
        with teardown_turn(self.path, 0.3, "waiter"):
            waited = time.monotonic() - start
        release.set()
        t.join()
        self.assertGreaterEqual(waited, 0.3)
        self.assertLess(waited, 3)

    def test_no_path_or_unopenable_path_runs_the_body(self):
        ran = []
        with teardown_turn("", 1, "test"):
            ran.append(1)
        with teardown_turn(os.path.join(self.dir.name, "no", "such", "dir"), 1, "test"):
            ran.append(2)
        self.assertEqual(ran, [1, 2])


if __name__ == "__main__":
    unittest.main()
