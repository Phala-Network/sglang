"""The K3 fused all-reduce dispatch in unicast mode, on CPU.

The kernels need 8 GPUs (test/registered/kernels/ops/communication/
test_ar_fusion_unicast.py); this covers the Python decisions around them:
when the unicast build is taken, which build the shared registry hands the
other K3 kernels, how messages split between the push family and the regular
all-reduce fallback, and that both switches off leaves the multicast-only
behaviour in place.
"""

import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import all_reduce_residual
from sglang.srt.environ import envs
from sglang.srt.layers.communication import k3_ar_fusion
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MB = 1024 * 1024
WORLD = 8


class FakeCommV2:
    """Stands in for CustomAllReduceV2 (the state checks the type)."""

    def __init__(self, has_multicast: bool, max_push_size: int = 2 * MB):
        self.disabled = False
        self.has_multicast = has_multicast
        self.max_push_size = max_push_size
        self.world_size = WORLD
        self.rank = 0
        self.config = SimpleNamespace(num_push_blocks=148)
        self.obj = SimpleNamespace(world_size=WORLD)


class K3ArFusionUnicastTest(CustomTestCase):
    def setUp(self):
        # a fresh state per test: cache_once keeps its map in a closure
        k3_ar_fusion._get_state = cache_once(k3_ar_fusion._get_state.__wrapped__)
        k3_ar_fusion._UNICAST_ACTIVE = False
        all_reduce_residual._COMM_MAP.clear()
        all_reduce_residual._MULTICAST.clear()
        self.calls = []

    def tearDown(self):
        k3_ar_fusion._UNICAST_ACTIVE = False
        all_reduce_residual._COMM_MAP.clear()
        all_reduce_residual._MULTICAST.clear()

    def _context(self, comm, **env):
        stack = ExitStack()
        fake_module = SimpleNamespace(CustomAllReduceV2=FakeCommV2)
        stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.distributed.device_communicators."
                    "custom_all_reduce_v2": fake_module
                },
            )
        )
        parallel = SimpleNamespace(
            tp_size=WORLD,
            tp_group=SimpleNamespace(
                ca_comm=comm, cpu_group=SimpleNamespace(group_name="tp")
            ),
        )
        stack.enter_context(
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel)
        )
        stack.enter_context(envs.SGLANG_K3_AR_FUSION.override(True))
        for name, value in env.items():
            stack.enter_context(getattr(envs, name).override(value))
        return stack

    # --- which build -------------------------------------------------------

    def test_no_multicast_and_switch_off_keeps_fusion_off(self):
        with self._context(FakeCommV2(has_multicast=False)):
            self.assertFalse(k3_ar_fusion.enabled())
        self.assertEqual(all_reduce_residual._MULTICAST, {})

    def test_no_multicast_with_switch_takes_unicast_build(self):
        with self._context(
            FakeCommV2(has_multicast=False), SGLANG_K3_AR_FUSION_UNICAST=1
        ):
            self.assertTrue(k3_ar_fusion.enabled())
            self.assertFalse(k3_ar_fusion.is_multicast())
        self.assertFalse(all_reduce_residual.is_multicast(WORLD))

    def test_multicast_ignores_unicast_switch(self):
        with self._context(
            FakeCommV2(has_multicast=True), SGLANG_K3_AR_FUSION_UNICAST=1
        ):
            self.assertTrue(k3_ar_fusion.enabled())
            self.assertTrue(k3_ar_fusion.is_multicast())
        self.assertTrue(all_reduce_residual.is_multicast(WORLD))

    def test_force_takes_unicast_build_on_multicast_plane(self):
        with self._context(
            FakeCommV2(has_multicast=True), SGLANG_K3_AR_FUSION_FORCE_UNICAST=1
        ):
            self.assertTrue(k3_ar_fusion.enabled())
            self.assertFalse(k3_ar_fusion.is_multicast())

    def test_unregistered_world_size_defaults_to_multicast(self):
        self.assertTrue(all_reduce_residual.is_multicast(WORLD))

    def test_push_window(self):
        comm = FakeCommV2(has_multicast=False, max_push_size=2 * MB)
        self.assertEqual(
            k3_ar_fusion._push_max_bytes_for(comm, multicast=True), 512 * 1024
        )
        with envs.SGLANG_K3_AR_FUSION_UNICAST_PUSH_MAX_KB.override(1024):
            self.assertEqual(
                k3_ar_fusion._push_max_bytes_for(comm, multicast=False), 1 * MB
            )
        with envs.SGLANG_K3_AR_FUSION_UNICAST_PUSH_MAX_KB.override(8192):
            # never beyond the slot
            self.assertEqual(
                k3_ar_fusion._push_max_bytes_for(comm, multicast=False), 2 * MB
            )

    # --- dispatch ----------------------------------------------------------

    def _record_push(self, world_size, x, residual=None, *, num_blocks=None):
        self.calls.append(("push", x.numel() * 2, num_blocks))
        return x

    def _record_push_norm(self, world_size, x, weight, eps, *, num_norm_rows):
        self.calls.append(("push_norm", x.numel() * 2, num_norm_rows))
        return x

    def _fake_regular_all_reduce(self, x):
        self.calls.append(("regular", x.numel() * 2))
        return x * WORLD  # a new tensor, as CustomAllReduceV2's push algos return

    def _dispatch_context(self):
        stack = self._context(
            FakeCommV2(has_multicast=False), SGLANG_K3_AR_FUSION_UNICAST=1
        )
        stack.enter_context(
            patch.object(all_reduce_residual, "all_reduce_push_res", self._record_push)
        )
        stack.enter_context(
            patch.object(
                all_reduce_residual, "all_reduce_push_norm", self._record_push_norm
            )
        )
        stack.enter_context(
            patch(
                "sglang.srt.distributed.communication_op."
                "tensor_model_parallel_all_reduce",
                self._fake_regular_all_reduce,
            )
        )
        return stack

    def test_small_message_takes_unicast_push(self):
        with self._dispatch_context():
            self.assertTrue(k3_ar_fusion.enabled())
            x = torch.ones(8, 7168, dtype=torch.bfloat16)
            residual = torch.ones_like(x)
            self.assertIs(k3_ar_fusion.all_reduce(x, residual), x)
        self.assertEqual(self.calls, [("push", 8 * 7168 * 2, None)])

    def test_large_message_takes_regular_all_reduce_in_place(self):
        with self._dispatch_context():
            k3_ar_fusion.enabled()
            x = torch.full((64, 7168), 2.0, dtype=torch.bfloat16)  # 896 KB > 512 KB
            residual = torch.ones_like(x)
            out = k3_ar_fusion.all_reduce(x, residual)
        self.assertIs(out, x)
        self.assertEqual(self.calls, [("regular", 64 * 7168 * 2)])
        torch.testing.assert_close(x, torch.full_like(x, 2.0 * WORLD + 1.0))

    def test_low_sm_is_a_push_on_few_blocks(self):
        with self._dispatch_context():
            k3_ar_fusion.enabled()
            x = torch.ones(8, 7168, dtype=torch.bfloat16)
            k3_ar_fusion.all_reduce_low_sm(
                x, num_blocks=4, unroll=8
            )  # pull tuning ignored
            with envs.SGLANG_K3_AR_FUSION_UNICAST_LOW_SM_BLOCKS.override(3):
                k3_ar_fusion.all_reduce_low_sm(x)
            big = torch.ones(64, 7168, dtype=torch.bfloat16)
            k3_ar_fusion.all_reduce_low_sm(big)
        self.assertEqual(
            self.calls,
            [
                ("push", 8 * 7168 * 2, 8),
                ("push", 8 * 7168 * 2, 3),
                ("regular", 64 * 7168 * 2),
            ],
        )

    def test_norm_dispatch_and_fallback_numerics(self):
        weight = torch.linspace(0.5, 1.5, k3_ar_fusion.NORM_DIM, dtype=torch.bfloat16)
        with self._dispatch_context():
            k3_ar_fusion.enabled()
            small = torch.ones(3 * 8, k3_ar_fusion.NORM_DIM, dtype=torch.bfloat16)
            k3_ar_fusion.all_reduce_norm(small, weight, 1e-6, num_tokens=8)
            # 80 tokens x [latent | shared]: 1.6 MB, above the window
            rows = torch.randn(3 * 80, k3_ar_fusion.NORM_DIM).to(torch.bfloat16)
            x = rows.clone()
            k3_ar_fusion.all_reduce_norm(x, weight, 1e-6, num_tokens=80)
        self.assertEqual(
            self.calls[0], ("push_norm", 3 * 8 * k3_ar_fusion.NORM_DIM * 2, 8)
        )
        self.assertEqual(self.calls[1], ("regular", 3 * 80 * k3_ar_fusion.NORM_DIM * 2))
        reduced = (rows * WORLD).float()
        normed = reduced[:80]
        factor = torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + 1e-6)
        expect = reduced.clone()
        expect[:80] = normed * factor * weight.float()
        torch.testing.assert_close(x.float(), expect.to(torch.bfloat16).float())

    def test_fits_use_unicast_window(self):
        with self._dispatch_context():
            k3_ar_fusion.enabled()
            # [num_tokens, 3584] bf16: 73 tokens = 511 KB fits, 74 does not
            self.assertTrue(k3_ar_fusion.finalize_push_fits(73))
            self.assertFalse(k3_ar_fusion.finalize_push_fits(74))
            with envs.SGLANG_K3_AR_FUSION_UNICAST_PUSH_MAX_KB.override(1024):
                self.assertTrue(k3_ar_fusion.finalize_push_fits(74))

    def test_multicast_registration_rejects_push_grid(self):
        all_reduce_residual.register_comm(SimpleNamespace(world_size=WORLD))
        x = torch.ones(8, 7168, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            all_reduce_residual.all_reduce_push_res(WORLD, x, None, num_blocks=2)

    def test_registration_mode_is_sticky(self):
        obj = SimpleNamespace(world_size=WORLD)
        all_reduce_residual.register_comm(obj, multicast=False)
        all_reduce_residual.register_comm(obj, multicast=False)
        with self.assertRaises(AssertionError):
            all_reduce_residual.register_comm(obj, multicast=True)


if __name__ == "__main__":
    unittest.main()
