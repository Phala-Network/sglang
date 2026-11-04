"""With FlashMLA prefill token packing, stored KV rows must be finite: a packed group
reads the union of its tokens' top-k rows and weighs each token's unpicked rows by
zero, so a NaN row would reach the other requests in the group (0 * NaN)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Stored(Exception):
    pass


def _rows():
    k = torch.tensor(
        [[1.5, float("nan")], [float("inf"), -2.0]], dtype=torch.bfloat16
    )
    k_rope = torch.tensor(
        [[-float("inf"), 0.25], [0.5, float("nan")]], dtype=torch.bfloat16
    )
    return k, k_rope


class TestRowsToStore(CustomTestCase):
    def _call(self, packing, k, k_rope):
        backend = SimpleNamespace(flashmla_prefill_token_packing=packing)
        return DeepseekSparseAttnBackend._kv_rows_to_store(backend, k, k_rope)

    def test_packing_zeroes_only_non_finite(self):
        k, k_rope = self._call(True, *_rows())
        self.assertEqual(k.tolist(), [[1.5, 0.0], [0.0, -2.0]])
        self.assertEqual(k_rope.tolist(), [[0.0, 0.25], [0.5, 0.0]])
        self.assertEqual(k.dtype, torch.bfloat16)

    def test_finite_rows_keep_their_bits(self):
        k = torch.tensor([[1e-3, -3.25, 0.0, -0.0]], dtype=torch.bfloat16)
        stored, _ = self._call(True, k, None)
        self.assertTrue(
            torch.equal(
                stored.view(-1).to(torch.float32).signbit(),
                k.view(-1).to(torch.float32).signbit(),
            )
        )
        self.assertTrue(torch.equal(stored, k))

    def test_no_rope(self):
        _, k_rope = self._call(True, _rows()[0], None)
        self.assertIsNone(k_rope)

    def test_without_packing_rows_are_untouched(self):
        k, k_rope = _rows()
        stored_k, stored_rope = self._call(False, k, k_rope)
        self.assertIs(stored_k, k)
        self.assertIs(stored_rope, k_rope)


class TestWritePaths(CustomTestCase):
    """Both forwards hand the pool the finite rows."""

    def _backend(self):
        backend = MagicMock()
        backend.flashmla_prefill_token_packing = True
        backend.use_mha = False
        backend.dsa_prefill_impl = backend.dsa_decode_impl = "flashmla_kv"
        backend._resolve_kpool_tail_backend.return_value = "flashmla_kv"
        backend._kv_rows_to_store = lambda k, k_rope: (
            DeepseekSparseAttnBackend._kv_rows_to_store(backend, k, k_rope)
        )
        backend.token_to_kv_pool.set_mla_kv_buffer.side_effect = _Stored
        return backend

    def _check(self, forward):
        backend = self._backend()
        layer = SimpleNamespace(is_cross_attention=False, layer_id=0)
        batch = MagicMock()
        batch.forward_mode.is_target_verify.return_value = False
        batch.forward_mode.is_draft_extend_v2.return_value = False
        k, k_rope = _rows()
        with self.assertRaises(_Stored):
            forward(backend, k, k, k, layer, batch, k_rope=k_rope)
        (_, loc, stored, stored_rope), _ = (
            backend.token_to_kv_pool.set_mla_kv_buffer.call_args
        )
        self.assertIs(loc, batch.out_cache_loc)
        self.assertTrue(torch.isfinite(stored).all())
        self.assertTrue(torch.isfinite(stored_rope).all())

    def test_extend(self):
        self._check(DeepseekSparseAttnBackend.forward_extend)

    def test_decode(self):
        self._check(DeepseekSparseAttnBackend.forward_decode)


if __name__ == "__main__":
    unittest.main()
