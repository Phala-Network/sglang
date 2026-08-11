"""Regression coverage for final-batch FlashMLA KV metadata repair."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.dsa_backend import (
    DSAFlashMLAMetadata,
    DeepseekSparseAttnBackend,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _flashmla_metadata(rows: int) -> DSAFlashMLAMetadata:
    return DSAFlashMLAMetadata(
        flashmla_metadata=torch.empty((4, 2), dtype=torch.int32),
        num_splits=torch.zeros(rows + 1, dtype=torch.int32),
    )


class TestDsaFlashMlaKvContract(CustomTestCase):
    def setUp(self):
        self.backend = object.__new__(DeepseekSparseAttnBackend)
        self.backend._compute_flashmla_metadata = Mock(
            side_effect=lambda cache_seqlens, seq_len_q: _flashmla_metadata(
                cache_seqlens.shape[0]
            )
        )

    @staticmethod
    def _q(rows: int) -> torch.Tensor:
        return torch.empty((rows, 1, 4, 8), dtype=torch.bfloat16)

    @staticmethod
    def _indices(rows: int, *, padded_from: int = None) -> torch.Tensor:
        indices = torch.arange(rows * 2, dtype=torch.int32).view(rows, 1, 2)
        if padded_from is not None:
            indices[padded_from:] = -1
        return indices

    @staticmethod
    def _metadata(cache_rows: int, planned_rows: int = None):
        if planned_rows is None:
            planned_rows = cache_rows
        return SimpleNamespace(
            dsa_cache_seqlens_int32=torch.arange(
                1, cache_rows + 1, dtype=torch.int32
            ),
            flashmla_metadata=_flashmla_metadata(planned_rows),
            flashmla_kv_cache_seqlens=None,
        )

    def test_matching_extend_batch_does_not_recompute(self):
        metadata = self._metadata(cache_rows=2)

        cache_seqlens, flashmla_metadata = (
            self.backend._prepare_flashmla_kv_contract(
                q=self._q(2),
                indices=self._indices(2),
                metadata=metadata,
            )
        )

        self.assertIs(cache_seqlens, metadata.dsa_cache_seqlens_int32)
        self.assertIs(flashmla_metadata, metadata.flashmla_metadata)
        self.backend._compute_flashmla_metadata.assert_not_called()

    def test_eagle_target_prefill_repairs_legal_padding_once(self):
        metadata = self._metadata(cache_rows=1)
        q = self._q(2)
        indices = self._indices(2, padded_from=1)

        cache_seqlens, flashmla_metadata = (
            self.backend._prepare_flashmla_kv_contract(
                q=q,
                indices=indices,
                metadata=metadata,
            )
        )
        second_cache_seqlens, second_flashmla_metadata = (
            self.backend._prepare_flashmla_kv_contract(
                q=q,
                indices=indices,
                metadata=metadata,
            )
        )

        self.assertEqual(cache_seqlens.tolist(), [1, 0])
        self.assertEqual(flashmla_metadata.num_splits.shape[0], 3)
        self.assertIs(second_cache_seqlens, cache_seqlens)
        self.assertIs(second_flashmla_metadata, flashmla_metadata)
        self.backend._compute_flashmla_metadata.assert_called_once()

    def test_stale_num_splits_recomputes_from_final_batch(self):
        metadata = self._metadata(cache_rows=2, planned_rows=1)

        _, flashmla_metadata = self.backend._prepare_flashmla_kv_contract(
            q=self._q(2),
            indices=self._indices(2),
            metadata=metadata,
        )

        self.assertEqual(flashmla_metadata.num_splits.shape[0], 3)
        self.backend._compute_flashmla_metadata.assert_called_once()

    def test_long_cache_metadata_is_trimmed_to_consumed_rows(self):
        metadata = self._metadata(cache_rows=3)

        cache_seqlens, flashmla_metadata = (
            self.backend._prepare_flashmla_kv_contract(
                q=self._q(2),
                indices=self._indices(2),
                metadata=metadata,
            )
        )

        self.assertEqual(cache_seqlens.tolist(), [1, 2])
        self.assertEqual(flashmla_metadata.num_splits.shape[0], 3)
        self.backend._compute_flashmla_metadata.assert_called_once()

    def test_short_metadata_for_real_row_fails_with_all_shapes(self):
        metadata = self._metadata(cache_rows=1)

        with self.assertRaisesRegex(
            RuntimeError,
            "q=.*indices=.*cache_seqlens=.*num_splits=.*tile_scheduler_metadata=",
        ):
            self.backend._prepare_flashmla_kv_contract(
                q=self._q(2),
                indices=self._indices(2),
                metadata=metadata,
            )

        self.backend._compute_flashmla_metadata.assert_not_called()

    def test_q_and_indices_row_mismatch_fails_before_recompute(self):
        metadata = self._metadata(cache_rows=2)

        with self.assertRaisesRegex(RuntimeError, "q and indices rows differ"):
            self.backend._prepare_flashmla_kv_contract(
                q=self._q(2),
                indices=self._indices(1),
                metadata=metadata,
            )

        self.backend._compute_flashmla_metadata.assert_not_called()


if __name__ == "__main__":
    unittest.main()
