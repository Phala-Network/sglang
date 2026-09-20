"""CPU coverage for file-backend hybrid prefix intersection."""

import tempfile
import unittest
from pathlib import Path

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _file_backend(directory: str, metadata_enabled: bool) -> HiCacheFile:
    rank, size = 3, 8
    backend = HiCacheFile(
        HiCacheStorageConfig(
            tp_rank=rank,
            tp_size=size,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=True,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="moonshotai/Kimi-K3",
            extra_config={
                "enable_metadata_cache": metadata_enabled,
                "metadata_ttl": -1.0,
            },
        ),
        file_path=directory,
    )
    # R8's File-L3 contract keeps replicated MLA KV TP-independent while
    # Mamba/KDA state is rank-owned. Tests must use those real namespaces.
    assert backend._get_component_key("fixture", PoolName.KV).endswith(
        "_moonshotai-Kimi-K3"
    )
    assert backend._get_component_key("fixture", PoolName.MAMBA).endswith(
        f"_moonshotai-Kimi-K3_mamba_tp{rank}_{size}"
    )
    return backend


class TestHiCacheFilePrefix(CustomTestCase):
    def check_prefix(self, kv_pages, pools, expected, num_pages=3):
        # Exercise real file scans and metadata lookups without constructing
        # the unrelated eviction/controller machinery.
        for metadata_enabled in (False, True):
            for reverse in (False, True):
                with self.subTest(metadata=metadata_enabled, reverse=reverse):
                    with tempfile.TemporaryDirectory() as directory:
                        backend = _file_backend(directory, metadata_enabled)
                        keys = [f"page{i}" for i in range(1, num_pages + 1)]
                        for name, pages in [(PoolName.KV, kv_pages)] + [
                            (name, pages) for name, pages, _, _ in pools
                        ]:
                            for page in pages:
                                Path(
                                    backend._get_component_path(keys[page - 1], name)
                                ).touch()
                        transfers = [
                            PoolTransfer(
                                name=name, keys=keys[-window:], hit_policy=policy
                            )
                            for name, _, policy, window in pools
                        ]
                        if reverse:
                            transfers.reverse()
                        # Second query also exercises warm metadata-cache hits.
                        for _ in range(2):
                            result = backend.batch_exists_v2(keys, transfers)
                            self.assertEqual(
                                result.kv_hit_pages, max(expected, default=0)
                            )
                            # R8 has only the legacy dense-prefix result shape;
                            # model that shape explicitly so red is the illegal
                            # endpoint assertion, not a missing-attribute error.
                            actual_endpoints = getattr(
                                result,
                                "restorable_prefix_pages",
                                list(range(1, result.kv_hit_pages + 1)),
                            )
                            self.assertEqual(actual_endpoints, expected)

    def test_aligned_checkpoints(self):
        self.check_prefix(
            {1, 2, 3},
            [
                (PoolName.SWA, {3}, PoolHitPolicy.TRAILING_PAGES, 1),
                (PoolName.MAMBA, {3}, PoolHitPolicy.TRAILING_PAGES, 1),
            ],
            [3],
        )

    def test_disjoint_checkpoints(self):
        self.check_prefix(
            {1, 2, 3},
            [
                (PoolName.SWA, {3}, PoolHitPolicy.TRAILING_PAGES, 1),
                (PoolName.MAMBA, {2}, PoolHitPolicy.TRAILING_PAGES, 1),
            ],
            [],
        )

    def test_preserves_non_contiguous_stop_points(self):
        self.check_prefix(
            {1, 2, 3},
            [(PoolName.MAMBA, {1, 3}, PoolHitPolicy.TRAILING_PAGES, 1)],
            [1, 3],
        )

    def test_earlier_common_checkpoint(self):
        self.check_prefix(
            {1, 2, 3},
            [
                (PoolName.SWA, {1, 3}, PoolHitPolicy.TRAILING_PAGES, 1),
                (PoolName.MAMBA, {1, 2}, PoolHitPolicy.TRAILING_PAGES, 1),
            ],
            [1],
        )

    def test_all_pages_truncates_trailing_pool(self):
        self.check_prefix(
            {1, 2, 3},
            [
                (PoolName.INDEXER, {1, 2}, PoolHitPolicy.ALL_PAGES, 3),
                (PoolName.MAMBA, {1, 3}, PoolHitPolicy.TRAILING_PAGES, 1),
            ],
            [1],
        )

    def test_multi_page_window_with_holes(self):
        self.check_prefix(
            {1, 2, 3, 4, 5},
            [
                (PoolName.SWA, {1, 2, 4, 5}, PoolHitPolicy.TRAILING_PAGES, 2),
                (PoolName.MAMBA, {2, 4}, PoolHitPolicy.TRAILING_PAGES, 1),
            ],
            [2],
            num_pages=5,
        )

    def test_prefix_shorter_than_window(self):
        self.check_prefix(
            {1, 2},
            [(PoolName.SWA, {1, 2}, PoolHitPolicy.TRAILING_PAGES, 3)],
            [1, 2],
        )

    def test_kv_hole_limits_auxiliary_hits(self):
        self.check_prefix(
            {1, 3},
            [(PoolName.MAMBA, {3}, PoolHitPolicy.TRAILING_PAGES, 1)],
            [],
        )

    def test_kv_only_and_empty(self):
        self.check_prefix({1, 3}, [], [1])
        self.check_prefix(set(), [], [])
        self.check_prefix(set(), [], [], num_pages=0)


if __name__ == "__main__":
    unittest.main()
