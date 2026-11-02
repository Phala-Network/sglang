"""SGLANG_CHUNKED_PREFILL_CHECKPOINT_TOKENS: which chunked-prefill chunks are
checkpointed into the radix tree."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

K = 1024


def _checkpointed(chunks, *, protected=0, branching=None, every=32 * K):
    """Run consecutive chunks from `protected`; return the chunk ends checkpointed."""
    req = SimpleNamespace(
        prefix_len=protected,
        extend_end=protected,
        mamba_branching_seqlen=branching,
        kv=SimpleNamespace(cache_protected_len=protected),
    )
    ends = []
    with envs.SGLANG_CHUNKED_PREFILL_CHECKPOINT_TOKENS.override(every):
        for size in chunks:
            req.extend_end = req.prefix_len + size
            if not Scheduler._defer_chunk_checkpoint(req):
                ends.append(req.extend_end)
                req.kv.cache_protected_len = req.extend_end
            req.prefix_len = req.extend_end
    return ends


class TestChunkedPrefillCheckpointTokens(CustomTestCase):
    def test_off_checkpoints_every_chunk(self):
        self.assertEqual(_checkpointed([8 * K] * 3, every=0), [8 * K, 16 * K, 24 * K])

    def test_cold_prompt_checkpoints_each_multiple(self):
        # 18 chunks of 8K; the last chunk (the prompt end) is not stashed.
        self.assertEqual(_checkpointed([8 * K] * 17), [32 * K, 64 * K, 96 * K, 128 * K])

    def test_matched_prefix_is_the_baseline(self):
        # A 100K match: nothing until the chunk that crosses 128K.
        self.assertEqual(_checkpointed([8 * K] * 5, protected=100 * K), [132 * K])

    def test_branching_point_is_checkpointed(self):
        self.assertEqual(
            _checkpointed([8 * K] * 4, protected=64 * K, branching=75 * K),
            [80 * K, 96 * K],
        )


if __name__ == "__main__":
    unittest.main()
