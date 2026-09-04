"""Tests for deferred chunked-prefill aborts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeReq:
    """Minimal stand-in for Req: only the fields the abort paths touch."""

    def __init__(self, rid: str):
        self.rid = rid
        self.req_pool_idx = 1
        self.mamba_pool_idx = None
        self.to_finish = None
        self._finished = False
        self.inflight_middle_chunks = 0
        self.mm_embedding_validation_count = 0
        self.time_stats = Mock()

    def finished(self):
        return self._finished


def _make_scheduler(pending_req, *, chunked_req, running_reqs) -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    sched.chunked_req = chunked_req
    sched._pending_chunked_abort_req = pending_req
    sched.waiting_queue = []
    sched.dllm_config = None
    sched.grammar_manager = Mock()
    sched.disaggregation_mode = None
    sched.enable_hicache_storage = False
    sched.tree_cache = Mock()
    sched.ipc_channels = SimpleNamespace(send_to_tokenizer=Mock())
    sched.ps = SimpleNamespace(pp_size=1)
    sched.running_batch = SimpleNamespace(reqs=running_reqs)
    sched.last_batch = None
    return sched


class TestPendingChunkedAbortRace(CustomTestCase):
    def test_req_left_chunked_slot_is_aborted(self):
        req = _FakeReq("zombie_rid")
        sched = _make_scheduler(req, chunked_req=None, running_reqs=[req])

        with (
            patch("sglang.srt.managers.scheduler.prepare_abort") as prepare_abort,
            patch("sglang.srt.managers.scheduler.release_kv_cache") as release,
            patch("sglang.srt.managers.scheduler._make_abort_req", return_value=Mock()),
        ):
            sched.process_pending_chunked_abort()

        prepare_abort.assert_called_once()
        release.assert_called_once_with(req, sched.tree_cache, is_insert=False)
        sched.ipc_channels.send_to_tokenizer.send_output.assert_called_once()
        self.assertIsNone(sched._pending_chunked_abort_req)

    def test_finished_req_only_clears_marker(self):
        req = _FakeReq("done_rid")
        req._finished = True
        sched = _make_scheduler(req, chunked_req=None, running_reqs=[])

        sched.process_pending_chunked_abort()

        self.assertIsNone(req.to_finish)
        self.assertIsNone(sched._pending_chunked_abort_req)


if __name__ == "__main__":
    unittest.main(verbosity=2)
