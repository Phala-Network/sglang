"""Cancellation regressions adapted from upstream PR #35255 plus r4 controls."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from test_tokenizer_manager_rid_cleanup import (
    _make_generate_obj,
    _make_req_state,
    _make_tm_for_generate,
    _make_tokenizer_manager,
)

from sglang.srt.managers.io_struct import AbortReq, GenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.test_utils import CustomTestCase


class TestReleaseReqStatesOnFailure(CustomTestCase):
    """Direct tests for _release_req_states_on_failure."""

    def test_undelivered_single_is_dropped(self):
        tm = _make_tokenizer_manager(self)
        rid = "d_single"
        tm.rid_to_state[rid] = _make_req_state(rid)
        tm._release_req_states_on_failure([rid])
        self.assertNotIn(rid, tm.rid_to_state)

    def test_undelivered_batch_removes_all(self):
        tm = _make_tokenizer_manager(self)
        rids = ["d0", "d1", "d2"]
        for r in rids:
            tm.rid_to_state[r] = _make_req_state(r)
        tm._release_req_states_on_failure(rids)
        for r in rids:
            self.assertNotIn(r, tm.rid_to_state)

    def test_ignores_already_removed(self):
        """A rid that is no longer present must not raise."""
        tm = _make_tokenizer_manager(self)
        tm.rid_to_state["p1"] = _make_req_state("p1")
        tm._release_req_states_on_failure(["p1", "already_gone"])
        self.assertNotIn("p1", tm.rid_to_state)

    def test_dispatched_single_is_aborted_and_state_kept(self):
        tm = _make_tokenizer_manager(self)
        tm.server_args.tokenizer_worker_num = 1
        tm._dispatch_to_scheduler = Mock()
        tm.enable_metrics = True
        tm.metrics_collector = MagicMock()
        rid = "d_live"
        state = _make_req_state(rid)
        state.dispatched = True
        tm.rid_to_state[rid] = state
        tm._release_req_states_on_failure([rid])
        tm._release_req_states_on_failure([rid])

        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        self.assertEqual(
            [type(m) for m in sent], [AbortReq], "expected exactly one AbortReq"
        )
        self.assertEqual(sent[0].rid, rid)
        self.assertIn(rid, tm.rid_to_state)
        self.assertTrue(state.abort_sent)
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_dispatched_batch_aborts_delivered_and_drops_rest(self):
        tm = _make_tokenizer_manager(self)
        tm.server_args.tokenizer_worker_num = 1
        tm._dispatch_to_scheduler = Mock()
        delivered, undelivered = "d_delivered", "d_undelivered"
        live = _make_req_state(delivered)
        live.dispatched = True
        tm.rid_to_state[delivered] = live
        tm.rid_to_state[undelivered] = _make_req_state(undelivered)
        tm._release_req_states_on_failure([delivered, undelivered])

        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        self.assertEqual([type(m) for m in sent], [AbortReq])
        self.assertEqual(sent[0].rid, delivered)
        self.assertIn(delivered, tm.rid_to_state)
        self.assertNotIn(undelivered, tm.rid_to_state)

    def test_abort_failure_does_not_stop_cleanup(self):
        tm = _make_tokenizer_manager(self)
        tm.server_args.tokenizer_worker_num = 1
        tm._dispatch_to_scheduler = Mock(side_effect=RuntimeError("send failed"))
        delivered, undelivered = "live", "pending"
        live = _make_req_state(delivered)
        live.dispatched = True
        tm.rid_to_state[delivered] = live
        tm.rid_to_state[undelivered] = _make_req_state(undelivered)

        with self.assertLogs(level="ERROR"):
            tm._release_req_states_on_failure([delivered, undelivered])

        self.assertIn(delivered, tm.rid_to_state)
        self.assertFalse(live.abort_sent)
        self.assertNotIn(undelivered, tm.rid_to_state)


class TestDisconnectAfterDispatchAbortsRequest(CustomTestCase):
    """Cancellation after dispatch must stop the scheduler request."""

    @patch(
        "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
        side_effect=lambda obj: obj,
    )
    def test_cancel_after_dispatch_sends_abort_and_keeps_state(self, _wrap_shm):
        tm = _make_tm_for_generate(self)
        tm.cuda_vmm_feature_transport = Mock()
        tm.cuda_vmm_feature_transport.prepare_for_dispatch = Mock(return_value=[])
        tm._dispatch_to_scheduler = Mock()
        rid = "disconnect_zombie"
        obj = _make_generate_obj(rid, is_single=True)
        obj.return_prompt_token_ids = False
        tokenized = MagicMock()
        tokenized.rid = rid
        tokenized.mm_inputs = None
        tm._tokenize_one_request = AsyncMock(return_value=tokenized)

        async def drive():
            task = asyncio.create_task(tm.generate_request(obj).__anext__())
            for _ in range(100):
                await asyncio.sleep(0)
                if tm._dispatch_to_scheduler.called:
                    break
            self.assertTrue(
                tm._dispatch_to_scheduler.called, "request never dispatched"
            )
            state = tm.rid_to_state.get(rid)
            self.assertIsNotNone(state)
            self.assertTrue(state.dispatched)

            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(drive())

        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        aborts = [m for m in sent if isinstance(m, AbortReq) and m.rid == rid]
        self.assertTrue(aborts, "disconnect must send an AbortReq to the scheduler")
        self.assertIn(rid, tm.rid_to_state)


class TestCancellationOwnership(CustomTestCase):
    def test_parent_placeholder_cleanup_keeps_dispatched_state(self):
        tm = _make_tm_for_generate(self)
        live = _make_req_state("actual")
        live.dispatched = True
        tm.rid_to_state = {"parent": _make_req_state("parent"), "actual": live}
        tm._discard_pending_req_states(
            SimpleNamespace(is_single=False, rid=["parent", "actual"])
        )
        self.assertEqual(set(tm.rid_to_state), {"actual"})

    def test_unknown_prefix_and_empty_abort_do_not_cancel_live_requests(self):
        tm = _make_tm_for_generate(self)
        tm.rid_to_state = {"batch_10": _make_req_state("batch_10")}
        tm._dispatch_to_scheduler = Mock()
        for rid in ("", "batch", "batch_1", "stale"):
            tm.abort_request(rid)
        tm._dispatch_to_scheduler.assert_not_called()

    def test_failed_abort_is_retryable_and_counted_once(self):
        tm = _make_tm_for_generate(self)
        tm.rid_to_state["live"] = _make_req_state("live")
        tm._dispatch_to_scheduler = Mock(side_effect=[RuntimeError("socket"), None])
        tm.enable_metrics = True
        tm.metrics_collector = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "socket"):
            tm.abort_request("live")
        self.assertFalse(tm.rid_to_state["live"].abort_sent)
        tm.abort_request("live")
        tm.abort_request("live")
        self.assertEqual(tm._dispatch_to_scheduler.call_count, 2)
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_generator_close_aborts_actual_generated_choices_only(self):
        tm = _make_tm_for_generate(self)
        tm._dispatch_to_scheduler = Mock()
        obj = _make_generate_obj(["parent_0", "parent_1"], is_single=False)

        async def batch(obj, request, request_rids):
            for rid in ("actual_0", "actual_1"):
                state = _make_req_state(rid)
                state.dispatched = True
                tm.rid_to_state[rid] = state
                request_rids.add(rid)
            yield {"text": "first"}
            await asyncio.Event().wait()

        tm._handle_batch_request = batch

        async def drive():
            gen = tm.generate_request(obj)
            await gen.__anext__()
            await gen.aclose()

        asyncio.run(drive())
        aborts = [c.args[0].rid for c in tm._dispatch_to_scheduler.call_args_list]
        self.assertCountEqual(aborts, ["actual_0", "actual_1"])
        self.assertEqual(set(tm.rid_to_state), {"actual_0", "actual_1"})

    def test_parallel_sampling_failure_cleans_generated_rid(self):
        tm = _make_tm_for_generate(self)
        obj = GenerateReqInput(text=["hello"], rid=["base"], sampling_params={"n": 2})
        tokenized = MagicMock()
        tokenized.mm_inputs = None
        tokenized.sampling_params = MagicMock()
        tm._tokenize_one_request = AsyncMock(return_value=tokenized)
        tm._send_one_request = Mock(side_effect=RuntimeError("dispatch failed"))

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
            asyncio.run(drive())
        self.assertFalse(tm.rid_to_state)

    def test_delayed_abort_does_not_cancel_completed_or_duplicate_requests(self):
        tm = _make_tm_for_generate(self)
        tm._dispatch_to_scheduler = Mock()
        live = _make_req_state("live")
        live.dispatched = True
        tm.rid_to_state["live"] = live
        obj = SimpleNamespace(is_single=False, rid=["live", "finished"])
        tm.abort_request("live")
        with patch(
            "sglang.srt.managers.tokenizer_manager.asyncio.sleep", new=AsyncMock()
        ):
            asyncio.run(tm.create_abort_task(obj)())
        self.assertEqual(tm._dispatch_to_scheduler.call_count, 1)


class TestPendingChunkedAbort(CustomTestCase):
    def make_scheduler(self, finished=False, holds_kv=True):
        req = SimpleNamespace(
            rid="chunk",
            finished=lambda: finished,
            kv=SimpleNamespace(holds_kv=holds_kv),
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.chunked_req = None
        scheduler._pending_chunked_abort_req = req
        scheduler.abort_request = Mock()
        return scheduler

    def test_transition_retries_abort_on_current_queue(self):
        scheduler = self.make_scheduler()
        scheduler.process_pending_chunked_abort()
        scheduler.abort_request.assert_called_once()
        self.assertEqual(scheduler.abort_request.call_args.args[0].rid, "chunk")
        self.assertIsNone(scheduler._pending_chunked_abort_req)

    def test_finished_or_freed_only_clears_marker(self):
        for finished, holds in [(True, True), (False, False)]:
            with self.subTest(finished=finished, holds=holds):
                scheduler = self.make_scheduler(finished, holds)
                scheduler.process_pending_chunked_abort()
                scheduler.abort_request.assert_not_called()
                self.assertIsNone(scheduler._pending_chunked_abort_req)


if __name__ == "__main__":
    unittest.main(verbosity=2)
