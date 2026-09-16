"""Backport regressions for upstream #35255 on the stable sync dispatcher."""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch, AsyncMock

from test_tokenizer_manager_rid_cleanup import (
    CustomTestCase,
    _make_generate_obj,
    _make_req_state,
    _make_tm_for_generate,
    _make_tokenizer_manager,
)
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler import Scheduler


class TestDispatchedCleanup(CustomTestCase):
    def manager(self):
        manager = _make_tokenizer_manager(self)
        manager.server_args.tokenizer_worker_num = 1
        manager._dispatch_to_scheduler = Mock()
        return manager

    def test_mixed_batch_aborts_delivered_once_and_keeps_its_state(self):
        manager = self.manager()
        live = _make_req_state("live")
        live.dispatched = True
        manager.rid_to_state.update(live=live, pending=_make_req_state("pending"))
        manager._release_req_states_on_failure(["live", "pending", "absent"])
        manager._release_req_states_on_failure(["live", "pending"])
        self.assertEqual(set(manager.rid_to_state), {"live"})
        self.assertTrue(live.abort_sent)
        self.assertEqual(manager._dispatch_to_scheduler.call_count, 1)
        sent = manager._dispatch_to_scheduler.call_args.args[0]
        self.assertIsInstance(sent, AbortReq)
        self.assertEqual(sent.rid, "live")

    def test_send_failure_can_retry_without_losing_state(self):
        manager = self.manager()
        live = _make_req_state("live")
        live.dispatched = True
        manager.rid_to_state["live"] = live
        manager._dispatch_to_scheduler.side_effect = RuntimeError("send failure")
        with self.assertLogs(level="ERROR"):
            manager._release_req_states_on_failure(["live"])
        self.assertFalse(live.abort_sent)
        self.assertIn("live", manager.rid_to_state)
        manager._dispatch_to_scheduler.side_effect = None
        manager._release_req_states_on_failure(["live"])
        self.assertTrue(live.abort_sent)

    @patch("sglang.srt.managers.tokenizer_manager.wrap_shm_features", side_effect=lambda obj: obj)
    def test_cancel_after_dispatch_aborts_before_discard(self, _wrap):
        manager = _make_tm_for_generate(self)
        manager.cuda_vmm_feature_transport = Mock()
        manager.cuda_vmm_feature_transport.prepare_for_dispatch.return_value = []
        manager._dispatch_to_scheduler = Mock()
        rid = "disconnected"
        obj = _make_generate_obj(rid, True)
        obj.return_prompt_token_ids = False
        tokenized = MagicMock()
        tokenized.rid = rid
        tokenized.mm_inputs = None
        manager._tokenize_one_request = AsyncMock(return_value=tokenized)

        async def drive():
            task = asyncio.create_task(manager.generate_request(obj).__anext__())
            for _ in range(100):
                await asyncio.sleep(0)
                if manager._dispatch_to_scheduler.called:
                    break
            self.assertTrue(manager.rid_to_state[rid].dispatched)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(drive())
        sent = [call.args[0] for call in manager._dispatch_to_scheduler.call_args_list]
        self.assertEqual(sum(isinstance(item, AbortReq) and item.rid == rid for item in sent), 1)
        self.assertIn(rid, manager.rid_to_state)

    def test_empty_id_never_aborts_all_requests(self):
        manager = self.manager()
        manager.abort_request("")
        manager._dispatch_to_scheduler.assert_not_called()


class TestDeferredChunkAbort(CustomTestCase):
    def test_moved_request_is_aborted_in_its_current_queue(self):
        req = SimpleNamespace(rid="moved", finished=lambda: False, kv=SimpleNamespace(holds_kv=True))
        scheduler = SimpleNamespace(_pending_chunked_abort_req=req, chunked_req=None, abort_request=Mock())
        Scheduler.process_pending_chunked_abort(scheduler)
        self.assertIsNone(scheduler._pending_chunked_abort_req)
        self.assertEqual(scheduler.abort_request.call_args.args[0].rid, req.rid)

    def test_finished_request_only_clears_pending_marker(self):
        req = SimpleNamespace(rid="finished", finished=lambda: True, kv=SimpleNamespace(holds_kv=False))
        scheduler = SimpleNamespace(_pending_chunked_abort_req=req, chunked_req=None, abort_request=Mock())
        Scheduler.process_pending_chunked_abort(scheduler)
        self.assertIsNone(scheduler._pending_chunked_abort_req)
        scheduler.abort_request.assert_not_called()
