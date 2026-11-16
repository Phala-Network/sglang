"""Bounded reproduction of the old Mock.state fixture and the corrected state."""
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "registered/unit/managers"))
import test_tokenizer_manager_rid_cleanup as fixture
from sglang.srt.entrypoints.request_disconnect import response_disconnect_watched


async def inspect_case(correct_state):
    case = unittest.TestCase()
    manager = fixture._make_tokenizer_manager(case)
    manager.abort_request = Mock()
    state = fixture._make_req_state("synthetic-await-diagnostic")
    state.obj.stream = True
    state.obj.background = False
    manager.rid_to_state[state.obj.rid] = state
    request = Mock()
    request.is_disconnected = AsyncMock(return_value=True)
    if correct_state:
        request.state = SimpleNamespace()
    stream = manager._wait_one_response(state.obj, request)
    with patch("sglang.srt.managers.tokenizer_manager._REQUEST_STATE_WAIT_TIMEOUT", 0.001):
        task = asyncio.create_task(anext(stream))
        try:
            await asyncio.sleep(0.025)
            chain = []
            current = task.get_coro()
            for _ in range(12):
                if current is None:
                    break
                code = getattr(current, "cr_code", None) or getattr(current, "ag_code", None)
                chain.append(code.co_name if code is not None else type(current).__name__)
                current = getattr(current, "cr_await", None) or getattr(current, "ag_await", None)
            result = {"correct_state": correct_state,
                      "disconnect_watched": response_disconnect_watched(request),
                      "pending_after_25ms": not task.done(), "await_chain_names_only": chain,
                      "disconnect_probe_calls": request.is_disconnected.await_count}
            print(json.dumps(result), flush=True)
            if correct_state:
                assert task.done() and request.is_disconnected.await_count == 1
                assert isinstance(task.exception(), ValueError)
            else:
                assert not task.done() and request.is_disconnected.await_count == 0
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await stream.aclose()
            case.doCleanups()


async def main():
    async with asyncio.timeout(2):
        await inspect_case(False)
        await inspect_case(True)


asyncio.run(main())
