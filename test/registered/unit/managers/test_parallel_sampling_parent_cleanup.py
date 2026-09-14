"""Expanded parallel-sample parent placeholders must not outlive a request."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("samples", [2, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_parallel_sampling_releases_all_parent_placeholders(
    stream, samples, batch_size
):
    request = GenerateReqInput(
        text=["first", "second"][:batch_size],
        sampling_params={"n": samples, "max_new_tokens": 16},
        stream=stream,
    )
    request.normalize_batch_and_arguments()
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.rid_to_state = {
        rid: SimpleNamespace(time_stats=MagicMock(), dispatched=False)
        for rid in request.rid
    }
    unrelated = object()
    manager.rid_to_state["unrelated-inflight"] = unrelated
    manager._tokenize_one_request = AsyncMock(
        side_effect=lambda obj: SimpleNamespace(
            rid=obj.rid,
            sampling_params=SimpleNamespace(max_new_tokens=16),
            mm_inputs=None,
            input_ids=[1, 2],
        )
    )
    manager._send_one_request = MagicMock(
        side_effect=lambda obj: manager._mark_state_dispatched(obj.rid)
    )

    def init_state(obj):
        manager.rid_to_state[obj.rid] = SimpleNamespace(
            time_stats=MagicMock(), dispatched=False
        )

    async def wait_response(obj, _request=None):
        manager.rid_to_state.pop(obj.rid)
        yield {"meta_info": {"id": obj.rid}, "text": "ok"}

    manager._init_req_state = init_state
    manager._wait_one_response = wait_response

    async def run():
        return [response async for response in manager._handle_batch_request(request)]

    outputs = asyncio.run(run())
    assert len(outputs) == (samples * batch_size if stream else 1)
    if not stream:
        assert len(outputs[0]) == samples * batch_size
    assert manager.rid_to_state == {"unrelated-inflight": unrelated}
