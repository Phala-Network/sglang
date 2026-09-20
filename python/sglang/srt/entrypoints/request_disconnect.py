"""One disconnect listener while an HTTP response is still being prepared."""

import asyncio


def response_disconnect_watched(request):
    return bool(
        getattr(
            getattr(request, "state", None), "sglang_response_disconnect_watched", False
        )
    )


async def await_response_or_disconnect(work, request, *, background=False):
    """Cancel and join pending work before handing receive to StreamingResponse.

    Call only after the endpoint has consumed the request body. Request waiters
    must not compete with this listener for the disconnect event. The caller
    retains ownership of its generators and closes them after this await.
    """
    if (
        request is None
        or background
        or not hasattr(request, "receive")
        or response_disconnect_watched(request)
    ):
        return await work

    async def disconnected():
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    request.state.sglang_response_disconnect_watched = True
    pending = asyncio.ensure_future(work)
    monitor = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait(
            (pending, monitor), return_when=asyncio.FIRST_COMPLETED
        )
        if monitor in done:
            monitor.result()
            raise ValueError("Request disconnected before the response was ready")
        return pending.result()
    finally:
        for task in (pending, monitor):
            if not task.done():
                task.cancel()
        await asyncio.gather(pending, monitor, return_exceptions=True)
        request.state.sglang_response_disconnect_watched = False
