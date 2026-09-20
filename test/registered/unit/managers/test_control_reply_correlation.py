"""Late control replies cannot complete a later admin command."""
import asyncio
import unittest
import msgspec
from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import (
    GetInternalStateReq, GetInternalStateReqOutput,
    SetInternalStateReq, SetInternalStateReqOutput,
)


class CorrelationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_native_read_cannot_satisfy_plugin_read(self):
        sent = asyncio.Queue()
        channel = FanOutCommunicator(sent.put_nowait, 1, correlate=True)
        first = asyncio.create_task(channel(GetInternalStateReq()))
        old = await sent.get()
        first.cancel()
        with self.assertRaises(asyncio.CancelledError): await first
        second = asyncio.create_task(channel(GetInternalStateReq()))
        new = await sent.get()
        self.assertNotEqual(old.control_nonce, new.control_nonce)
        channel.handle_recv(GetInternalStateReqOutput(internal_state={'old':True},control_nonce=old.control_nonce))
        await asyncio.sleep(0)
        self.assertFalse(second.done())
        channel.handle_recv(GetInternalStateReqOutput(internal_state={'new':True},control_nonce=new.control_nonce))
        self.assertEqual((await second)[0].internal_state, {'new':True})

    async def test_old_success_cannot_confirm_rejected_patch(self):
        sent = asyncio.Queue()
        channel = FanOutCommunicator(sent.put_nowait, 1, correlate=True)
        one = asyncio.create_task(channel(SetInternalStateReq(server_args={})))
        old = await sent.get(); one.cancel()
        with self.assertRaises(asyncio.CancelledError): await one
        two = asyncio.create_task(channel(SetInternalStateReq(server_args={})))
        new = await sent.get()
        channel.handle_recv(SetInternalStateReqOutput(updated=True, control_nonce=old.control_nonce))
        channel.handle_recv(SetInternalStateReqOutput(updated=True))
        await asyncio.sleep(0); self.assertFalse(two.done())
        reply = SetInternalStateReqOutput(updated=False,control_nonce=new.control_nonce)
        reply = msgspec.msgpack.decode(msgspec.msgpack.encode(reply),type=SetInternalStateReqOutput)
        channel.handle_recv(reply)
        self.assertFalse((await two)[0].updated)

    async def test_sync_send_reply_and_send_failure_leave_valid_state(self):
        def send(request):
            channel.handle_recv(GetInternalStateReqOutput(internal_state={},control_nonce=request.control_nonce))
        channel = FanOutCommunicator(send, 1, correlate=True)
        self.assertEqual(len(await channel(GetInternalStateReq())),1)
        def failing(_request): raise OSError('synthetic send failure')
        channel._send = failing
        with self.assertRaises(OSError): await channel(GetInternalStateReq())
        self.assertIsNone(channel._nonce)
        self.assertIsNone(channel._result_event)
        channel._send = send
        self.assertEqual(len(await channel(GetInternalStateReq())),1)

    async def test_fanout_requires_all_current_nonce_replies(self):
        sent = asyncio.Queue()
        channel = FanOutCommunicator(sent.put_nowait, 2, correlate=True)
        task = asyncio.create_task(channel(GetInternalStateReq()))
        req = await sent.get()
        channel.handle_recv(GetInternalStateReqOutput(internal_state={},control_nonce='wrong'))
        channel.handle_recv(GetInternalStateReqOutput(internal_state={'rank':0},control_nonce=req.control_nonce))
        await asyncio.sleep(0); self.assertFalse(task.done())
        channel.handle_recv(GetInternalStateReqOutput(internal_state={'rank':1},control_nonce=req.control_nonce))
        self.assertEqual(len(await task),2)


if __name__ == '__main__': unittest.main()
