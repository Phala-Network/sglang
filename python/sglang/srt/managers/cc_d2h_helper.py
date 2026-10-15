"""Non-blocking result D2H under NVIDIA Confidential Computing.

With GPU CC on, a pinned device-to-host copy that depends on unfinished GPU work blocks the
calling thread until that work completes, holding the CUDA context's driver lock. The overlap
scheduler's `copy_to_cpu`, issued right after the forward it copies, therefore serializes the
scheduler with the GPU. The same copy issued from another process (another CUDA context) that
waits on an IPC event returns at once and does not stall this context.

Each scheduler process therefore gets a helper process. `copy_to_cpu` packs the result tensors
into a persistent GPU staging slot (device-to-device copies, which do not block), records an
interprocess event and hands the slot to the helper. The helper waits for the event, copies the
slot to pinned memory and publishes it in shared memory. The result fields become views of that
shared memory and `copy_done` waits for the publication, so callers keep the existing contract:
read the host tensors only after `copy_done.synchronize()`.

A tensor that does not fit the slot, and a call with no free slot, take the regular path.
"""

import atexit
import collections
import time
from multiprocessing import shared_memory
from typing import Callable

import torch
import torch.multiprocessing as mp

from sglang.srt.environ import envs

_ALIGN = 64


def enabled() -> bool:
    return envs.SGLANG_CC_D2H_HELPER.get() and torch.cuda.is_available()


def _helper_main(device, staging, event_handles, shm_name, slots, slot_bytes, conn):
    torch.cuda.set_device(device)
    events = [
        torch.cuda.Event.from_ipc_handle(torch.device("cuda", device), h)
        for h in event_handles
    ]
    shm = shared_memory.SharedMemory(name=shm_name)
    host = torch.frombuffer(shm.buf, dtype=torch.uint8)
    seqs = torch.frombuffer(
        shm.buf, dtype=torch.int64, offset=slots * slot_bytes, count=slots
    )
    pinned = torch.empty(slots * slot_bytes, dtype=torch.uint8, pin_memory=True)
    stream = torch.cuda.Stream(device)
    while (msg := conn.recv()) is not None:
        slot, seq, nbytes = msg
        lo, hi = slot * slot_bytes, slot * slot_bytes + nbytes
        stream.wait_event(events[slot])
        with torch.cuda.stream(stream):
            pinned[lo:hi].copy_(staging[lo:hi], non_blocking=True)
        stream.synchronize()
        host[lo:hi].copy_(pinned[lo:hi])
        seqs[slot] = seq  # published after the data


class _Helper:
    def __init__(self, device: torch.device):
        self.slots = envs.SGLANG_CC_D2H_HELPER_SLOTS.get()
        self.slot_bytes = envs.SGLANG_CC_D2H_HELPER_SLOT_KB.get() * 1024
        self.device = device
        self.staging = torch.empty(
            self.slots * self.slot_bytes, dtype=torch.uint8, device=device
        )
        self.events = [torch.cuda.Event(interprocess=True) for _ in range(self.slots)]
        self.shm = shared_memory.SharedMemory(
            create=True, size=self.slots * (self.slot_bytes + 8)
        )
        self.seqs = torch.frombuffer(
            self.shm.buf,
            dtype=torch.int64,
            offset=self.slots * self.slot_bytes,
            count=self.slots,
        )
        self.seqs.fill_(-1)
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(
            target=_helper_main,
            args=(
                device.index,
                self.staging,
                [e.ipc_handle() for e in self.events],
                self.shm.name,
                self.slots,
                self.slot_bytes,
                child,
            ),
            daemon=True,
        )
        self.proc.start()
        self.free = collections.deque(range(self.slots))
        self.seq = 0
        atexit.register(self.shm.unlink)

    def submit(self, slot: int, nbytes: int) -> int:
        self.events[slot].record()
        self.seq += 1
        self.conn.send((slot, self.seq, nbytes))
        return self.seq


class _Packer:
    """Stands in for the D2H primitive during one `copy_to_cpu`."""

    def __init__(self, helper: _Helper, slot: int, fallback: Callable):
        self.helper, self.fallback = helper, fallback
        self.base = slot * helper.slot_bytes
        self.used = 0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        nbytes = t.numel() * t.element_size()
        if (
            not t.is_cuda
            or t.device != self.helper.device
            or self.used + nbytes > self.helper.slot_bytes
        ):
            return self.fallback(t)
        if not nbytes:
            return torch.empty(t.shape, dtype=t.dtype)
        lo = self.base + self.used
        self.helper.staging[lo : lo + nbytes].copy_(
            t.contiguous().view(-1).view(torch.uint8)
        )
        t.record_stream(torch.cuda.current_stream(t.device))
        self.used += -(-nbytes // _ALIGN) * _ALIGN
        return torch.frombuffer(
            self.helper.shm.buf, dtype=t.dtype, count=t.numel(), offset=lo
        ).view(t.shape)


class _HelperDone:
    """`copy_done` for a slot handed to the helper."""

    def __init__(self, helper: _Helper, slot: int, seq: int):
        self.helper, self.slot, self.seq, self.done = helper, slot, seq, False

    def query(self) -> bool:
        return self.done or int(self.helper.seqs[self.slot]) == self.seq

    def synchronize(self) -> None:
        if self.done:
            return
        seqs, spins = self.helper.seqs, 0
        while int(seqs[self.slot]) != self.seq:
            spins += 1
            if spins > 200:
                time.sleep(20e-6)
            if spins % 200_000 == 0 and not self.helper.proc.is_alive():
                raise RuntimeError("cc_d2h_helper: the helper process exited")
        self.done = True
        self.helper.free.append(self.slot)


_helpers = {}


def copy_result_to_cpu(result, copy: Callable, d2h: Callable, *args) -> None:
    """Run `copy(d2h, *args)` with the helper's D2H; `copy` records `result.copy_done`."""
    device = torch.device("cuda", torch.cuda.current_device())
    helper = _helpers.get(device) or _helpers.setdefault(device, _Helper(device))
    if not helper.free:
        copy(d2h, *args)
        return
    slot = helper.free.popleft()
    packer = _Packer(helper, slot, d2h)
    copy(packer, *args)
    if not packer.used:
        helper.free.append(slot)
        return
    result.copy_done = _HelperDone(helper, slot, helper.submit(slot, packer.used))
