from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def cc_async_d2h_enabled() -> bool:
    """Opt-in gate for attributable CC D2H qualification on the pinned runtime."""
    value = os.environ.get("SGLANG_CC_ASYNC_D2H", "0")
    if value not in ("0", "1"):
        raise ValueError("SGLANG_CC_ASYNC_D2H must be '0' or '1'")
    if value == "0":
        return False
    from sglang.srt.utils.confidential_compute import is_confidential_compute

    if not is_confidential_compute():
        raise ValueError("SGLANG_CC_ASYNC_D2H=1 requires GPU CC/PPCIE detection")
    return True


class HostCopyDone:
    """Completion handle for an async device->host copy run on a worker thread.

    A drop-in for ``torch.cuda.Event`` as far as copy-completion consumers are
    concerned: it exposes the same ``record()`` / ``synchronize()`` / ``query()``
    surface, so a caller can hand it to code that already waits on a CUDA event
    (e.g. store it in a ``copy_done`` field) without that code needing to know
    the copy was offloaded to a host thread.

    Unlike a CUDA event it also carries any exception the copy raised;
    ``synchronize()`` re-raises it so a failed copy aborts the consumer instead
    of letting it read invalid host tensors. ``error`` is None iff the copy
    succeeded.
    """

    def __init__(self):
        self._done = threading.Event()
        self.error: Optional[BaseException] = None

    def record(self, *args, **kwargs) -> None:
        # Some copy routines end with ``handle.record()`` (a CUDA-event habit);
        # the real completion signal is the worker calling ``set_done``, so this
        # is a no-op kept for drop-in parity with torch.cuda.Event.
        pass

    def set_done(self, error: Optional[BaseException] = None) -> None:
        """Signal completion. Called by the worker once the copy has finished
        (``error`` None) or failed (``error`` set)."""
        self.error = error
        self._done.set()

    def synchronize(self) -> None:
        """Block until the copy completes; re-raise if it failed."""
        self._done.wait()
        if self.error is not None:
            raise RuntimeError(
                "Async device->host copy failed; destination tensors are invalid"
            ) from self.error

    def query(self) -> bool:
        """True once the copy has completed (successfully or not)."""
        return self._done.is_set()


class AsyncD2HCopyWorker:
    """Runs blocking device->host copies on a dedicated daemon thread.

    Some environments force a device-to-host ``cudaMemcpyAsync`` to be
    synchronous and block AT ISSUE — most notably NVIDIA Confidential Computing
    (bounce-buffer CC), where the host destination is staged through an encrypted
    bounce buffer. Issuing such a copy inline then stalls the *submitting* thread
    for the whole copy, which is fatal to any pipeline that relies on that thread
    staying free (for example the SGLang overlap scheduler, which must keep
    launching the next step).

    This worker moves the (still-blocking) copy off the caller's thread so the
    caller keeps running. The pattern:
      - the caller submits ``copy_fn`` with the source-producing stream current;
        ``submit`` records a readiness event on that stream, hands back a
        ``HostCopyDone``, and returns immediately;
      - this worker ``cudaEventSynchronize``-s on that event (event-sync, NOT a
        stream-wait, so the blocking copy never stalls the caller's CUDA API
        calls), runs the copy on its own private ``d2h_copy_stream``, blocks
        until it completes, and signals the handle;
      - the caller (or a downstream consumer) later calls
        ``HostCopyDone.synchronize()`` to wait for the copy.

    The stream is private to this worker on purpose: it must never be a stream
    the caller also enqueues onto. If it were shared, this worker's blanket
    ``synchronize()`` could block on unrelated work the caller queued after
    submitting (e.g. a later ``wait_stream``), re-coupling the caller to the copy
    and defeating the point.

    The copy itself stays synchronous when the platform forces it; it is merely
    non-blocking *to the submitting thread*.
    """

    def __init__(self, device_module):
        self.device_module = device_module
        # Private stream, created and owned here so nothing outside this worker
        # can enqueue onto it (see class docstring). Created on the caller's
        # thread, so it lands on the current device.
        self.d2h_copy_stream = device_module.Stream()
        self._queue: queue.Queue = queue.Queue()
        self._state_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._loop, name="sglang-d2h-copy-worker", daemon=True
        )
        self._thread.start()

    def submit(
        self, copy_fn: Callable[[], None], *, done: Optional[HostCopyDone] = None
    ) -> HostCopyDone:
        """Record readiness on the CURRENT stream and enqueue the copy.

        Must be called with the stream that produced the copy sources as the
        current stream: ``submit`` records a completion event on it synchronously
        and the worker waits on that event before copying. ``copy_fn`` performs
        the actual ``.to("cpu", ...)`` copies. The returned ``HostCopyDone`` is
        signaled once the copies complete (or fail); its ``synchronize()`` blocks
        until then and re-raises on failure.
        """
        # Publishing the caller-provided handle before enqueue avoids a race
        # with copy routines that access batch_result.copy_done.record().
        with self._state_lock:
            if self._closed:
                raise RuntimeError("AsyncD2HCopyWorker is shut down")
            src_ready = self.device_module.Event()
            src_ready.record()
            if done is None:
                done = HostCopyDone()
            self._queue.put((src_ready, copy_fn, done))
            return done

    def _loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._run_copy(*item)
            finally:
                # Do not pin the last completed batch/closure while Queue.get
                # is idle. The helper's frame also releases its copy_fn ref.
                del item

    def _run_copy(self, src_ready, copy_fn, done):
        error = None
        try:
            # Event-sync rather than a shared-stream wait: the host thread
            # submitting the next forward must remain independent of D2H.
            src_ready.synchronize()
            with self.device_module.stream(self.d2h_copy_stream):
                copy_fn()
            self.d2h_copy_stream.synchronize()
        except Exception as exc:
            logger.exception("AsyncD2HCopyWorker copy failed")
            error = exc
        finally:
            done.set_done(error=error)

    def shutdown(self, timeout: float = 2.0) -> bool:
        """Drain accepted work and join for a bounded time; return if stopped."""
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()
