from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from multiprocessing import Process
from typing import Callable, List, Optional

import psutil

from sglang.srt.utils.cudacore_pyspy_dump_utils import pyspy_dump_schedulers

logger = logging.getLogger(__name__)

HARD_WATCHDOG_GRACE_SECONDS = 5.0


class Watchdog:
    @staticmethod
    def create(
        debug_name: str,
        watchdog_timeout: Optional[float],
        soft: bool = False,
        test_stuck_time: float = 0,
    ) -> Watchdog:
        if watchdog_timeout is None:
            assert test_stuck_time == 0, (
                f"stuck tester can be enabled only if soft watchdog is enabled."
            )
            return _WatchdogNoop()
        return _WatchdogReal(
            debug_name=debug_name,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
            test_stuck_time=test_stuck_time,
        )

    def feed(self):
        pass

    @contextmanager
    def disable(self):
        yield


class _WatchdogReal(Watchdog):
    def __init__(
        self,
        debug_name: str,
        watchdog_timeout: float,
        soft: bool = False,
        test_stuck_time: float = 0,
    ):
        self._counter = 0
        self._active = True
        self._test_stuck_time = test_stuck_time
        self._test_stuck_triggered = False
        self._raw = WatchdogRaw(
            debug_name=debug_name,
            get_counter=lambda: self._counter,
            is_active=lambda: self._active,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
        )
        logger.info(f"Watchdog {self._raw.debug_name} initialized.")
        if self._test_stuck_time > 0:
            logger.info(
                f"Watchdog {self._raw.debug_name} is configured to use {test_stuck_time=}."
            )

    def feed(self):
        # Only trigger the test stuck behavior once to avoid blocking server
        # startup health checks while still testing watchdog timeout detection
        if self._test_stuck_time > 0 and not self._test_stuck_triggered:
            self._test_stuck_triggered = True
            logger.info(
                f"Watchdog {self._raw.debug_name} start deliberately stuck for {self._test_stuck_time}s"
            )
            time.sleep(self._test_stuck_time)
            logger.info(
                f"Watchdog {self._raw.debug_name} end deliberately stuck for {self._test_stuck_time}s"
            )

        self._counter += 1

    @contextmanager
    def disable(self):
        assert self._active
        self._active = False
        try:
            yield
        finally:
            assert not self._active
            self._active = True


class _WatchdogNoop(Watchdog):
    pass


class WatchdogRaw:
    def __init__(
        self,
        debug_name: str,
        get_counter: Callable[[], int],
        is_active: Callable[[], bool],
        watchdog_timeout: float,
        soft: bool = False,
        dump_info: Optional[Callable[[], str]] = None,
        on_failure: Optional[Callable[[], None]] = None,
    ):
        self.debug_name = debug_name
        self.get_counter = get_counter
        self.is_active = is_active
        self.watchdog_timeout = watchdog_timeout
        self.soft = soft
        self.dump_info = dump_info
        self.on_failure = on_failure

        self.parent_process = psutil.Process().parent()
        t = threading.Thread(target=self._watchdog_thread, daemon=True)
        t.start()

    def _watchdog_thread(self):
        try:
            while True:
                self._watchdog_once()
        except Exception as e:
            logger.error(
                f"{self.debug_name} watchdog thread crashed: {e}", exc_info=True
            )

    def _watchdog_once(self):
        watchdog_last_counter = 0
        watchdog_last_time = time.perf_counter()
        was_active = True

        while True:
            current = time.perf_counter()
            if self.is_active():
                current_counter = self.get_counter()
                if was_active and watchdog_last_counter == current_counter:
                    if current > watchdog_last_time + self.watchdog_timeout:
                        break
                else:
                    watchdog_last_counter = current_counter
                    watchdog_last_time = current
                was_active = True
            else:
                # Idle time is not evidence of stalled model execution. Arm a
                # fresh timeout when work resumes without a completed forward.
                was_active = False
                watchdog_last_time = current
            time.sleep(self.watchdog_timeout / 2)

        self._handle_timeout()

    def _handle_timeout(self):
        logger.error(
            f"{self.debug_name} watchdog timeout "
            f"({self.watchdog_timeout=}, {self.soft=})"
        )
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        if self.soft:
            # Diagnostics must never prevent a hard watchdog from terminating.
            if self.dump_info is not None and (info_msg := self.dump_info()):
                logger.error(f"{self.debug_name} debug info:\n{info_msg}")
            pyspy_dump_schedulers()
            return

        if self.on_failure is not None:
            try:
                self.on_failure()
            except Exception:
                logger.exception("Failed to publish terminal engine failure")
        try:
            # Keep process identities while the parent still exists. Its exit
            # reparents children, so a later tree walk cannot find the siblings.
            descendants = self.parent_process.children(recursive=True)
        except psutil.Error:
            descendants = []
        try:
            self.parent_process.send_signal(signal.SIGQUIT)
            self.parent_process.wait(timeout=HARD_WATCHDOG_GRACE_SECONDS)
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            logger.exception("Parent did not exit after watchdog SIGQUIT")

        self._terminate_engine(descendants)

    def _terminate_engine(self, descendants):
        # Also runs after a parent exits without cleaning up, or was already
        # gone. A fatal scheduler must never return to its stalled event loop.
        try:
            for process in [*descendants, self.parent_process]:
                if process.pid == os.getpid():
                    continue
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
                except psutil.Error:
                    logger.exception("Failed to kill a failed engine process")
        finally:
            os._exit(1)


class SubprocessWatchdog:
    """Monitors subprocess liveness and triggers SIGQUIT when a crash is detected.

    When a subprocess crashes (e.g., NCCL timeout causing C++ std::terminate()),
    Python exception handlers never run, leaving the main process as a zombie
    service. This watchdog polls subprocess liveness in a daemon thread and
    sends SIGQUIT to trigger proper cleanup.

    See: https://github.com/sgl-project/sglang/issues/18421
    """

    def __init__(
        self,
        processes: List[Process],
        process_names: Optional[List[str]] = None,
        interval: float = 1.0,
        on_failure: Optional[Callable[[], None]] = None,
    ):
        self._processes = processes
        self._names = process_names or [f"process_{i}" for i in range(len(processes))]
        self._interval = interval
        self._on_failure = on_failure
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None or not self._processes:
            return
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="subprocess-watchdog"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
            self._thread = None

    def _monitor_loop(self) -> None:
        try:
            while not self._stop_event.wait(self._interval):
                if self._check_processes():
                    return
        except Exception as e:
            logger.error(f"SubprocessWatchdog thread crashed: {e}", exc_info=True)

    def _check_processes(self) -> bool:
        for proc, name in zip(self._processes, self._names):
            if proc.is_alive() or proc.exitcode == 0:
                continue

            logger.error(
                f"Subprocess {name} (pid={proc.pid}) crashed "
                f"with exit code {proc.exitcode}. "
                f"Triggering SIGQUIT for cleanup..."
            )
            try:
                if self._on_failure is not None:
                    self._on_failure()
            finally:
                os.kill(os.getpid(), signal.SIGQUIT)
            return True
        return False
