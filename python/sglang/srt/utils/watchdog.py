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
        hard_exit_timeout: float = 30.0,
        hard_exit_fn: Optional[Callable[[int], None]] = None,
    ):
        self.debug_name = debug_name
        self.get_counter = get_counter
        self.is_active = is_active
        self.watchdog_timeout = watchdog_timeout
        self.soft = soft
        self.dump_info = dump_info
        self.hard_exit_timeout = hard_exit_timeout
        self.hard_exit_fn = hard_exit_fn or os._exit

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

        while True:
            current = time.perf_counter()
            if self.is_active():
                current_counter = self.get_counter()
                if watchdog_last_counter == current_counter:
                    if current > watchdog_last_time + self.watchdog_timeout:
                        break
                else:
                    watchdog_last_counter = current_counter
                    watchdog_last_time = current
            time.sleep(self.watchdog_timeout / 2)

        hard_exit_guard = None
        if not self.soft:
            # Diagnostics are useful, but they must never be allowed to keep a
            # wedged scheduler alive forever. The timer runs in a separate
            # thread and terminates this scheduler even if an invariant check
            # or py-spy dump blocks.
            hard_exit_guard = threading.Timer(
                self.hard_exit_timeout,
                self._hard_exit_after_diagnostic_timeout,
            )
            hard_exit_guard.daemon = True
            hard_exit_guard.start()

        # Timeout diagnostics must be best-effort. A broken CUDA context can
        # make an invariant dump raise before a hard watchdog signals the main
        # process, which leaves the container alive but unusable.
        if self.dump_info is not None:
            try:
                info_msg = self.dump_info()
            except Exception as e:
                logger.error(
                    f"{self.debug_name} failed to dump watchdog debug info: {e}",
                    exc_info=True,
                )
            else:
                if info_msg:
                    logger.error(f"{self.debug_name} debug info:\n{info_msg}")

        try:
            pyspy_dump_schedulers()
        except Exception as e:
            logger.error(
                f"{self.debug_name} failed to dump scheduler stacks: {e}",
                exc_info=True,
            )
        logger.error(
            f"{self.debug_name} watchdog timeout "
            f"({self.watchdog_timeout=}, {self.soft=})"
        )
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        if not self.soft:
            try:
                self.parent_process.send_signal(signal.SIGQUIT)
            finally:
                # Do not rely on the parent SIGQUIT handler alone. Exiting the
                # scheduler also gives the parent-side SubprocessWatchdog an
                # independent signal that the serving process is unrecoverable.
                self.hard_exit_fn(1)
                if hard_exit_guard is not None:
                    hard_exit_guard.cancel()

    def _hard_exit_after_diagnostic_timeout(self) -> None:
        logger.error(
            "%s hard-watchdog diagnostics exceeded %.1fs; force-exiting the "
            "scheduler process.",
            self.debug_name,
            self.hard_exit_timeout,
        )
        self.hard_exit_fn(1)


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
        sigquit_grace_period: float = 5.0,
        hard_exit_fn: Optional[Callable[[int], None]] = None,
    ):
        self._processes = processes
        self._names = process_names or [f"process_{i}" for i in range(len(processes))]
        self._interval = interval
        self._sigquit_grace_period = sigquit_grace_period
        self._hard_exit_fn = hard_exit_fn or os._exit
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
                os.kill(os.getpid(), signal.SIGQUIT)
            finally:
                # The SIGQUIT handler performs crash diagnostics before killing
                # the process tree. If signal delivery fails, or diagnostics
                # block on a broken CUDA context, the old behavior leaves a
                # permanently alive but unusable service. Give the normal
                # handler a short grace period, then unconditionally terminate
                # the container's main process.
                if self._sigquit_grace_period > 0:
                    time.sleep(self._sigquit_grace_period)
                logger.error(
                    "SIGQUIT cleanup did not terminate the process within %.1fs; "
                    "force-exiting to allow the service supervisor to restart it.",
                    self._sigquit_grace_period,
                )
                self._hard_exit_fn(1)
            return True
        return False
