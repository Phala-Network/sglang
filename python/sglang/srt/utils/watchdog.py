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
from sglang.srt.utils.framework_log_privacy import safe_exception_type

logger = logging.getLogger(__name__)

HARD_WATCHDOG_GRACE_SECONDS = 5.0


def _best_effort_error(message, *args):
    try:
        logger.error(message, *args)
    except Exception:
        # A closed pipe, disk-full handler, or custom sink cannot own recovery.
        pass


def _best_effort_flush():
    for stream in (sys.stderr, sys.stdout):
        try:
            stream.flush()
        except Exception:
            pass


class Watchdog:
    @staticmethod
    def create(
        debug_name: str,
        watchdog_timeout: Optional[float],
        soft: bool = False,
        test_stuck_time: float = 0,
    ) -> Watchdog:
        if watchdog_timeout is None:
            assert (
                test_stuck_time == 0
            ), f"stuck tester can be enabled only if soft watchdog is enabled."
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
    ):
        self.debug_name = debug_name
        self.get_counter = get_counter
        self.is_active = is_active
        self.watchdog_timeout = watchdog_timeout
        self.soft = soft
        self.dump_info = dump_info

        self.parent_process = psutil.Process().parent()
        t = threading.Thread(target=self._watchdog_thread, daemon=True)
        t.start()

    def _watchdog_thread(self):
        try:
            while True:
                self._watchdog_once()
        except Exception as e:
            _best_effort_error(
                "Watchdog thread crashed (type=%s)", safe_exception_type(e)
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

        self._handle_timeout()

    def _run_diagnostics(self):
        """Run best-effort diagnostics for a non-fatal soft watchdog timeout."""
        try:
            counter, active = self.get_counter(), self.is_active()
        except Exception as e:
            _best_effort_error(
                "Soft watchdog counter snapshot failed (type=%s)",
                safe_exception_type(e),
            )
        else:
            _best_effort_error(
                "Soft watchdog state (counter=%d, active=%s)",
                counter if type(counter) is int else 0,
                active if type(active) is bool else False,
            )
        if self.dump_info is not None:
            try:
                # Existing callbacks may perform diagnostics, but their opaque
                # return values can contain requests, RIDs, or token sequences.
                self.dump_info()
            except Exception as e:
                _best_effort_error(
                    "Soft watchdog debug snapshot failed (type=%s)",
                    safe_exception_type(e),
                )
            else:
                _best_effort_error(
                    "Soft watchdog debug snapshot completed (details redacted)"
                )

        try:
            pyspy_dump_schedulers()
        except Exception as e:
            _best_effort_error(
                "Soft watchdog scheduler stack dump failed (type=%s)",
                safe_exception_type(e),
            )

    def _handle_timeout(self):
        """Handle a detected timeout without letting diagnostics block recovery."""
        if self.soft:
            _best_effort_error(
                "Soft watchdog timeout (timeout_seconds=%s)",
                (
                    self.watchdog_timeout
                    if type(self.watchdog_timeout) in (int, float)
                    else 0
                ),
            )
            self._run_diagnostics()
            _best_effort_flush()
            return

        # Do not touch logging, streams, disk, or diagnostic callbacks before
        # the existing signal/wait/kill recovery sequence has been attempted.
        outcome, failure_type = "parent exited", "none"
        try:
            try:
                self.parent_process.send_signal(signal.SIGQUIT)
            except psutil.NoSuchProcess:
                outcome = "parent already gone"
                return
            except Exception as e:
                failure_type = safe_exception_type(e)

            try:
                self.parent_process.wait(timeout=HARD_WATCHDOG_GRACE_SECONDS)
                return
            except psutil.NoSuchProcess:
                outcome = "parent already gone"
                return
            except psutil.TimeoutExpired:
                pass
            except Exception as e:
                failure_type = safe_exception_type(e)

            try:
                self.parent_process.kill()
                outcome = "SIGKILL requested"
            except psutil.NoSuchProcess:
                outcome = "parent already gone"
            except Exception as e:
                outcome, failure_type = "SIGKILL failed", safe_exception_type(e)
        finally:
            _best_effort_error(
                "Hard watchdog timeout (timeout_seconds=%s, grace_seconds=%s, outcome=%s, type=%s)",
                (
                    self.watchdog_timeout
                    if type(self.watchdog_timeout) in (int, float)
                    else 0
                ),
                HARD_WATCHDOG_GRACE_SECONDS,
                outcome,
                failure_type,
            )
            _best_effort_flush()


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
    ):
        self._processes = processes
        self._names = process_names or [f"process_{i}" for i in range(len(processes))]
        self._interval = interval
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
            _best_effort_error(
                "SubprocessWatchdog thread crashed (type=%s)", safe_exception_type(e)
            )

    def _check_processes(self) -> bool:
        for proc, _ in zip(self._processes, self._names):
            if proc.is_alive() or proc.exitcode == 0:
                continue

            os.kill(os.getpid(), signal.SIGQUIT)
            _best_effort_error(
                "Subprocess crashed; SIGQUIT requested (pid=%d, exit_code=%d)",
                proc.pid if type(proc.pid) is int else 0,
                proc.exitcode if type(proc.exitcode) is int else 0,
            )
            return True
        return False
