# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CUDA core dump and py-spy dump utilities."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
from errno import ENXIO
from pathlib import Path
from typing import List

import psutil

from sglang.srt.utils.framework_log_privacy import safe_exception_type

logger = logging.getLogger(__name__)

PYSPY_DUMP_TIMEOUT_SECONDS = 10


def _best_effort_error(message, *args):
    try:
        logger.error(message, *args)
    except Exception:
        # Diagnostic output failure must not abort later recovery or targets.
        pass


def _resolve_cuda_coredump_pipe_path(proc: psutil.Process) -> Path:
    pipe_template = os.environ.get("CUDA_COREDUMP_PIPE")
    if pipe_template is None:
        pipe_path = f"corepipe.cuda.{platform.node()}.{proc.pid}"
    else:
        pipe_path = (
            pipe_template.replace("%h", platform.node())
            .replace("%p", str(proc.pid))
            .replace("%t", str(int(time.time())))
        )

    path = Path(pipe_path)
    if path.is_absolute():
        return path

    try:
        return Path(proc.cwd()) / path
    except (psutil.Error, OSError):
        return Path.cwd() / path


def _is_sglang_scheduler_process(proc: psutil.Process) -> bool:
    try:
        proc_title = " ".join(proc.cmdline())
    except (psutil.Error, OSError):
        return False
    return proc_title.startswith("sglang::scheduler")


def collect_scheduler_processes() -> List[psutil.Process]:
    current = psutil.Process()
    return [
        proc
        for proc in current.children(recursive=True)
        if _is_sglang_scheduler_process(proc)
    ]


def pyspy_dump_schedulers(scheduler_only=False):
    """py-spy dump on all scheduler in a local node."""
    if scheduler_only:
        procs = collect_scheduler_processes()
        if not procs:
            _best_effort_error("No sglang scheduler processes found for py-spy dump.")
            return
        pids = [proc.pid for proc in procs]
    else:
        pids = [psutil.Process().pid]
    for pid in pids:
        for attempt, native in enumerate((True, False)):
            try:
                cmd = ["py-spy", "dump"]
                if native:
                    cmd.append("--native")
                cmd.extend(("--pid", str(pid)))
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=PYSPY_DUMP_TIMEOUT_SECONDS,
                )
                # py-spy stdout/stderr may contain argv and customer data.
                # Capture them so the child cannot inherit the service sinks,
                # but never render or persist either stream in diagnostics.
                _best_effort_error(
                    "Py-spy dump completed (pid=%s, native=%s, output redacted)",
                    pid,
                    native,
                )
                break
            except subprocess.TimeoutExpired:
                _best_effort_error(
                    "Py-spy timed out (timeout_seconds=%s, pid=%s)",
                    PYSPY_DUMP_TIMEOUT_SECONDS,
                    pid,
                )
                if attempt == 1:
                    _best_effort_error("All py-spy dump attempts failed (pid=%s)", pid)
            except subprocess.CalledProcessError:
                _best_effort_error(
                    "Py-spy command failed (pid=%s, type=CalledProcessError)", pid
                )
                if attempt == 1:
                    _best_effort_error("All py-spy dump attempts failed (pid=%s)", pid)
            except OSError as e:
                _best_effort_error(
                    "Py-spy could not start (pid=%s, type=%s)",
                    pid,
                    safe_exception_type(e),
                )
                if attempt == 1:
                    _best_effort_error("All py-spy dump attempts failed (pid=%s)", pid)


def trigger_cuda_user_coredump(scheduler_only=False):
    """Trigger CUDA user-induced GPU core dumps by writing to coredump pipes."""
    if os.environ.get("CUDA_ENABLE_USER_TRIGGERED_COREDUMP") != "1":
        _best_effort_error(
            "CUDA user-triggered coredump is not enabled. Set "
            "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 before CUDA initialization."
        )

    if scheduler_only:
        procs = collect_scheduler_processes()
        if not procs:
            _best_effort_error("No sglang scheduler processes found for CUDA coredump.")
            return
    else:
        procs = [psutil.Process()]

    for proc in procs:
        try:
            pipe_path = _resolve_cuda_coredump_pipe_path(proc)
            fd = os.open(pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            try:
                os.write(fd, b"1")
            finally:
                os.close(fd)
            _best_effort_error(
                "Triggered CUDA user coredump (pid=%s)",
                proc.pid,
            )
        except FileNotFoundError:
            _best_effort_error(
                "CUDA coredump pipe not found (pid=%s). Ensure "
                "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 was set before this "
                "process initialized CUDA.",
                proc.pid,
            )
        except OSError as e:
            if e.errno == ENXIO:
                _best_effort_error(
                    "CUDA coredump pipe has no reader (pid=%s)",
                    proc.pid,
                )
            else:
                _best_effort_error(
                    "Failed to trigger CUDA user coredump (pid=%s, type=%s)",
                    proc.pid,
                    safe_exception_type(e),
                )
        except psutil.Error as e:
            _best_effort_error(
                "Failed to resolve CUDA coredump target (type=%s)",
                safe_exception_type(e),
            )
