"""A terminal failure latch shared by one engine's spawned processes."""

import atexit
import os
import tempfile
from typing import Optional


def _remove_state(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def create_engine_failure_state() -> str:
    # Create before spawning. Only the launcher creates a generation; readers
    # must never create or reset a missing state file.
    fd, path = tempfile.mkstemp(prefix="sglang-engine-state-")
    try:
        os.write(fd, b"0")
    finally:
        os.close(fd)
    atexit.register(_remove_state, path)
    return path


def mark_engine_failed(path: Optional[str]) -> None:
    if path is None:
        return
    fd = os.open(path, os.O_WRONLY)
    try:
        os.write(fd, b"1")
    finally:
        os.close(fd)


def engine_has_failed(path: Optional[str]) -> bool:
    if path is None:
        return False
    try:
        with open(path, "rb") as state:
            return state.read(2) != b"0"
    except OSError:
        # A removed, unreadable or corrupt latch cannot advertise readiness.
        return True
