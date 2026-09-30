import importlib
import os
import signal
import sys

import pytest

from sglang.srt.debug_utils.hot_reload import HotReloader, reload_function
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="weekly", runner_config="cpu")

SOURCE = """
import functools

def traced(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return ("traced", fn(*args, **kwargs))
    return wrapper

class Base:
    def value(self):
        return "base"

class Pool(Base):
    def value(self):
        return "old " + super().value()

    @traced
    def size(self):
        return 1
"""


@pytest.fixture
def module(tmp_path, monkeypatch):
    (tmp_path / "hot_reload_sample.py").write_text(SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module("hot_reload_sample")
    yield mod, tmp_path / "hot_reload_sample.py"
    sys.modules.pop("hot_reload_sample", None)


def test_reload_function_updates_live_objects(module):
    mod, path = module
    pool = mod.Pool()
    path.write_text(SOURCE.replace('"old "', '"new "').replace("return 1", "return 2"))

    reload_function("hot_reload_sample.Pool.value")
    reload_function("hot_reload_sample.Pool.size")

    assert pool.value() == "new base", "super() still resolves"
    assert pool.size() == ("traced", 2), "the decorator is kept, its target reloaded"


def test_sigusr2_runs_the_script_at_the_next_poll(tmp_path):
    script = tmp_path / "patch.py"
    script.write_text("scheduler.seen = callable(reload_function)\n")
    previous = signal.getsignal(signal.SIGUSR2)
    try:
        reloader = HotReloader(str(script))
        scheduler = type("Scheduler", (), {})()
        reloader.poll(scheduler)
        assert not hasattr(scheduler, "seen"), "nothing runs without a signal"

        os.kill(os.getpid(), signal.SIGUSR2)
        reloader.poll(scheduler)
        assert scheduler.seen is True

        script.write_text("raise RuntimeError('bad patch')\n")
        os.kill(os.getpid(), signal.SIGUSR2)
        reloader.poll(scheduler)  # logged, never raised into the scheduler loop
    finally:
        signal.signal(signal.SIGUSR2, previous)
