"""Cold compile results must not wait for unrelated native cache cleanup."""

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarBackend
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class StopCompile(BaseException):
    pass


class ObservedCondition(threading.Condition):
    def __init__(self):
        super().__init__()
        self.wait_entered = threading.Event()

    def wait(self, timeout=None):
        self.wait_entered.set()
        return super().wait(timeout)


class Value(BaseGrammarObject):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def copy(self):
        return Value(self.key)


class Compiler:
    """A gated native cache, including zero-charge failed-future ownership."""

    def __init__(self):
        self.condition = threading.Condition()
        self.active = set()
        self.completed = set()
        self.failed = set()
        self.entered = {}
        self.clear_calls = 0
        self.clear_entered = threading.Event()
        self.clear_gate = threading.Event()
        self.clear_gate.set()
        self.clear_error = None

    def event(self, key):
        with self.condition:
            return self.entered.setdefault(key, threading.Event())

    def compile(self, key, gate=None, error=None):
        with self.condition:
            self.active.add(key)
            self.event(key).set()
        try:
            if gate is not None:
                assert gate.wait(5), "fixture compiler was not released"
            if error is not None:
                raise error
            return key
        finally:
            with self.condition:
                self.active.remove(key)
                self.completed.add(key)
                if error is not None:
                    self.failed.add(key)
                self.condition.notify_all()

    def clear_cache(self):
        # The production native LRU waits for in-flight futures during clear.
        with self.condition:
            self.clear_calls += 1
            self.clear_entered.set()
            assert self.condition.wait_for(lambda: not self.active, timeout=5)
        assert self.clear_gate.wait(5), "fixture clear was not released"
        with self.condition:
            if self.clear_error is not None:
                error, self.clear_error = self.clear_error, None
                raise error
            self.completed.clear()
            self.failed.clear()


@pytest.fixture
def backend():
    obj = object.__new__(XGrammarGrammarBackend)
    obj.use_cache_budget = True
    obj._compile_condition = ObservedCondition()
    obj._compile_local = threading.local()
    obj._active_native_compiles = 0
    obj._native_clear_pending = False
    obj.grammar_compiler = Compiler()
    obj.any_whitespace = True
    obj.max_whitespace_cnt = None
    return obj


@pytest.fixture
def async_backend(backend, monkeypatch):
    monkeypatch.setenv("SGLANG_GRAMMAR_COMPILER_WORKERS", "2")
    BaseGrammarBackend.__init__(backend)
    backend.gates = {}

    def dispatch(key):
        return Value(
            backend._compile(
                backend.grammar_compiler.compile, key, backend.gates.get(key)
            )
        )

    backend.dispatch_regex = backend.dispatch_ebnf = dispatch
    yield backend
    for gate in backend.gates.values():
        gate.set()
    backend.grammar_compiler.clear_gate.set()
    backend.executor.shutdown(wait=True, cancel_futures=True)


def test_short_result_returns_while_unrelated_compile_is_active(backend):
    compiler = backend.grammar_compiler
    gate = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        slow = pool.submit(backend._compile, compiler.compile, "slow", gate)
        assert compiler.event("slow").wait(2)
        fast = pool.submit(backend._compile, compiler.compile, "fast")
        try:
            assert fast.result(0.5) == "fast"
            assert not slow.done()
            assert not compiler.clear_calls
        finally:
            gate.set()
        assert slow.result(2) == "slow"
    assert compiler.clear_calls == 1
    assert not compiler.completed and not compiler.failed
    assert not backend._active_native_compiles and not backend._native_clear_pending


def test_new_cold_work_waits_for_cleanup_instead_of_growing_failed_cache(backend):
    compiler = backend.grammar_compiler
    gate = threading.Event()
    with ThreadPoolExecutor(max_workers=3) as pool:
        slow = pool.submit(backend._compile, compiler.compile, "slow", gate)
        assert compiler.event("slow").wait(2)
        failed = pool.submit(
            backend._compile,
            compiler.compile,
            "bad-first",
            None,
            RuntimeError("fixture"),
        )
        try:
            with pytest.raises(RuntimeError, match="fixture"):
                failed.result(0.5)
            later = pool.submit(
                backend._compile,
                compiler.compile,
                "bad-next",
                None,
                RuntimeError("fixture"),
            )
            assert not compiler.event("bad-next").wait(0.1)
            assert compiler.failed == {"bad-first"}
            assert not later.done()
        finally:
            gate.set()
        assert slow.result(2) == "slow"
        with pytest.raises(RuntimeError, match="fixture"):
            later.result(2)
    assert not compiler.completed and not compiler.failed
    assert not backend._active_native_compiles and not backend._native_clear_pending


def test_new_compile_cannot_enter_during_last_clear(backend):
    compiler = backend.grammar_compiler
    compiler.clear_gate.clear()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(backend._compile, compiler.compile, "first")
        assert compiler.clear_entered.wait(2)
        second = pool.submit(backend._compile, compiler.compile, "second")
        try:
            assert not compiler.event("second").wait(0.1)
        finally:
            compiler.clear_gate.set()
        assert first.result(2) == "first"
        assert second.result(2) == "second"
    assert compiler.clear_calls == 2 and not compiler.completed


@pytest.mark.parametrize("error_type", [RuntimeError, StopCompile])
def test_compile_exceptions_always_release_cleanup_state(backend, error_type):
    compiler = backend.grammar_compiler
    with pytest.raises(error_type):
        backend._compile(compiler.compile, "bad", None, error_type("fixture"))
    assert not backend._active_native_compiles and not backend._native_clear_pending
    assert not compiler.completed and not compiler.failed
    assert backend._compile(compiler.compile, "next") == "next"


@pytest.mark.parametrize("error_type", [RuntimeError, StopCompile])
def test_clear_exception_wakes_next_compile_and_does_not_close_gate(
    backend, error_type
):
    compiler = backend.grammar_compiler
    gate = threading.Event()
    with ThreadPoolExecutor(max_workers=3) as pool:
        slow = pool.submit(backend._compile, compiler.compile, "slow", gate)
        assert compiler.event("slow").wait(2)
        first = pool.submit(backend._compile, compiler.compile, "first")
        assert first.result(0.5) == "first"
        compiler.clear_error = error_type("clear fixture")
        later = pool.submit(backend._compile, compiler.compile, "later")
        try:
            assert not compiler.event("later").wait(0.1)
        finally:
            gate.set()
        with pytest.raises(error_type, match="clear fixture"):
            slow.result(2)
        assert later.result(2) == "later"
    assert not backend._active_native_compiles and not backend._native_clear_pending
    assert not compiler.completed and not compiler.failed


def test_legacy_nonbudget_compile_has_no_cleanup_gate():
    obj = object.__new__(XGrammarGrammarBackend)
    obj.use_cache_budget = False
    assert obj._compile(lambda: "legacy") == "legacy"


@pytest.mark.parametrize("compile_error", [RuntimeError, StopCompile])
@pytest.mark.parametrize("clear_error", [RuntimeError, StopCompile])
def test_cleanup_error_preserves_original_compile_failure(
    backend, compile_error, clear_error
):
    compiler = backend.grammar_compiler
    compiler.clear_error = clear_error("cleanup fixture")
    with pytest.raises(compile_error, match="compile fixture"):
        backend._compile(
            compiler.compile, "bad", None, compile_error("compile fixture")
        )
    assert not backend._active_native_compiles and not backend._native_clear_pending
    assert backend._compile(compiler.compile, "next") == "next"
    assert not compiler.completed and not compiler.failed


def test_reservation_without_native_work_does_not_close_gate(backend):
    compiler = backend.grammar_compiler
    gate = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        slow = pool.submit(backend._compile, compiler.compile, "slow", gate)
        assert compiler.event("slow").wait(2)
        try:
            with backend._compile_context():
                pass
            assert not backend._native_clear_pending and not compiler.clear_calls
            fast = pool.submit(backend._compile, compiler.compile, "fast")
            assert fast.result(0.5) == "fast"
        finally:
            gate.set()
        assert slow.result(2) == "slow"
    assert compiler.clear_calls == 1


@pytest.mark.parametrize("use_reasoner", [False, True])
@pytest.mark.parametrize("keep_one_waiter", [False, True])
def test_gate_wait_rechecks_live_singleflight_waiters(
    async_backend, use_reasoner, keep_one_waiter
):
    backend = async_backend
    compiler = backend.grammar_compiler
    outer = backend
    key_type = "regex"
    if use_reasoner:
        outer = object.__new__(ReasonerGrammarBackend)
        BaseGrammarBackend.__init__(outer)
        outer.grammar_backend = backend
        outer._kimik3_json_tokenizer = None
        outer._budget_variant_keys = OrderedDict()  # set by main's __init__
        key_type = "full_assistant_ebnf"
    gate = backend.gates["slow"] = threading.Event()
    try:
        slow, _ = outer.get_cached_or_future_value((key_type, "slow"), False)
        assert compiler.event("slow").wait(2)
        fast, _ = outer.get_cached_or_future_value((key_type, "fast"), False)
        assert fast.result(0.5).key == "fast"
        outer.release_future(fast)
        first, _ = outer.get_cached_or_future_value((key_type, "later"), False)
        assert backend._compile_condition.wait_entered.wait(2)
        second, _ = outer.get_cached_or_future_value((key_type, "later"), False)
        assert first._grammar_source is second._grammar_source
        first.cancel()
        outer.release_future(first)
        if not keep_one_waiter:
            second.cancel()
            outer.release_future(second)
        snapshot = outer.budget_snapshot()
        assert snapshot["compiles_started"] == 2
        assert snapshot["compiles_finished"] == 1
        assert not compiler.event("later").is_set()
        gate.set()
        slow._grammar_source.result(2)
        first._grammar_source.result(2)
        outer.release_future(slow)
        if keep_one_waiter:
            assert second.result(2).key == "later"
            outer.release_future(second)
        outer.executor.shutdown(wait=True)
        assert compiler.event("later").is_set() == keep_one_waiter
        snapshot = outer.budget_snapshot()
        assert (
            snapshot["compiles_started"]
            == snapshot["compiles_finished"]
            == (3 if keep_one_waiter else 2)
        )
        assert snapshot["pending_requests"] == snapshot["inflight_compiles"] == 0
        assert compiler.clear_calls == (2 if keep_one_waiter else 1)
        assert not backend._active_native_compiles and not backend._native_clear_pending
    finally:
        gate.set()
        if use_reasoner:
            outer.executor.shutdown(wait=True, cancel_futures=True)


def test_direct_dispatch_and_async_worker_share_cleanup_admission(async_backend):
    backend = async_backend
    compiler = backend.grammar_compiler
    gate = backend.gates["slow"] = threading.Event()
    slow, _ = backend.get_cached_or_future_value(("regex", "slow"), False)
    assert compiler.event("slow").wait(2)
    try:
        assert backend.dispatch_regex("direct").key == "direct"
        later, _ = backend.get_cached_or_future_value(("regex", "later"), False)
        assert backend._compile_condition.wait_entered.wait(2)
        assert not compiler.event("later").is_set()
    finally:
        gate.set()
    assert slow.result(2).key == "slow"
    assert later.result(2).key == "later"
    backend.release_future(slow)
    backend.release_future(later)
    assert not backend._active_native_compiles and not backend._native_clear_pending


def test_already_abandoned_queue_item_skips_closed_gate_immediately(async_backend):
    backend = async_backend
    compiler = backend.grammar_compiler
    slow_gate = backend.gates["slow"] = threading.Event()
    dequeued = threading.Event()
    worker_gate = threading.Event()

    def hold_worker():
        dequeued.set()
        assert worker_gate.wait(5)

    try:
        slow, _ = backend.get_cached_or_future_value(("regex", "slow"), False)
        assert compiler.event("slow").wait(2)
        fast, _ = backend.get_cached_or_future_value(("regex", "fast"), False)
        assert fast.result(0.5).key == "fast"
        backend.release_future(fast)
        held_worker = backend.executor.submit(hold_worker)
        assert dequeued.wait(2)
        abandoned, _ = backend.get_cached_or_future_value(("regex", "abandoned"), False)
        assert abandoned.cancel()
        backend.release_future(abandoned)
        worker_gate.set()
        held_worker.result(0.5)
        assert abandoned._grammar_source.result(0.5) is None
        # A marker on the only available worker also waits for source callbacks.
        backend.executor.submit(lambda: None).result(0.5)
        assert not backend._compile_condition.wait_entered.is_set()
        assert not compiler.event("abandoned").is_set() and not compiler.clear_calls
        snapshot = backend.budget_snapshot()
        assert snapshot["pending_requests"] == snapshot["inflight_compiles"] == 1
        assert snapshot["compiles_started"] == 2
    finally:
        worker_gate.set()
        slow_gate.set()
    slow.result(2)
    backend.release_future(slow)
    assert not backend._active_native_compiles and not backend._native_clear_pending


def test_reset_during_active_compile_and_admission_wait_does_not_deadlock(
    async_backend,
):
    backend = async_backend
    compiler = backend.grammar_compiler
    gate = backend.gates["slow"] = threading.Event()
    slow, _ = backend.get_cached_or_future_value(("regex", "slow"), False)
    assert compiler.event("slow").wait(2)
    fast, _ = backend.get_cached_or_future_value(("regex", "fast"), False)
    assert fast.result(0.5).key == "fast"
    backend.release_future(fast)
    abandoned, _ = backend.get_cached_or_future_value(("regex", "abandoned"), False)
    assert backend._compile_condition.wait_entered.wait(2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        reset = pool.submit(backend.reset)
        try:
            assert compiler.clear_entered.wait(2)
            new, _ = backend.get_cached_or_future_value(
                ("regex", "new-generation"), False
            )
        finally:
            gate.set()
        reset.result(2)
    slow._grammar_source.result(2)
    abandoned._grammar_source.result(2)
    assert new.result(2).key == "new-generation"
    backend.release_future(new)
    backend.executor.shutdown(wait=True)
    assert not compiler.event("abandoned").is_set()
    snapshot = backend.budget_snapshot()
    assert snapshot["cache_generation"] == 1
    assert snapshot["pending_requests"] == snapshot["inflight_compiles"] == 0
    assert not backend._active_native_compiles and not backend._native_clear_pending
    assert not compiler.completed and not compiler.failed


@pytest.mark.parametrize("kind", ["json", "structural_tag"])
def test_cleanup_runtime_error_becomes_manager_500_and_next_request_recovers(
    async_backend, kind
):
    backend = async_backend
    compiler = backend.grammar_compiler
    compiler.compile_json_schema = lambda **kwargs: compiler.compile("json")
    compiler.compile_structural_tag = lambda *args: compiler.compile("tag")
    backend._from_context = lambda ctx, key, stats: Value(key)
    key = (
        kind,
        "{}"
        if kind == "json"
        else '{"format":{"type":"json_schema","json_schema":{}}}',
    )
    compiler.clear_error = RuntimeError("cleanup fixture")
    failed, _ = backend.get_cached_or_future_value(key, False)
    with pytest.raises(RuntimeError, match="cleanup fixture"):
        failed.result(2)
    value = GrammarManager._future_grammar_error(failed)
    assert isinstance(value, InvalidGrammarObject)
    assert value.status_code == 500 and value.error_type == "internal_server_error"
    backend.set_cache(key, value, future=failed)
    assert not backend.cache
    manager = object.__new__(GrammarManager)
    manager.grammar_backend = backend
    manager._release_request_future(SimpleNamespace(grammar=failed))
    following, _ = backend.get_cached_or_future_value(key, False)
    assert isinstance(following.result(2), Value)
    manager._release_request_future(SimpleNamespace(grammar=following))
    snapshot = backend.budget_snapshot()
    assert snapshot["pending_requests"] == snapshot["inflight_compiles"] == 0
    assert not backend._active_native_compiles and not backend._native_clear_pending


def test_handled_native_failure_is_not_replaced_by_outer_cleanup_failure(async_backend):
    backend = async_backend
    compiler = backend.grammar_compiler
    compiler.compile_json_schema = lambda **kwargs: compiler.compile(
        "json", error=RuntimeError("native fixture")
    )
    compiler.clear_error = RuntimeError("cleanup fixture")
    failed, _ = backend.get_cached_or_future_value(("json", "{}"), False)
    value = failed.result(2)
    assert isinstance(value, InvalidGrammarObject)
    assert value.error_message == "Grammar compiler failed (RuntimeError)"
    backend.release_future(failed)
    assert not backend._pending and not backend._inflight
    assert backend._compile(lambda: "next") == "next"


@pytest.mark.parametrize("skip_stage", ["before_gate", "after_gate"])
@pytest.mark.parametrize("new_compile_fails", [False, True])
def test_late_join_gets_new_flight_and_old_callback_preserves_its_owner(
    async_backend, skip_stage, new_compile_fails
):
    backend = async_backend
    compiler = backend.grammar_compiler
    skipped = threading.Event()
    old_return_gate = threading.Event()
    new_native_gate = backend.gates["shared"] = threading.Event()
    worker_gates = [threading.Event(), threading.Event()]
    worker_entered = [threading.Event(), threading.Event()]
    slow_gate = backend.gates["slow"] = threading.Event()
    original_run = backend._run_compile
    first = [True]

    def run(flight_key, key, reasoning):
        is_first = key[1] == "shared" and first[0]
        if is_first:
            first[0] = False
        value = original_run(flight_key, key, reasoning)
        if is_first:
            assert value is None
            skipped.set()
            assert old_return_gate.wait(5)
        return value

    def hold_worker(index):
        worker_entered[index].set()
        assert worker_gates[index].wait(5)

    def dispatch(key):
        error = (
            RuntimeError("new compile fixture")
            if key == "shared" and new_compile_fails
            else None
        )
        return Value(
            backend._compile(compiler.compile, key, backend.gates.get(key), error)
        )

    backend._run_compile = run
    backend.dispatch_regex = dispatch
    try:
        if skip_stage == "before_gate":
            blockers = [
                backend.executor.submit(hold_worker, index) for index in range(2)
            ]
            assert all(event.wait(2) for event in worker_entered)
        else:
            slow, _ = backend.get_cached_or_future_value(("regex", "slow"), False)
            assert compiler.event("slow").wait(2)
            fast, _ = backend.get_cached_or_future_value(("regex", "fast"), False)
            assert fast.result(0.5).key == "fast"
            backend.release_future(fast)
        old, _ = backend.get_cached_or_future_value(("regex", "shared"), False)
        if skip_stage == "after_gate":
            assert backend._compile_condition.wait_entered.wait(2)
        old.cancel()
        backend.release_future(old)
        if skip_stage == "before_gate":
            for gate in worker_gates:
                gate.set()
            for blocker in blockers:
                blocker.result(2)
        else:
            slow_gate.set()
            slow.result(2)
            backend.release_future(slow)
        assert skipped.wait(2)
        new, _ = backend.get_cached_or_future_value(("regex", "shared"), False)
        assert new._grammar_source is not old._grammar_source
        assert compiler.event("shared").wait(2)
        old_return_gate.set()
        assert old._grammar_source.result(2) is None
        backend.executor.submit(lambda: None).result(2)
        flight = backend._inflight[new._grammar_flight_key]
        assert flight[0] is new._grammar_source and not new.done()
        assert old not in backend._pending
        assert not hasattr(old._grammar_source, "_grammar_waiters")
        new_native_gate.set()
        if new_compile_fails:
            with pytest.raises(RuntimeError, match="new compile fixture"):
                new.result(2)
        else:
            assert new.result(2).key == "shared"
        backend.release_future(new)
        backend.executor.shutdown(wait=True)
        snapshot = backend.budget_snapshot()
        assert snapshot["pending_requests"] == snapshot["inflight_compiles"] == 0
        assert (
            snapshot["compiles_started"]
            == snapshot["compiles_finished"]
            == (1 if skip_stage == "before_gate" else 3)
        )
        assert not hasattr(new._grammar_source, "_grammar_waiters")
    finally:
        old_return_gate.set()
        new_native_gate.set()
        slow_gate.set()
        for gate in worker_gates:
            gate.set()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
