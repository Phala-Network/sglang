"""Bounded cache ownership, admission, reset, and real abort-status regressions."""

import gc
import threading
import time
import weakref
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from collections import OrderedDict

import sglang.srt.managers.schedule_batch as schedule_batch
from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarBackend
from sglang.srt.managers.schedule_batch import Req


class SizedGrammar(BaseGrammarObject):
    def __init__(self, size=1000):
        super().__init__()
        self.size = size
        self.progress = []

    def cache_size_bytes(self):
        return self.size

    def copy(self):
        return SizedGrammar(self.size)


class Backend(BaseGrammarBackend):
    def __init__(self):
        super().__init__()
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()
        self.calls = []

    def dispatch_json(self, value):
        self.calls.append(value)
        self.entered.set()
        assert self.gate.wait(5), "test compiler gate was not released"
        if value == "unsupported":
            return InvalidGrammarObject("unsupported fixture")
        if value == "internal":
            raise RuntimeError("synthetic compiler fault")
        return SizedGrammar()


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("SGLANG_GRAMMAR_COMPILER_WORKERS", "1")
    obj = Backend()
    yield obj
    obj.gate.set()
    obj.reset()
    obj.executor.shutdown(wait=True, cancel_futures=True)


def manager(backend):
    mgr = object.__new__(GrammarManager)
    mgr.grammar_backend = backend
    mgr.grammar_queue = []
    mgr._enable_strict_thinking = False
    mgr.grammar_sync_size = 1
    mgr.grammar_sync_group = None
    mgr.pp_rank = 0
    mgr.pp_size = 1
    mgr.pp_group = None
    mgr.grammar_pp_sync_work_list = []
    mgr._synced_grammar_errors = {}
    mgr.SGLANG_GRAMMAR_POLL_INTERVAL = 0.005
    mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 1000
    return mgr


def request(schema="ok", thinking=False):
    req = SimpleNamespace(
        rid="grammar-budget-fixture",
        require_reasoning=thinking,
        sampling_params=SimpleNamespace(
            json_schema=schema,
            regex=None,
            ebnf=None,
            structural_tag=None,
            custom_params=None,
        ),
        grammar=None,
        grammar_key=None,
        grammar_wait_ct=0,
        to_finish=None,
        multimodal_inputs=None,
        session=None,
        # Main's grammar key may depend on how the prompt ends.
        origin_input_ids=[],
    )
    req.finished = lambda: req.to_finish is not None
    req.release_mm_inputs_on_abort = lambda: None  # called by main's abort
    # Exercise the product abort method, including its actual FINISH_ABORT object.
    req.set_finish_with_abort = lambda *a, **kw: Req.set_finish_with_abort(
        req, *a, **kw
    )
    return req


@pytest.fixture(autouse=True)
def abort_rank(monkeypatch):
    monkeypatch.setattr(
        schedule_batch, "get_parallel", lambda: SimpleNamespace(tp_rank=1)
    )


def drain(mgr):
    result = []
    for _ in range(100):
        result.extend(mgr.get_ready_grammar_requests())
        if not mgr.grammar_queue:
            return result
    raise AssertionError("grammar queue did not drain")


def test_lru_evicts_actual_owner_and_keeps_active_copy(backend):
    backend.cache_max_entries = 2
    a, b, c = (("json", x) for x in "abc")
    template = SizedGrammar()
    ref = weakref.ref(template)
    backend.set_cache(a, template)
    active, hit = backend.get_cached_or_future_value(a, False)
    assert hit
    backend.set_cache(b, SizedGrammar())
    backend.get_cached_or_future_value(a, False)
    backend.set_cache(c, SizedGrammar())
    assert list(backend.cache) == [a, c]
    del template
    backend.set_cache(b, SizedGrammar())
    gc.collect()
    assert ref() is None
    active.progress.append("still owned by request")
    assert active.progress == ["still owned by request"]


def test_bytes_limit_includes_schema_keys_and_oversized_values(backend):
    backend.cache_max_bytes = 1800
    for i in range(8):
        backend.set_cache(("json", str(i)), SizedGrammar())
    assert len(backend.cache) == 1
    assert backend.cache_size_bytes <= 1800
    backend.set_cache(("json", "large-value"), SizedGrammar(2000))
    backend.set_cache(("json", "x" * 2000), SizedGrammar(1))
    assert len(backend.cache) == 1


def test_invalid_budget_and_ttl(backend):
    backend.invalid_cache_max_entries = 2
    backend.invalid_cache_max_bytes = 1000
    backend.invalid_cache_ttl = 0.01
    for i in range(12):
        backend.set_cache(("json", str(i)), InvalidGrammarObject("unsupported"))
    assert len(backend.cache) <= 2
    assert backend._invalid_cache_bytes <= 1000
    key = next(iter(backend.cache))
    backend._invalid_expirations[key] = time.monotonic() - 1
    future, hit = backend.get_cached_or_future_value(key, False)
    assert not hit
    future.result(2)
    backend.release_future(future)
    assert key not in backend.cache


@pytest.mark.parametrize("status", [500, 503, 504])
def test_transient_service_failures_do_not_poison_negative_cache(backend, status):
    backend.set_cache(
        ("json", "x"), InvalidGrammarObject("temporary", status_code=status)
    )
    assert not backend.cache


def test_pending_limits_cover_running_queued_and_unconsumed_results(backend):
    backend.max_pending_compiles = 2
    backend.gate.clear()
    first, _ = backend.get_cached_or_future_value(("json", "first"), False)
    assert backend.entered.wait(2)
    second, _ = backend.get_cached_or_future_value(("json", "second"), False)
    rejected, hit = backend.get_cached_or_future_value(("json", "third"), False)
    assert not hit and rejected.result().status_code == 503
    backend.release_future(first)
    assert not first.cancel()
    assert len(backend._pending) == 2
    assert second.cancel()
    backend.release_future(second)
    # Cancelling a Future leaves its work item in ThreadPoolExecutor's queue.
    # Keep admission charged until the worker dequeues and skips that item.
    assert len(backend._pending) == 2
    backend.gate.set()
    first.result(2)
    backend.executor.shutdown(wait=True)
    assert not backend._pending and backend._pending_bytes == 0


def test_pending_schema_byte_limit_does_not_submit(backend):
    backend.max_pending_bytes = 100
    future, _ = backend.get_cached_or_future_value(("json", "x" * 200), False)
    assert future.result().status_code == 503
    assert not backend.calls and not backend._pending


def test_eight_same_cold_requests_compile_once_with_independent_results(backend):
    backend.gate.clear()
    items = [
        backend.get_cached_or_future_value(("json", "shared"), False)[0]
        for _ in range(8)
    ]
    assert backend.entered.wait(2)
    assert len(set(items)) == 8 and len(backend._inflight) == 1
    assert len(backend._pending) == 8 and backend.calls == ["shared"]
    backend.gate.set()
    values = [future.result(2) for future in items]
    assert len({id(value) for value in values}) == 8
    values[0].progress.append("request zero only")
    assert all(not value.progress for value in values[1:])
    for future in items:
        backend.release_future(future)
    assert not backend._pending and not backend._inflight


def test_cancel_one_queued_waiter_preserves_other_waiter(backend):
    backend.gate.clear()
    held, _ = backend.get_cached_or_future_value(("json", "held"), False)
    assert backend.entered.wait(2)
    a, _ = backend.get_cached_or_future_value(("json", "shared"), False)
    b, _ = backend.get_cached_or_future_value(("json", "shared"), False)
    assert a.cancel()
    backend.release_future(a)
    assert not b._grammar_source.cancelled()
    backend.gate.set()
    held.result(2)
    assert isinstance(b.result(2), SizedGrammar)
    backend.release_future(held)
    backend.release_future(b)
    assert backend.calls == ["held", "shared"]
    assert not backend._pending and not backend._inflight


def test_singleflight_waiters_still_obey_pending_capacity(backend):
    backend.max_pending_compiles = 2
    backend.gate.clear()
    items = [
        backend.get_cached_or_future_value(("json", "shared"), False)[0]
        for _ in range(3)
    ]
    assert items[2].result().status_code == 503
    assert len(backend._pending) == 2 and len(backend._inflight) == 1
    backend.gate.set()
    for future in items[:2]:
        future.result(2)
        backend.release_future(future)
    assert backend.calls == ["shared"] and not backend._pending


def test_cancelled_queue_flood_cannot_reuse_slots_before_dequeue(backend):
    backend.max_pending_compiles = 2
    backend.gate.clear()
    held, _ = backend.get_cached_or_future_value(("json", "held"), False)
    assert backend.entered.wait(2)
    queued, _ = backend.get_cached_or_future_value(("json", "queued"), False)
    assert queued.cancel()
    backend.release_future(queued)
    for i in range(100):
        rejected, _ = backend.get_cached_or_future_value(("json", f"flood-{i}"), False)
        assert rejected.result().status_code == 503
    assert len(backend._pending) == 2 and len(backend._inflight) == 2
    backend.gate.set()
    held.result(2)
    backend.release_future(held)
    backend.executor.shutdown(wait=True)
    assert backend.calls == ["held"]
    assert not backend._pending and not backend._inflight


def test_reset_keeps_old_singleflight_separate_from_new_generation(backend):
    backend.gate.clear()
    a, _ = backend.get_cached_or_future_value(("json", "shared"), False)
    assert backend.entered.wait(2)
    b, _ = backend.get_cached_or_future_value(("json", "shared"), False)
    backend.reset()
    new, _ = backend.get_cached_or_future_value(("json", "shared"), False)
    assert len(backend._inflight) == 2
    backend.gate.set()
    for future in (a, b):
        value = future.result(2)
        backend.set_cache(("json", "shared"), value, future=future)
    assert not backend.cache
    new.result(2)
    backend.release_future(new)
    assert backend.calls == ["shared", "shared"]
    assert not backend._pending and not backend._inflight


def test_aborting_one_running_waiter_keeps_shared_request_alive(backend):
    backend.gate.clear()
    mgr = manager(backend)
    a, b = request("shared"), request("shared")
    mgr.process_req_with_grammar(a)
    mgr.process_req_with_grammar(b)
    assert backend.entered.wait(2)
    a.set_finish_with_abort("external cancellation")
    assert mgr.get_ready_grammar_requests() == [a]
    assert len(backend._pending) == 2
    backend.gate.set()
    assert drain(mgr) == [b] and b.to_finish is None
    assert backend.calls == ["shared"]
    assert not backend._pending and not backend._inflight


def test_reset_rejects_late_cache_insertion_and_releases_budget(backend):
    backend.gate.clear()
    future, _ = backend.get_cached_or_future_value(("json", "old"), False)
    assert backend.entered.wait(2)
    backend.reset()
    backend.gate.set()
    value = future.result(2)
    backend.set_cache(("json", "old"), value, future=future)
    backend.executor.shutdown(wait=True)
    assert not backend.cache and backend.cache_size_bytes == 0
    assert not backend._pending and backend._pending_bytes == 0


def test_reasoner_reset_delegates_to_native(backend):
    wrapper = object.__new__(ReasonerGrammarBackend)
    BaseGrammarBackend.__init__(wrapper)
    wrapper.grammar_backend = backend
    wrapper._budget_variant_keys = OrderedDict()  # set by main's __init__
    backend.set_cache(("json", "inner"), SizedGrammar())
    wrapper.set_cache(("regex", "outer"), SizedGrammar())
    wrapper.reset()
    assert not backend.cache and not wrapper.cache
    wrapper.executor.shutdown(wait=True)


@pytest.mark.parametrize(
    "schema,status,kind",
    [
        ("unsupported", 400, "invalid_request_error"),
        ("internal", 500, "internal_server_error"),
    ],
)
def test_manager_cold_and_negative_cached_abort_metadata(backend, schema, status, kind):
    mgr = manager(backend)
    for _ in range(2):
        req = request(schema)
        assert mgr.process_req_with_grammar(req)
        assert isinstance(req.grammar, Future)
        assert drain(mgr) == [req]
        reason = req.to_finish.to_json()
        assert reason["status_code"] == status and reason["err_type"] == kind
    assert len(backend.calls) == (1 if status == 400 else 2)
    assert not backend._pending


def test_manager_hot_and_cold_use_same_ordered_sync_queue(backend):
    backend.set_cache(("json", "hot"), SizedGrammar())
    mgr = manager(backend)
    hot, cold = request("hot"), request("cold")
    assert mgr.process_req_with_grammar(hot)
    assert mgr.process_req_with_grammar(cold)
    assert mgr.grammar_queue == [hot, cold]
    assert drain(mgr) == [hot, cold]
    assert hot.grammar is not backend.cache[("json", "hot")]
    assert not backend._pending


def test_manager_capacity_503_and_timeout_504_are_not_schema_errors(backend):
    mgr = manager(backend)
    backend.max_pending_compiles = 1
    backend.gate.clear()
    first, rejected = request("slow"), request("capacity")
    mgr.process_req_with_grammar(first)
    mgr.process_req_with_grammar(rejected)
    mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 1
    ready = mgr.get_ready_grammar_requests()
    assert set(map(id, ready)) == {id(first), id(rejected)}
    assert first.to_finish.to_json()["status_code"] == 504
    assert rejected.to_finish.to_json()["status_code"] == 503
    assert not backend.cache
    backend.gate.set()


def test_abort_releases_future_even_after_req_grammar_cleared(backend):
    backend.gate.clear()
    mgr = manager(backend)
    req = request("aborted")
    mgr.process_req_with_grammar(req)
    req.set_finish_with_abort("external cancellation")
    drain(mgr)
    backend.gate.set()
    backend.executor.shutdown(wait=True)
    assert not backend._pending and backend._pending_bytes == 0


def test_remote_timeout_does_not_revisit_locally_aborted_grammar(backend, monkeypatch):
    backend.gate.clear()
    mgr = manager(backend)
    req = request("abort-timeout-overlap")
    mgr.process_req_with_grammar(req)
    req.set_finish_with_abort("external cancellation")
    mgr.grammar_sync_size = 2

    def gathered(values, local, group):
        values[:] = [local, (set(), {0}, {})]

    monkeypatch.setattr("torch.distributed.all_gather_object", gathered)
    assert mgr.get_ready_grammar_requests() == [req]
    assert req.to_finish.to_json()["message"] == "external cancellation"
    backend.gate.set()
    backend.executor.shutdown(wait=True)
    assert not backend._pending


def test_real_native_singleflight_copy_preserves_cold_and_hot_stats():
    from test_k3_json_boundary import Tokenizer

    from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend

    tok = Tokenizer()
    backend = XGrammarGrammarBackend(
        tok, tok.vocab_size, model_eos_token_ids=[tok.model_eos]
    )
    try:
        key = ("regex", "hello")
        future, hit = backend.get_cached_or_future_value(key, False)
        assert not hit
        cold = future.result(5)
        assert cold.grammar_stats is not None and not cold.grammar_stats.is_cache_hit
        assert not future._grammar_source.result().grammar_stats.is_cache_hit
        backend.set_cache(key, cold.copy(), False, future=future)
        backend.release_future(future)
        hot, hit = backend.get_cached_or_future_value(key, False)
        assert hit and hot.grammar_stats.is_cache_hit
        assert hot.grammar_stats is not cold.grammar_stats
        assert not cold.grammar_stats.is_cache_hit
    finally:
        backend.reset()
        backend.executor.shutdown(wait=True)


def test_real_native_compiles_only_requested_mode_and_cache_copy_is_isolated():
    import json

    from test_k3_json_boundary import SCHEMA, Tokenizer

    from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
    from sglang.srt.parser.reasoning_parser import KimiK3Detector

    tok = Tokenizer()
    native = XGrammarGrammarBackend(
        tok, tok.vocab_size, model_eos_token_ids=[tok.model_eos]
    )
    wrapper = ReasonerGrammarBackend(
        native, SimpleNamespace(detector=KimiK3Detector()), tok
    )
    key = ("json", json.dumps(SCHEMA))
    contexts = []
    for mode in (False, True, False, True):
        value, hit = wrapper.get_cached_or_future_value(key, mode)
        if not hit:
            future = value
            value = future.result(5)
            assert not isinstance(value, InvalidGrammarObject)
            wrapper.set_cache(key, value.copy(), mode, future=future)
            wrapper.release_future(future)
        contexts.append(value.grammar.ctx)
        assert hit == (len(contexts) > 2)
        assert value._kimik3_json_reasoning is mode
        assert not native.grammar_compiler.get_cache_size_bytes()
        assert not value.grammar.accepted_tokens
        if hit:
            assert (
                value.grammar.matcher
                is not wrapper.cache[wrapper.cache_key(key, mode)].grammar.matcher
            )
    assert contexts[0] is contexts[2] and contexts[1] is contexts[3]
    assert contexts[0] is not contexts[1]
    wrapper.reset()
    wrapper.executor.shutdown(wait=True)
    native.executor.shutdown(wait=True)


def test_real_native_active_matcher_survives_eviction_and_oversized_bypass():
    import json

    from test_k3_json_boundary import SCHEMA, Tokenizer, feed

    from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
    from sglang.srt.parser.reasoning_parser import KimiK3Detector

    tok = Tokenizer()
    native = XGrammarGrammarBackend(
        tok, tok.vocab_size, model_eos_token_ids=[tok.model_eos]
    )
    wrapper = ReasonerGrammarBackend(
        native, SimpleNamespace(detector=KimiK3Detector()), tok
    )
    wrapper.cache_max_entries = 1
    active = None
    for index in range(5):
        key = ("json", json.dumps(dict(SCHEMA, description=f"fixture-{index}")))
        future, hit = wrapper.get_cached_or_future_value(key, False)
        assert not hit
        obj = future.result(5)
        wrapper.set_cache(key, obj.copy(), False, future=future)
        wrapper.release_future(future)
        if active is None:
            active = obj
        assert (
            len(wrapper.cache) == 1
            and not native.grammar_compiler.get_cache_size_bytes()
        )
    assert active.grammar.ctx is not next(iter(wrapper.cache.values())).grammar.ctx
    feed(active, (tok, native), tok.encode('{"ok":true}', add_special_tokens=False))
    feed(active, (tok, native), [tok.model_eos])
    assert active.is_terminated()
    wrapper.reset()
    wrapper.cache_max_bytes = 1
    key = ("json", json.dumps(SCHEMA))
    future, _ = wrapper.get_cached_or_future_value(key, False)
    obj = future.result(5)
    wrapper.set_cache(key, obj.copy(), False, future=future)
    wrapper.release_future(future)
    assert not wrapper.cache
    feed(obj, (tok, native), tok.encode('{"ok":true}', add_special_tokens=False))
    feed(obj, (tok, native), [tok.model_eos])
    assert obj.is_terminated()
    wrapper.reset()
    wrapper.executor.shutdown(wait=True)
    native.executor.shutdown(wait=True)
