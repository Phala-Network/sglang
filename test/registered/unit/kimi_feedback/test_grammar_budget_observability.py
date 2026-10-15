"""Fixed-size observations must not change ownership, block, or expose schemas."""
import responses_cpu_helpers
import json
import threading
from unittest.mock import patch

import pytest
from test_grammar_cache_budget import Backend, backend
from sglang.srt.constrained.base_grammar_backend import BaseGrammarBackend


def test_singleflight_counts_compiles_not_waiters_and_reset_preserves_totals(backend):
    secret = 'synthetic-private-schema'
    key = ('json', secret)
    backend.gate.clear()
    futures = [backend.get_cached_or_future_value(key, False)[0] for _ in range(8)]
    assert backend.entered.wait(2)
    current = backend.budget_snapshot()
    assert current['available'] and current['compiles_started'] == 1
    assert current['compiles_finished'] == 0 and current['pending_requests'] == 8
    assert current['cache_lookup_misses'] == 8
    assert secret not in json.dumps(current)
    backend.gate.set()
    for future in futures:
        value = future.result(2)
        backend.set_cache(key, value.copy(), future=future)
        backend.release_future(future)
    value, hit = backend.get_cached_or_future_value(key, False)
    assert hit
    before = backend.budget_snapshot()
    assert before['cache_lookup_hits'] == 1
    assert before['compiles_started'] == before['compiles_finished'] == 1
    assert before['pending_requests'] == before['inflight_compiles'] == 0
    assert before['compile_wall_seconds'] >= 0 and before['compile_queue_seconds'] >= 0
    assert before['cache_entries'] == 1 and before['approx_cache_bytes'] > 0
    backend.reset()
    after = backend.budget_snapshot()
    assert after['cache_entries'] == after['approx_cache_bytes'] == 0
    assert after['cache_generation'] == before['cache_generation'] + 1
    for name in ('cache_lookup_hits','cache_lookup_misses','compiles_started','compiles_finished','compile_wall_seconds','compile_queue_seconds'):
        assert after[name] == before[name]
    assert value is not None


def test_capacity_rejection_does_not_count_as_compile(backend):
    backend.max_pending_bytes = 1
    future, hit = backend.get_cached_or_future_value(('json', 'synthetic'), False)
    assert not hit and future.result(2).status_code == 503
    snapshot = backend.budget_snapshot()
    assert snapshot['capacity_rejections'] == snapshot['cache_lookup_misses'] == 1
    assert snapshot['compiles_started'] == snapshot['compiles_finished'] == 0
    assert snapshot['pending_requests'] == 0


def test_compile_exception_finishes_counter_without_changing_failure(backend):
    future, hit = backend.get_cached_or_future_value(('json', 'internal'), False)
    with pytest.raises(RuntimeError, match='synthetic compiler fault'):
        future.result(2)
    backend.release_future(future)
    snapshot = backend.budget_snapshot()
    assert snapshot['compiles_started'] == snapshot['compiles_finished'] == 1
    assert snapshot['pending_requests'] == snapshot['inflight_compiles'] == 0


def test_snapshot_does_not_wait_for_worker_lock(backend):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def hold():
        with backend._cache_lock:
            entered.set()
            assert release.wait(3)
    holder=threading.Thread(target=hold)
    holder.start()
    assert entered.wait(2)
    rows=[]
    def read():
        rows.append(backend.budget_snapshot())
        finished.set()
    reader=threading.Thread(target=read)
    reader.start()
    try:
        assert finished.wait(0.5), 'snapshot waited for a compile lock'
        assert rows == [{'available':False,'reason':'busy'}]
    finally:
        release.set()
        holder.join(2)
        reader.join(2)
    assert backend.budget_snapshot()['available']


def test_preserved_pp_path_reports_unknown_without_creating_new_budget():
    with patch('sglang.srt.constrained.base_grammar_backend.grammar_cache_budget_enabled', return_value=False):
        backend=BaseGrammarBackend()
    try:
        assert backend.budget_snapshot()=={'available':False,'reason':'budget_disabled'}
        assert not hasattr(backend,'_budget_observations')
    finally:
        backend.executor.shutdown(wait=True)
