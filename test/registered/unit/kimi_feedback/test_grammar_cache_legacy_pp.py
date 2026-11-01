"""The PP1 cache change must preserve PP's existing cache and one-way path."""

import datetime
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_grammar_cache_budget import Backend, SizedGrammar, manager, request
from test_k3_json_boundary import SCHEMA, Tokenizer

import sglang.srt.managers.schedule_batch as schedule_batch
from sglang.srt.constrained import base_grammar_backend as base_module
from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarBackend
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.parser.reasoning_parser import KimiK3Detector
from sglang.srt.runtime_context import get_parallel


@pytest.mark.parametrize("pp_size", [1, 2])
def test_direct_and_factory_select_identical_cache_policy(monkeypatch, pp_size):
    from sglang.srt.constrained import xgrammar_backend as native_module

    real_compiler = native_module.GrammarCompiler
    calls = []

    def compiler(**kwargs):
        calls.append(set(kwargs))
        return real_compiler(**kwargs)

    monkeypatch.setattr(native_module, "GrammarCompiler", compiler)
    monkeypatch.setattr(
        base_module,
        "get_exec",
        lambda: SimpleNamespace(kernel=SimpleNamespace(grammar_backend="xgrammar")),
    )
    monkeypatch.setattr(
        base_module,
        "get_serving",
        lambda: SimpleNamespace(
            constrained_json_disable_any_whitespace=False,
            constrained_json_max_whitespace_cnt=None,
            reasoning_parser=None,
        ),
    )
    tok = Tokenizer()
    with get_parallel().override(pp_size=pp_size):
        direct = XGrammarGrammarBackend(
            tok, tok.vocab_size, model_eos_token_ids=[tok.model_eos]
        )
        factory = base_module.create_grammar_backend(
            SimpleNamespace(), tok, tok.vocab_size, {tok.model_eos}
        )
        for obj in (direct, factory):
            assert obj.use_cache_budget is (pp_size == 1)
            assert isinstance(obj.cache, OrderedDict) is (pp_size == 1)
            obj.executor.shutdown(wait=True)
    expected = (
        {"tokenizer_info", "cache_limit_bytes", "max_threads"}
        if pp_size == 1
        else {"tokenizer_info"}
    )
    # Main also builds one uncached framing compiler per backend.
    assert [c for c in calls if "cache_enabled" not in c] == [expected, expected]


def test_pp_ignores_new_budget_env_and_keeps_negative_cache(monkeypatch):
    monkeypatch.setenv("SGLANG_GRAMMAR_CACHE_MAX_BYTES", "-1")
    monkeypatch.setenv("SGLANG_GRAMMAR_COMPILER_WORKERS", "0")
    monkeypatch.setenv("SGLANG_GRAMMAR_MAX_PENDING_COMPILES", "0")
    with get_parallel().override(pp_size=2):
        backend = Backend()
    try:
        assert not backend.use_cache_budget and not hasattr(backend, "_pending")
        assert backend.cache_size_bytes is None
        assert backend.executor._max_workers == min(32, (os.cpu_count() or 1) + 4)
        for i in range(40):
            backend.set_cache(("json", str(i)), SizedGrammar())
        bad = ("json", "unsupported")
        backend.set_cache(bad, InvalidGrammarObject("fixture"))
        assert len(backend.cache) == 41
        value, hit = backend.get_cached_or_future_value(bad, False)
        assert hit and isinstance(value, InvalidGrammarObject)
        assert not hasattr(backend, "_invalid_expirations")
        backend.reset()
        assert not backend.cache
    finally:
        backend.executor.shutdown(wait=True)


def test_pp_keeps_paired_thinking_contexts_and_wrapper_only_reset():
    tok = Tokenizer()
    with get_parallel().override(pp_size=2):
        native = XGrammarGrammarBackend(
            tok, tok.vocab_size, model_eos_token_ids=[tok.model_eos]
        )
        wrapper = ReasonerGrammarBackend(
            native, SimpleNamespace(detector=KimiK3Detector()), tok
        )
    try:
        key = ("json", json.dumps(SCHEMA))
        future, hit = wrapper.get_cached_or_future_value(key, False)
        assert not hit
        obj = future.result(5)
        assert not isinstance(obj, InvalidGrammarObject)
        assert (
            obj._kimik3_json_reasoning is None and len(obj._kimik3_json_grammars) == 2
        )
        wrapper.set_cache(key, obj.copy())
        first = obj.grammar.ctx
        alternate, hit = wrapper.get_cached_or_future_value(key, True)
        assert hit and alternate.grammar.ctx is not first
        again, hit = wrapper.get_cached_or_future_value(key, False)
        assert hit and again.grammar.ctx is first
        assert len(wrapper.cache) == 1 and wrapper.cache_key(key, True) == key
        assert native.grammar_compiler.get_cache_size_bytes() > 0
        wrapper.reset()
        assert not wrapper.cache and native.grammar_compiler.get_cache_size_bytes() > 0
    finally:
        wrapper.executor.shutdown(wait=True)
        native.executor.shutdown(wait=True)


class PPGroup:
    send_object = GroupCoordinator.send_object
    recv_object = GroupCoordinator.recv_object

    def __init__(self, group, ranks, rank):
        self.cpu_group, self.ranks = group, ranks
        self.world_size, self.rank_in_group = len(ranks), ranks.index(rank)


def _pp_rank(rank, rendezvous, metrics):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=30),
    )
    tp_group = pp_group = None
    for ranks in ([0, 1], [2, 3], [0, 2], [1, 3]):
        group = dist.new_group(ranks, backend="gloo")
        if rank in ranks:
            if ranks[1] - ranks[0] == 1:
                tp_group = group
            else:
                pp_group = PPGroup(group, ranks, rank)
    schedule_batch.get_parallel = lambda: SimpleNamespace(tp_rank=1)
    with get_parallel().override(pp_size=2, pp_rank=rank // 2):
        backend = Backend()
    mgr = manager(backend)
    mgr.pp_size, mgr.pp_rank = 2, rank // 2
    mgr.pp_group, mgr.grammar_sync_size, mgr.grammar_sync_group = pp_group, 2, tp_group
    assert not backend.use_cache_budget
    results = []

    def run(name, abort=False, timeout=False):
        req = request(name)
        assert mgr.process_req_with_grammar(req)
        if abort:
            mgr.abort_requests(SimpleNamespace(abort_all=True))
        if timeout:
            mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 1
        returned = []
        while mgr.grammar_queue:
            returned.extend(mgr.get_ready_grammar_requests())
        mgr._drain_pp_sync_work()
        assert returned == [req]
        status = req.to_finish.to_json()["status_code"] if req.to_finish else 200
        expected = 400 if name == "unsupported" or abort or timeout else 200
        assert status == expected, (rank, name, status)
        results.append((name, status))
        mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 1000
        dist.barrier()

    run("cold")
    req = request("cold")
    assert not mgr.process_req_with_grammar(req) and not mgr.grammar_queue
    assert isinstance(req.grammar, SizedGrammar)
    run("unsupported")
    req = request("unsupported")
    assert (
        not mgr.process_req_with_grammar(req)
        and req.to_finish.to_json()["status_code"] == 400
    )
    backend.gate.clear()
    run("cancel", abort=True)
    run("timeout", timeout=True)
    backend.gate.set()
    backend.executor.shutdown(wait=True)
    reports = [None] * 4
    dist.all_gather_object(
        reports,
        dict(
            rank=rank, outcomes=results, cache_entries=len(backend.cache), budget=False
        ),
    )
    if rank == 0:
        Path(metrics).write_text(json.dumps(reports, indent=2))
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo required")
def test_real_gloo_pp_preserves_cold_hot_cancel_timeout(tmp_path):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    metrics = (
        Path(os.environ.get("GRAMMAR_CACHE_EVIDENCE_DIR", str(tmp_path)))
        / "legacy-pp-metrics.json"
    )
    mp.start_processes(
        _pp_rank,
        args=(str(tmp_path / "rv-pp"), str(metrics)),
        nprocs=4,
        join=True,
        start_method="fork",
    )
    reports = json.loads(metrics.read_text())
    assert all(r["outcomes"] == reports[0]["outcomes"] for r in reports)
