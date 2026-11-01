"""Real CPU Gloo ranks exercise asymmetric cache state and compiler outcomes."""

import datetime
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_grammar_cache_budget import Backend, SizedGrammar, manager, request

import sglang.srt.managers.schedule_batch as schedule_batch


class PPGroup:
    def __init__(self, group, size):
        self.group, self.size = group, size

    def all_gather_object(self, value):
        values = [None] * self.size
        dist.all_gather_object(values, value, group=self.group)
        return values


def _rank(rank, world, tp_size, rendezvous, metrics):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=60),
    )
    pp_size = world // tp_size
    tp_group = None
    pp_group = None
    for stage in range(pp_size):
        group = dist.new_group(
            list(range(stage * tp_size, (stage + 1) * tp_size)), backend="gloo"
        )
        if rank // tp_size == stage:
            tp_group = group
    for lane in range(tp_size):
        group = dist.new_group(list(range(lane, world, tp_size)), backend="gloo")
        if rank % tp_size == lane:
            pp_group = group
    schedule_batch.get_parallel = lambda: SimpleNamespace(tp_rank=1)
    backend = Backend()
    mgr = manager(backend)
    mgr.grammar_sync_size = tp_size
    mgr.grammar_sync_group = tp_group
    mgr.pp_rank = rank // tp_size
    mgr.pp_size = pp_size
    mgr.pp_group = PPGroup(pp_group, pp_size)
    mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 50
    outcomes = []

    def run_batch(names, expected=200, cancel_rank=None):
        reqs = [request(name) for name in names]
        start = time.perf_counter()
        for req in reqs:
            assert mgr.process_req_with_grammar(req)
        if rank == cancel_rank:
            # Cancel a real queued executor Future behind an occupied worker.
            assert reqs[0].grammar.cancel()
        product_seconds = time.perf_counter() - start
        lengths = [None] * world
        dist.all_gather_object(lengths, len(mgr.grammar_queue))
        assert lengths == [len(reqs)] * world
        result = []
        while mgr.grammar_queue:
            start = time.perf_counter()
            returned = mgr.get_ready_grammar_requests()
            product_seconds += time.perf_counter() - start
            result.extend(returned)
            states = [None] * world
            dist.all_gather_object(states, len(mgr.grammar_queue))
            assert len(set(states)) == 1, states
        statuses = [
            r.to_finish.to_json()["status_code"] if r.to_finish else 200 for r in reqs
        ]
        all_statuses = [None] * world
        dist.all_gather_object(all_statuses, statuses)
        assert all_statuses == [statuses] * world
        assert statuses == (
            [expected] * len(reqs) if isinstance(expected, int) else expected
        )
        assert len(result) == len(reqs)
        outcomes.append(statuses)
        return product_seconds

    # Deliberately unequal positive LRU state and hit/miss/eviction order.
    backend.cache_max_entries = 1 if rank % 2 else 4
    if rank % 2:
        backend.set_cache(("json", "shared"), SizedGrammar())
    run_batch(["shared", "other", "shared"])
    run_batch(["other", "shared"])

    # Different negative cache ages; expired rank compiles while others hit.
    run_batch(["unsupported"], 400)
    if rank % 2:
        backend._invalid_expirations[("json", "unsupported")] = time.monotonic() - 1
    run_batch(["unsupported"], 400)

    # Only one rank has exhausted local capacity; every TP/PP rank must abort.
    backend.gate.clear()
    held = None
    if rank == world - 1:
        backend.max_pending_compiles = 1
        held, _ = backend.get_cached_or_future_value(("json", "held"), False)
        assert backend.entered.wait(2)
    else:
        backend.gate.set()
    run_batch(["capacity-asymmetry"], 503)
    backend.gate.set()
    if held is not None:
        held.result(5)
        backend.release_future(held)
    backend.max_pending_compiles = 32
    dist.barrier()

    # A queued future cancelled on only one rank propagates one typed failure.
    # Other ranks may already have a usable native template when this occurs.
    if rank == world - 1:
        backend.executor.shutdown(wait=True)
        from concurrent.futures import ThreadPoolExecutor

        backend.executor = ThreadPoolExecutor(max_workers=1)
        backend.gate.clear()
        backend.entered.clear()
        held, _ = backend.get_cached_or_future_value(("json", "cancel-held"), False)
        assert backend.entered.wait(2)
    run_batch(["cancel-asymmetry"], 503, cancel_rank=world - 1)
    if rank == world - 1:
        backend.gate.set()
        held.result(5)
        backend.release_future(held)
    dist.barrier()

    # A late PP stage / TP lane times out while the others have ready objects.
    if rank == world - 1:
        backend.gate.clear()
    mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 1
    run_batch(["timeout-asymmetry"], 504)
    backend.gate.set()
    mgr.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 50
    backend.executor.shutdown(wait=True)
    from concurrent.futures import ThreadPoolExecutor

    backend.executor = ThreadPoolExecutor(max_workers=2)
    dist.barrier()

    # Reset while work is running keeps active ownership but never refills old gen.
    backend.gate.clear()
    backend.entered.clear()
    req = request("reset-asymmetry")
    mgr.process_req_with_grammar(req)
    assert backend.entered.wait(2)
    mgr.clear()
    backend.gate.set()
    while mgr.grammar_queue:
        mgr.get_ready_grammar_requests()
    assert not backend.cache and not backend._pending
    assert req.to_finish is None

    # Measure real batched hot admission synchronization, separately from masks.
    samples = []
    backend.cache_max_entries = 32
    run_batch(["hot"] * 8)
    for _ in range(12):
        samples.append(run_batch(["hot"] * 8))
    reports = [None] * world
    dist.all_gather_object(
        reports,
        dict(
            rank=rank,
            hot_batch_median_seconds=statistics.median(samples),
            pending=len(backend._pending),
            cache_bytes=backend.cache_size_bytes,
        ),
    )
    if rank == 0:
        Path(metrics).write_text(
            json.dumps(
                dict(tp=tp_size, pp=pp_size, cases=outcomes, ranks=reports), indent=2
            )
        )
    backend.reset()
    backend.executor.shutdown(wait=True)
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo required")
@pytest.mark.parametrize("tp,pp", [(8, 1)])
def test_real_gloo_cache_and_error_asymmetry(tmp_path, tp, pp):
    # fork shares imported CPU libraries; eight independent spawn imports would
    # exceed this functional test's memory budget. No CUDA context is initialized.
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    metrics = (
        Path(os.environ.get("GRAMMAR_CACHE_EVIDENCE_DIR", str(tmp_path)))
        / f"sync-{tp}x{pp}-metrics.json"
    )
    mp.start_processes(
        _rank,
        args=(tp * pp, tp, str(tmp_path / "rendezvous"), str(metrics)),
        nprocs=tp * pp,
        join=True,
        start_method="fork",
    )
    result = json.loads(metrics.read_text())
    assert all(r["pending"] == 0 for r in result["ranks"])
