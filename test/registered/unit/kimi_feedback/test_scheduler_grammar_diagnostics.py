"""Execute the actual internal-state handler and serialization with CPU fixtures."""

import json
from types import SimpleNamespace

import responses_cpu_helpers  # noqa: F401
from test_grammar_cache_budget import SizedGrammar
from test_grammar_cache_budget import backend as backend

from sglang.srt.managers.io_struct import GetInternalStateReq
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def state(backend=None, tp_rank=0, tp_size=1):
    # The fork's main line reads internal state through this collaborator.
    scheduler = SimpleNamespace(
        grammar_manager=SimpleNamespace(grammar_backend=backend),
        metrics_reporter=SimpleNamespace(last_gen_throughput=1.0, step_time_dict={}),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(weight_load_mem_usage=1.0),
            graph_memory_usage=None,
        ),
        token_to_kv_pool_allocator=SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(mem_usage=3.0)
        ),
        startup_available_gpu_memory_gb=4.0,
        startup_time=1.0,
        max_total_num_tokens=100,
        swa_tokens_per_layer=None,
        max_running_requests=8,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        draft_worker=None,
        rust_server=None,
    )
    controller = SchedulerInternalStateController(scheduler, record_step_time=False)
    with (
        get_context().override_server_args(tp_size=tp_size),
        get_parallel().override(tp_rank=tp_rank),
    ):
        output = controller.get_internal_state(GetInternalStateReq())
    return output.internal_state


def test_actual_internal_state_preserves_capacity_and_exposes_only_numeric_budget(
    backend,
):
    backend.set_cache(("json", "synthetic-private-schema"), SizedGrammar(2048))
    result = state(backend, tp_rank=3, tp_size=8)
    snapshot = result["grammar_budget"]
    assert snapshot["available"] and snapshot["cache_entries"] == 1
    assert snapshot["approx_cache_bytes"] >= 2048
    assert snapshot["tp_rank"] == 3
    assert "synthetic-private-schema" not in json.dumps(snapshot)
    assert result["effective_max_running_requests_per_dp"] == 8
    assert result["memory_usage"]["token_capacity"] == 100


def test_disabled_backend_is_unknown_not_zero_capacity():
    snapshot = state()["grammar_budget"]
    assert not snapshot["available"] and snapshot["reason"] == "backend_disabled"
    assert "approx_cache_bytes" not in snapshot


def test_busy_snapshot_is_serializable_without_fabricated_zero(backend, monkeypatch):
    monkeypatch.setattr(
        backend, "budget_snapshot", lambda: {"available": False, "reason": "busy"}
    )
    snapshot = state(backend)["grammar_budget"]
    assert snapshot["reason"] == "busy" and not snapshot["available"]
    assert "approx_cache_bytes" not in snapshot
    json.dumps(snapshot, allow_nan=False)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
