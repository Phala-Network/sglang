"""CPU metadata fixtures and execution of the real scheduler startup method."""

import ast
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.managers.scheduler_components import startup_spec_identity as obs


def runner(layers, quantization):
    weight = torch.empty((2, 3), dtype=torch.bfloat16)
    method = type("UnquantizedLinearMethod", (), {})()
    head = NS(weight=weight, quant_method=method)
    embed = NS(weight=torch.empty_like(weight), quant_method=method)
    pool = NS(
        dtype=torch.float8_e4m3fn,
        store_dtype=torch.uint8,
        size=64,
        page_size=64,
        layer_num=layers,
        start_layer=0,
        end_layer=layers - 1,
        kv_cache_dim=6,
        post_capture_active=True,
        kv_buffer=[torch.empty((128, 1, 6), dtype=torch.uint8) for _ in range(layers)],
    )
    quant = type("ModelOptFp4Config", (), {})() if quantization else None
    model = NS(
        lm_head=head,
        quant_config=quant,
        model=NS(embed_tokens=embed, quant_config=None),
    )
    return NS(
        model=model,
        kv_cache_dtype=torch.float8_e4m3fn,
        max_total_num_tokens=64,
        token_to_kv_pool=pool,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
        model_config=NS(
            quantization=quantization,
            hf_config=NS(architectures=["Xing4_0ForCausalLM"]),
        ),
    )


def scheduler():
    target, draft = runner(40, "modelopt_fp4"), runner(1, None)
    draft.req_to_token_pool = target.req_to_token_pool
    draft.token_to_kv_pool_allocator = target.token_to_kv_pool_allocator
    draft.model.lm_head.weight = target.model.lm_head.weight
    draft.model.model.embed_tokens.weight = target.model.model.embed_tokens.weight
    return NS(
        tp_worker=NS(model_runner=target),
        draft_worker=NS(draft_runner=draft),
        spec_algorithm=NS(name="EAGLE", is_none=lambda: False),
    )


def test_actual_tiny_tensor_metadata_and_shared_objects(caplog):
    s = scheduler()
    with caplog.at_level("INFO", logger=obs.__name__):
        record = obs.emit_xing_spec_startup_identity(s)
    assert record["target"]["pool"]["logical_dtype"] == "torch.float8_e4m3fn"
    assert record["draft"]["pool"]["storage_dtype"] == "torch.uint8"
    assert record["target"]["pool"]["buffers"]["kv_buffer"]["count"] == 40
    assert record["draft"]["pool"]["buffers"]["kv_buffer"] == {
        "count": 1,
        "tensors": [{"dtype": "torch.uint8", "shape": [128, 1, 6]}],
    }
    assert record["draft"]["head"]["quant_method_class"] == "UnquantizedLinearMethod"
    assert record["target"]["quant_config_class"] == "ModelOptFp4Config"
    assert record["draft"]["quant_config_class"] is None
    assert record["sharing"] == dict(
        kv_pool_same_object=False,
        req_to_token_pool_same_object=True,
        token_allocator_same_object=True,
        head_weight_same_object=True,
        embedding_weight_same_object=True,
    )
    rows = [
        x.message
        for x in caplog.records
        if x.message.startswith("XING_SPEC_STARTUP_IDENTITY ")
    ]
    assert len(rows) == 1 and json.loads(rows[0].split(" ", 1)[1]) == record


@pytest.mark.parametrize("algorithm", ["NONE", "NGRAM", "DFLASH", "FROZEN_KV_MTP"])
def test_other_algorithms_emit_nothing(algorithm, caplog):
    s = scheduler()
    s.spec_algorithm.name = algorithm
    assert obs.emit_xing_spec_startup_identity(s) is None
    assert not caplog.records


def test_non_xing_does_not_inspect_draft(caplog):
    s = scheduler()
    s.tp_worker.model_runner.model_config.hf_config.architectures = ["OtherForCausalLM"]
    s.draft_worker = object()
    assert obs.emit_xing_spec_startup_identity(s) is None
    assert not caplog.records


def test_no_draft_emits_nothing(caplog):
    s = scheduler()
    s.draft_worker = None
    assert obs.emit_xing_spec_startup_identity(s) is None
    assert not caplog.records


def test_eagle3_metadata_supported():
    s = scheduler()
    s.spec_algorithm.name = "EAGLE3"
    assert obs.emit_xing_spec_startup_identity(s)["algorithm"] == "EAGLE3"


def test_wrong_head_method_is_reported_without_changing_it():
    s = scheduler()
    head = s.draft_worker.draft_runner.model.lm_head
    method = type("ModelOptFp4LinearMethod", (), {})()
    head.quant_method = method
    assert (
        obs.emit_xing_spec_startup_identity(s)["draft"]["head"]["quant_method_class"]
        == "ModelOptFp4LinearMethod"
    )
    assert head.quant_method is method


def test_independent_weights_not_misreported_as_shared():
    s = scheduler()
    draft = s.draft_worker.draft_runner
    draft.model.lm_head.weight = draft.model.lm_head.weight.clone()
    draft.model.model.embed_tokens.weight = (
        draft.model.model.embed_tokens.weight.clone()
    )
    r = obs.emit_xing_spec_startup_identity(s)
    assert r["sharing"]["head_weight_same_object"] is False
    assert r["sharing"]["embedding_weight_same_object"] is False


def test_no_tensor_value_reads_or_device_transfers(monkeypatch):
    s = scheduler()

    def denied(*a, **kw):
        raise AssertionError("Tensor values or device transfer were requested")

    for name in ("item", "tolist", "numpy", "cpu", "to", "__repr__", "__str__"):
        monkeypatch.setattr(torch.Tensor, name, denied)
    r = obs.emit_xing_spec_startup_identity(s)
    assert r is not None and r["draft"]["head"]["weight"]["shape"] == [2, 3]


def test_metadata_error_is_nonfatal_and_message_is_not_logged(caplog):
    s = scheduler()

    class Broken:
        @property
        def dtype(self):
            raise ValueError("private sentinel not for logs")

    s.draft_worker.draft_runner.model.lm_head.weight = Broken()
    assert obs.emit_xing_spec_startup_identity(s) is None
    assert "error_type=ValueError" in caplog.text
    assert "private sentinel" not in caplog.text


def test_real_scheduler_startup_observes_after_resize_and_weight_finalize(
    monkeypatch, caplog
):
    # Execute the actual method without importing unrelated GPU scheduler modules.
    import sglang

    source = Path(sglang.__file__).parent / "srt/managers/scheduler.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "init_model_worker"
    )
    events = []
    s = scheduler()
    target = s.tp_worker.model_runner
    draft = s.draft_worker
    for name in (
        "init_tp_model_worker",
        "maybe_init_draft_worker",
        "init_memory_pools",
        "init_all_attention_backends",
        "init_all_cuda_graphs",
    ):
        setattr(s, name, lambda name=name: events.append(name))
    s.tp_worker.start_startup_weight_load = lambda: events.append("load_start")
    s.tp_worker.finalize_startup_weight_load = lambda: events.append("weight_finalize")
    s.enable_overlap = False
    s.enable_dp_attention = False
    s.model_config = NS(context_len=262144)
    s.spec_algorithm.is_dflash_family = lambda: False
    s.tp_worker.get_worker_info = lambda: (
        192,
        4096,
        64,
        None,
        262144,
        262143,
        0,
        "cpu",
        None,
        None,
        None,
        None,
    )
    s.tp_worker.get_pad_input_ids_func = lambda: None
    draft.hicache_draft_plan = None
    draft.prewarm_sampling = lambda: events.append("prewarm")
    draft._draft_model_runners = lambda: (draft.draft_runner,)

    def resize(**kwargs):
        events.append("resize")
        target.token_to_kv_pool.size = 192
        target.max_total_num_tokens = 192

    target.post_capture_resize_kv_pool = resize
    target.device = "cpu"
    device = NS(Stream=lambda **kw: object(), stream=lambda stream: nullcontext())
    original = obs.emit_xing_spec_startup_identity
    records = []

    def observe(value):
        events.append("observe")
        records.append(original(value))

    monkeypatch.setattr(obs, "emit_xing_spec_startup_identity", observe)
    setup = NS(
        refresh_deep_gemm_layout_memory_budget=lambda *a, **kw: events.append("refresh")
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.model_runner_components.cuda_graph_setup",
        setup,
    )
    group = NS(cpu_group=object())
    parallel = NS(
        pp_size=1,
        pp_max_micro_batch_size=64,
        tp_group=group,
        attn_tp_group=group,
        attn_cp_group=group,
        pp_group=group,
        world_group=group,
        tp_rank=0,
    )
    namespace = dict(
        time=NS(perf_counter=lambda: 0.0),
        get_model=lambda: NS(is_startup_weight_load_overlap=True),
        get_parallel=lambda: parallel,
        get_exec=lambda: NS(moe=NS(elastic_ep_backend=None)),
        get_schedule=lambda: NS(min_free_slots_delay=None, chunked_prefill_size=4096),
        resolve_min_free_slots=lambda *a, **kw: None,
        set_random_seed=lambda seed: None,
        get_available_gpu_memory=lambda *a, **kw: 1.0,
        get_device=lambda: NS(gpu_id=0),
        logger=obs.logger,
        use_mlx=lambda: False,
        torch=NS(get_device_module=lambda device: device),
        kv_cache_builder=NS(
            prepare_hicache_staging=lambda **kw: events.append("hicache")
        ),
        prewarm_graph_pool_borrow=lambda: events.append("borrow"),
    )
    namespace["torch"] = NS(get_device_module=lambda ignored: device)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    namespace["init_model_worker"](s)
    assert (
        events.index("init_all_cuda_graphs")
        < events.index("resize")
        < events.index("weight_finalize")
        < events.index("observe")
        < events.index("refresh")
    )
    assert len(records) == 1 and records[0]["target"]["pool"]["size"] == 192
    assert s.model_worker is draft
