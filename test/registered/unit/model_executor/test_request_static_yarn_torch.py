"""Real Torch/kernel checks; CUDA cases run only on an actual GPU."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.configs.request_static_yarn import RequestStaticYarnConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)


@pytest.mark.parametrize("feature", [False, True])
def test_actual_tokenizer_budget_path_cannot_clip_request_static_budget(feature):
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    manager = object.__new__(TokenizerManager)
    manager.context_len = 1000
    manager.allow_auto_truncate = True
    manager.validate_total_tokens = True
    manager.num_reserved_tokens = 0
    manager.model_config = SimpleNamespace(
        request_static_yarn_config=object() if feature else None,
        joint_head_config=None, pointer_head=None
    )
    req = SimpleNamespace(sampling_params={"max_new_tokens": 2})
    execution = SimpleNamespace(
        features=SimpleNamespace(enable_encoder_swa_bounded_replay=False)
    )
    with patch(
        "sglang.srt.managers.tokenizer_manager.get_exec", return_value=execution
    ):
        if feature:
            with pytest.raises(ValueError, match="maximum context length"):
                manager._validate_token_budget(req, list(range(999)))
            assert req.sampling_params["max_new_tokens"] == 2
        else:
            manager._validate_token_budget(req, list(range(999)))
            assert req.sampling_params["max_new_tokens"] == 1


PARAMETERS = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 64,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
}


def rope(factor=None, dtype=torch.float32):
    params = dict(PARAMETERS)
    if factor is None:
        params["request_static_factors"] = [1, 2, 4]
    elif factor == 1:
        params["rope_type"] = "default"
    else:
        params["factor"] = factor
    return get_rope(
        head_size=256,
        rotary_dim=256,
        max_position=256,
        base=10000000,
        rope_scaling=params,
        partial_rotary_factor=0.25,
        dtype=dtype,
    )


def inputs(device, order=(1, 2, 4), vision=True):
    gen = torch.Generator(device=device).manual_seed(71983)
    positions = torch.cat(
        [
            torch.randint(0, int(64 * factor), (3, 4), generator=gen, device=device)
            for factor in order
        ],
        dim=1,
    )
    if not vision:
        positions = positions[0:1].expand(3, -1).contiguous()
    q = torch.randn(12, 512, generator=gen, device=device)
    k = torch.randn(12, 256, generator=gen, device=device)
    return positions, q, k


@pytest.mark.parametrize("vision", [False, True])
def test_combined_native_matches_isolated_static_and_native_1x(vision):
    combined = rope()
    positions, q, k = inputs("cpu", vision=vision)
    offsets = torch.tensor(
        [combined.scaling_factor_to_offset[f] for f in (1, 2, 4)]
    ).repeat_interleave(4)
    actual = combined.forward_native(positions + offsets, q.clone(), k.clone())
    for index, factor in enumerate((1, 2, 4)):
        sl = slice(index * 4, (index + 1) * 4)
        expected = rope(factor).forward_native(
            positions[:, sl], q[sl].clone(), k[sl].clone()
        )
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result[sl], reference, rtol=0, atol=0)
    assert torch.equal(actual[0][:, 64:256], q[:, 64:256])


@pytest.mark.parametrize("uniform,lengths", [(False, [1, 3, 2]), (True, [4, 4, 4])])
def test_real_forward_batch_offsets_preserve_physical_and_vision_positions(
    uniform, lengths
):
    config = RequestStaticYarnConfig.from_hf_config(
        SimpleNamespace(
            rope_parameters={**PARAMETERS, "request_static_factors": [1, 2, 4]}
        )
    )
    profiles = [config.select(1, f) for f in (4, 1, 2)]
    n = sum(lengths)
    physical = torch.arange(n)
    original = torch.stack((physical, physical + 2, physical + 5))
    # Exercise the actual ForwardBatch method without constructing unrelated
    # scheduler resources or replacing its implementation with a simulator.
    fb = object.__new__(ForwardBatch)
    fb.positions = physical.clone()
    fb.mrope_positions = original.clone()
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rope_profile=p) for p in profiles], extend_lens=lengths
    )
    runner = SimpleNamespace(
        device="cpu", model_config=SimpleNamespace(request_static_yarn_config=config)
    )
    fb._apply_request_static_yarn_offsets(runner, batch, uniform=uniform)
    expected_offsets = torch.tensor([p.offset for p in profiles]).repeat_interleave(
        torch.tensor(lengths)
    )
    assert torch.equal(fb.mrope_positions, original + expected_offsets)
    assert torch.equal(fb.positions, physical)


@pytest.mark.parametrize("explicit_positions", [False, True])
@pytest.mark.parametrize("mixed_mm", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_spec_mrope_entrypoint_applies_profiles_once(
    explicit_positions, mixed_mm, enabled
):
    config = RequestStaticYarnConfig.from_hf_config(
        SimpleNamespace(
            rope_parameters={**PARAMETERS, "request_static_factors": [1, 2, 4]}
        )
    )
    profiles = [config.select(1, f) for f in (2, 4, 1)]
    physical = torch.tensor([65, 66, 67, 129, 130, 131, 4, 5, 6])
    original = physical.clone()
    deltas = [0, 7, 0] if mixed_mm else [0, 0, 0]
    mm = [None, SimpleNamespace(mrope_position_delta=torch.tensor([[7]])), None]
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rope_profile=p) for p in profiles],
        multimodal_inputs=mm if mixed_mm else [None] * 3,
        spec_info=SimpleNamespace(positions=physical),
    )
    runner = SimpleNamespace(
        device="cpu",
        model_config=SimpleNamespace(
            request_static_yarn_config=config if enabled else None
        ),
    )
    fb = object.__new__(ForwardBatch)
    fb.positions = physical
    fb.seq_lens = torch.tensor([68, 132, 7])
    fb.compute_spec_mrope_positions(
        runner, batch, seq_positions=physical if explicit_positions else None
    )
    per_request = torch.tensor(deltas)
    if enabled:
        per_request += torch.tensor([p.offset for p in profiles])
    expected = original + per_request.repeat_interleave(3)
    assert torch.equal(fb.mrope_positions, expected.unsqueeze(0).repeat(3, 1))
    assert torch.equal(fb.positions, original)
    assert torch.equal(batch.spec_info.positions, original)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "mode,prepared",
    [
        (ForwardMode.DECODE, "spec"),
        (ForwardMode.TARGET_VERIFY, "spec"),
        (ForwardMode.DRAFT_EXTEND_V2, "extend"),
        (ForwardMode.TARGET_VERIFY, "compute"),
        (ForwardMode.DRAFT_EXTEND_V2, "compute"),
    ],
)
def test_init_new_precomputed_mrope_paths_apply_profile_exactly_once(
    enabled, mode, prepared
):
    """Real init_new chooses physical/prepared positions before model-only encoding."""
    import sglang.srt.model_executor.forward_batch_info as module

    class BatchInputs(SimpleNamespace):
        def __getattr__(self, name):
            return None

    config = RequestStaticYarnConfig.from_hf_config(
        SimpleNamespace(
            rope_parameters={**PARAMETERS, "request_static_factors": [1, 2, 4]}
        )
    )
    profiles = [config.select(1, f) for f in (2, 4, 1)]
    width = 1 if mode == ForwardMode.DECODE else 4
    physical = torch.arange(width * 3, dtype=torch.int64)
    mrope = physical.unsqueeze(0).repeat(3, 1)
    prepared_copy = mrope.clone()
    batch = BatchInputs(
        forward_mode=mode,
        seq_lens=torch.tensor([67, 131, 7]),
        seq_lens_cpu=torch.tensor([67, 131, 7]),
        seq_lens_sum=205,
        input_ids=torch.zeros(width * 3, dtype=torch.int64),
        req_pool_indices=torch.tensor([1, 2, 3]),
        out_cache_loc=torch.arange(width * 3),
        reqs=[SimpleNamespace(rope_profile=p, lora_id=None, rid="cpu") for p in profiles],
        extend_lens=[width] * 3,
        prefix_lens=[0] * 3,
        extend_num_tokens=width * 3,
        extend_logprob_start_lens=[0] * 3,
        multimodal_inputs=[None] * 3,
        spec_info=SimpleNamespace(positions=physical),
        sampling_info=None,
    )
    runner = SimpleNamespace(
        device="cpu",
        is_draft_worker=mode != ForwardMode.TARGET_VERIFY,
        model_config=SimpleNamespace(
            model_is_mrope=True,
            request_static_yarn_config=config if enabled else None,
        ),
        ngram_embedding_manager=SimpleNamespace(enabled=False),
        lora_manager=None,
        kv_index_translator=SimpleNamespace(own_plan=lambda _: MagicMock()),
    )
    # Device planning and unrelated metadata are stubs; init_new's own position
    # selection and both real mrope methods run unmodified.
    with ExitStack() as stack:
        for name in (
            "_maybe_init_non_generation_fields",
            "_init_num_token_non_padded",
            "init_mlp_sync_metadata",
        ):
            stack.enter_context(patch.object(ForwardBatch, name))
        stack.enter_context(patch.object(module, "_maybe_build_forward_token_modalities"))
        stack.enter_context(
            patch.object(module, "get_parallel", return_value=SimpleNamespace(attn_dcp_size=1))
        )
        stack.enter_context(patch.object(module, "is_pin_memory_available", return_value=False))
        result = ForwardBatch.init_new(
            batch,
            runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
            spec_mrope_positions=mrope if prepared == "spec" else None,
            extend_position_info=(
                physical,
                torch.tensor([0, width, width * 2]),
                mrope if prepared == "extend" else None,
            ),
        )
    offsets = (
        torch.tensor([p.offset for p in profiles]).repeat_interleave(width)
        if enabled
        else torch.zeros(width * 3, dtype=torch.int64)
    )
    assert torch.equal(result.mrope_positions, prepared_copy + offsets)
    assert torch.equal(result.positions, physical)
    assert torch.equal(mrope, prepared_copy)
    assert torch.equal(batch.spec_info.positions, physical)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA device required")
@pytest.mark.parametrize("vision", [False, True])
def test_cuda_mixed_profiles_match_isolated_tables(vision):
    combined = rope().to("cuda")
    positions, q, k = inputs("cuda", vision=vision)
    offsets = torch.tensor(
        [combined.scaling_factor_to_offset[f] for f in (1, 2, 4)], device="cuda"
    ).repeat_interleave(4)
    actual = combined.forward_cuda(positions + offsets, q.clone(), k.clone())
    for index, factor in enumerate((1, 2, 4)):
        sl = slice(index * 4, (index + 1) * 4)
        expected = (
            rope(factor)
            .to("cuda")
            .forward_cuda(positions[:, sl], q[sl].clone(), k[sl].clone())
        )
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result[sl], reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA device required")
def test_cuda_graph_replays_new_mixed_profile_positions():
    combined = rope().to("cuda")
    positions, q, k = inputs("cuda")
    position_buffer = torch.zeros_like(positions)
    q_buffer, k_buffer = q.clone(), k.clone()
    for _ in range(3):
        combined.forward_cuda(position_buffer, q_buffer, k_buffer)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = combined.forward_cuda(position_buffer, q_buffer, k_buffer)
    for order in [(1, 2, 4), (4, 1, 2), (2, 4, 1)]:
        positions, q, k = inputs("cuda", order=order)
        offsets = torch.tensor(
            [combined.scaling_factor_to_offset[f] for f in order], device="cuda"
        ).repeat_interleave(4)
        encoded = positions + offsets
        expected = combined.forward_cuda(encoded, q.clone(), k.clone())
        position_buffer.copy_(encoded)
        q_buffer.copy_(q)
        k_buffer.copy_(k)
        graph.replay()
        torch.cuda.synchronize()
        for actual_tensor, expected_tensor in zip(result, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def fused_inputs(order=(1, 2, 4), vision=True):
    """Ordinary 27B head layout; tables stay small to isolate kernel/layout math."""
    positions = torch.cat(
        [
            torch.tensor(
                [
                    [0, 1, 64 * factor - 2, 64 * factor - 1],
                    [0, 3, 64 * factor - 1, 64 * factor - 2],
                    [1, 0, 64 * factor - 3, 64 * factor - 1],
                ],
                device="cuda",
            )
            for factor in order
        ],
        dim=1,
    )
    if not vision:
        positions = positions[0:1].expand(3, -1).contiguous()
    gen = torch.Generator(device="cuda").manual_seed(71984)
    q_gate = torch.randn(
        12, 24 * 2 * 256, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    k = torch.randn(12, 4 * 256, device="cuda", dtype=torch.bfloat16, generator=gen)
    q_weight = torch.randn(256, device="cuda", dtype=torch.bfloat16, generator=gen)
    k_weight = torch.randn(256, device="cuda", dtype=torch.bfloat16, generator=gen)
    return positions, q_gate, k, q_weight, k_weight


def call_actual_qwen_fused(rotary, positions, q_gate, k, q_weight, k_weight):
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    return fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        q_weight,
        k_weight,
        rotary.cos_sin_cache,
        positions,
        1e-6,
        24,
        4,
        256,
        64,
        has_gate=True,
        mrope_axis_map=rotary.axis_map,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA device required")
@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vision", [False, True])
def test_actual_qwen_fused_mixed_profiles_match_isolated_tables(table_dtype, vision):
    combined = rope(dtype=table_dtype).to(device="cuda", dtype=table_dtype)
    positions, q_gate, k, q_weight, k_weight = fused_inputs(vision=vision)
    offsets = torch.tensor(
        [combined.scaling_factor_to_offset[f] for f in (1, 2, 4)], device="cuda"
    ).repeat_interleave(4)
    actual = call_actual_qwen_fused(
        combined, positions + offsets, q_gate, k, q_weight, k_weight
    )
    for index, factor in enumerate((1, 2, 4)):
        sl = slice(index * 4, (index + 1) * 4)
        isolated = rope(factor, dtype=table_dtype).to(device="cuda", dtype=table_dtype)
        expected = call_actual_qwen_fused(
            isolated, positions[:, sl], q_gate[sl], k[sl], q_weight, k_weight
        )
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result[sl], reference, rtol=0, atol=0)
    torch.testing.assert_close(
        actual[2], q_gate.view(12, 24, 512)[..., 256:], rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA device required")
@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
def test_actual_qwen_fused_graph_replays_profile_reordering(table_dtype):
    combined = rope(dtype=table_dtype).to(device="cuda", dtype=table_dtype)
    positions, q_gate, k, q_weight, k_weight = fused_inputs()
    position_buffer = torch.zeros_like(positions)
    q_gate_buffer, k_buffer = q_gate.clone(), k.clone()
    for _ in range(3):
        call_actual_qwen_fused(
            combined, position_buffer, q_gate_buffer, k_buffer, q_weight, k_weight
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call_actual_qwen_fused(
            combined, position_buffer, q_gate_buffer, k_buffer, q_weight, k_weight
        )
    for order in [(1, 2, 4), (4, 1, 2), (2, 4, 1)]:
        positions, q_gate, k, _q_weight, _k_weight = fused_inputs(order=order)
        offsets = torch.tensor(
            [combined.scaling_factor_to_offset[f] for f in order], device="cuda"
        ).repeat_interleave(4)
        position_buffer.copy_(positions + offsets)
        q_gate_buffer.copy_(q_gate)
        k_buffer.copy_(k)
        graph.replay()
        torch.cuda.synchronize()
        for index, factor in enumerate(order):
            sl = slice(index * 4, (index + 1) * 4)
            isolated = rope(factor, dtype=table_dtype).to(
                device="cuda", dtype=table_dtype
            )
            expected = call_actual_qwen_fused(
                isolated, positions[:, sl], q_gate[sl], k[sl], q_weight, k_weight
            )
            for actual_tensor, expected_tensor in zip(result, expected):
                torch.testing.assert_close(
                    actual_tensor[sl], expected_tensor, rtol=0, atol=0
                )
