"""CPU regression for the real EAGLE target startup dummy; no CUDA claims."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.configs.request_static_yarn import RequestStaticYarnConfig
from sglang.srt.layers.rotary_embedding.request_static_yarn import (
    scale_request_static_yarn_query,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner import base_runner, flashinfer_autotune
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.speculative import spec_info
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic-mla", "default-none"])
def test_real_target_autotune_dummy_uses_short_model_positions(monkeypatch, dynamic):
    cfg = RequestStaticYarnConfig.from_hf_config(
        NS(
            model_type="xing4_0",
            rope_theta=10000,
            rope_scaling=dict(
                rope_type="deepseek_yarn",
                factor=128,
                request_static_factors=[64, 128],
                original_max_position_embeddings=4096,
                beta_fast=32,
                beta_slow=1,
                mscale=1,
                mscale_all_dim=1,
                _request_static_yarn_lookahead=8,
            ),
        )
    ) if dynamic else None
    if cfg is not None:
        assert cfg.position_kind == "mla" and cfg.profiles[0][1] == 0

    # Replace only process/hardware state; allocation, autotune dispatch, dummy
    # ForwardBatch construction, EAGLE verify metadata, and the Q guard are real.
    monkeypatch.setattr(base_runner, "get_parallel", lambda: NS(pp_size=1, dp_size=1))
    monkeypatch.setattr(base_runner, "get_disagg", lambda: NS(disaggregation_mode="null"))
    monkeypatch.setattr(base_runner, "get_flags", lambda: NS(capture=NS(enable_torch_compile=False)))
    monkeypatch.setattr(base_runner, "get_exec", lambda: NS(features=NS(enable_encoder_swa_bounded_replay=False)))
    monkeypatch.setattr(base_runner, "get_server_return_hidden_states_mode", lambda: CaptureHiddenMode.NULL)
    for name in ("require_mlp_tp_gather", "require_attn_tp_gather", "require_gathered_buffer", "enable_num_token_non_padded"):
        monkeypatch.setattr(base_runner, name, lambda: False)
    monkeypatch.setattr(base_runner, "set_dp_buffer_len", lambda *args: None)
    monkeypatch.setattr(base_runner, "set_is_extend_in_batch", lambda *args: None)
    monkeypatch.setattr(base_runner, "forward_context", lambda *args: nullcontext())
    monkeypatch.setattr(flashinfer_autotune, "flashinfer_autotune_context", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(spec_info, "get_spec_config", lambda: NS(
        speculative_num_steps=3, speculative_eagle_topk=1, speculative_num_draft_tokens=4))

    seen = []
    query = torch.arange(-32, 32, dtype=torch.bfloat16).view(8, 2, 4)
    query[0, 0, 0] = -0.0
    attention = NS(
        request_static_yarn_scale_ratio=1.1 if dynamic else None,
        request_static_yarn_long_offset=cfg.profiles[1][1] if dynamic else None,
    )

    def forward(input_ids, positions, batch):
        scaled = scale_request_static_yarn_query(query, batch, attention)
        assert torch.equal(scaled.view(torch.int16), query.view(torch.int16))
        assert batch.forward_mode == ForwardMode.TARGET_VERIFY
        assert batch.batch_size == 2 and input_ids.numel() == 8
        assert isinstance(batch.spec_info, EagleVerifyInput)
        assert batch.spec_info.draft_token_num == 4
        assert batch.spec_info.hidden_states.shape == (8, 8)
        assert batch.positions is positions
        if dynamic:
            assert batch.rope_positions.data_ptr() == positions.data_ptr()
            assert batch.rope_positions.dtype == torch.int64
            assert torch.equal(batch.rope_positions, physical)
            assert torch.all(batch.rope_positions < attention.request_static_yarn_long_offset)
        else:
            assert batch.rope_positions is None
        assert torch.equal(positions, physical)
        assert torch.equal(batch.out_cache_loc, cache_locs)
        seen.append(batch)

    mr = NS(
        device="cpu", dtype=torch.bfloat16,
        model_config=NS(hidden_size=8, vocab_size=16, dtype=torch.bfloat16,
            is_encoder_decoder=False, hf_config=NS(), request_static_yarn_config=cfg),
        model=NS(forward=forward),
        attn_backend=NS(get_cuda_graph_seq_len_fill_value=lambda: 1,
            init_forward_metadata=lambda batch: None),
        ngram_embedding_manager=NS(enabled=False),
        get_pp_proxy_topk_size=lambda: None,
        get_pp_proxy_residual_num_blocks=lambda: None,
        spec_algorithm=SpeculativeAlgorithm.from_string("EAGLE"),
        decode_num_tokens_per_req=lambda: 4,
        is_generation=True, is_draft_worker=False, lora_manager=None, canary_manager=None,
        prepare_dummy_forward_batch=lambda batch: batch,
        tp_group=NS(barrier=lambda: None),
    )
    runner = object.__new__(EagerRunner)
    runner.model_runner, runner._eager_max_bs = mr, 2
    buffers, batch_size = runner._autotune_buffers()
    buffers.positions.copy_(torch.arange(19, 27))
    buffers.out_cache_loc.copy_(torch.arange(101, 109))
    physical, cache_locs = buffers.positions.clone(), buffers.out_cache_loc.clone()
    runner._flashinfer_autotune(buffers=buffers, batch_size=batch_size)
    assert len(seen) == 1
    assert torch.equal(buffers.positions, physical)
    assert torch.equal(buffers.out_cache_loc, cache_locs)
