"""CPU contract checks of the actual Xing policy, including public 512K."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/configs/request_static_yarn.py"
spec = importlib.util.spec_from_file_location("xing_yarn_policy_under_test", SOURCE)
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


def xing_config(**updates):
    rope = {
        "rope_type": "yarn",
        "factor": 128,
        "request_static_factors": [64, 128],
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1,
        "mscale_all_dim": 1,
        **updates,
    }
    return policy.RequestStaticYarnConfig.from_hf_config(SimpleNamespace(
        model_type="xing4_0", rope_scaling=rope, rope_theta=10000,
        qk_nope_head_dim=128, qk_rope_head_dim=64, rope_interleave=True,
    ))


def request(prompt, output, factor=None, namespace=None):
    return SimpleNamespace(
        origin_input_ids=range(prompt),
        sampling_params=SimpleNamespace(
            max_new_tokens=output, custom_params={"yarn_factor": factor}
        ),
        rope_profile=None, extra_key=namespace, multimodal_inputs=None,
    )


@pytest.mark.parametrize("budget,factor", [
    (0, 64), (262144, 64), (262145, 128), (524288, 128),
])
def test_public_budget_boundaries_are_independent_of_internal_reserve(budget, factor):
    profile = xing_config().select(budget, reserved_tokens=128)
    assert profile.factor == factor
    assert profile.budget == budget
    assert profile.capacity == (262144 if factor == 64 else 524288)
    assert profile.reserved_tokens == 128
    with pytest.raises(ValueError, match="profile capacity"):
        xing_config().select(524289)


def test_exact_512k_admission_preserves_original_output_and_null_contract():
    req = request(524032, 256)
    policy.admit_request_static_yarn(req, xing_config(), 524288, reserved_tokens=65)
    assert req.sampling_params.max_new_tokens == 256
    assert req.rope_profile.metadata() == {
        "factor": 128, "id": req.rope_profile.profile_id,
        "identity": req.rope_profile.profile_id,
        "budget": 524288, "admitted_budget": 524288,
        "capacity": 524288, "reserved_tokens": 65,
    }
    unbounded = request(32, None)
    policy.admit_request_static_yarn(unbounded, xing_config(), 524288, reserved_tokens=97)
    assert unbounded.rope_profile.factor == 128
    assert unbounded.rope_profile.budget == 524288
    assert unbounded.sampling_params.max_new_tokens is None
    with pytest.raises(ValueError, match="server context"):
        policy.admit_request_static_yarn(request(524032, 257), xing_config(), 524288)


def test_explicit_native_undercoverage_never_clips_budget():
    req = request(262140, 5, 64)
    with pytest.raises(ValueError, match="profile capacity"):
        policy.admit_request_static_yarn(req, xing_config(), 524288)
    assert req.sampling_params.max_new_tokens == 5
    assert req.rope_profile is None and req.extra_key is None


def test_fixed_profile_survives_request_lifecycle_and_preserves_caller_namespace():
    req = request(10, 10, 128, 'tenant:["yarn:x"]')
    policy.admit_request_static_yarn(req, xing_config(), 524288)
    profile, namespace = req.rope_profile, req.extra_key
    req.origin_input_ids = range(300000)
    req.sampling_params.max_new_tokens = 1
    req.sampling_params.custom_params["yarn_factor"] = 64
    policy.admit_request_static_yarn(req, xing_config(), 524288)
    assert req.rope_profile is profile and req.extra_key == namespace
    assert policy.json.loads(namespace) == ['tenant:["yarn:x"]', profile.profile_id]
    with pytest.raises(Exception):
        profile.factor = 64


def test_offsets_and_full_math_identity_include_deepseek_normalization():
    config = xing_config()
    assert [p[1] for p in config.profiles] == [0, 262144]
    assert len({p[2] for p in config.profiles}) == 2
    assert config.select(1).profile_id == xing_config(rope_type="deepseek_yarn").select(1).profile_id
    config.validate_model_support(["Xing4_0ForCausalLM"], False)
    with pytest.raises(ValueError, match="Xing4_0ForCausalLM"):
        config.validate_model_support(["DeepseekV3ForCausalLM"], False)


@pytest.mark.parametrize("updates", [
    {"factor": 64}, {"request_static_factors": [1, 2]},
    {"request_static_factors": [128, 64]},
    {"original_max_position_embeddings": 262144}, {"mscale_all_dim": 0},
    {"beta_fast": 16}, {"rope_theta": 1000000}, {"partial_rotary_factor": 0.5},
])
def test_nonofficial_native_math_fails_closed(updates):
    with pytest.raises(ValueError):
        xing_config(**updates)


@pytest.mark.parametrize("factor", [2, True, "128", float("nan"), float("inf")])
def test_malformed_explicit_factor_rejected(factor):
    with pytest.raises(ValueError):
        xing_config().select(32, factor)


def test_trtllm_runtime_supported_tp1_tp2_and_bf16_fp8_kv():
    for tp in (1, 2):
        for dtype in ("auto", "bfloat16", "fp8_e4m3"):
            xing_config().validate_runtime_support(SimpleNamespace(
                attention_backend="trtllm_mla", tp_size=tp, kv_cache_dtype=dtype,
            ))


@pytest.mark.parametrize("updates", [
    {"attention_backend": None}, {"attention_backend": "flashinfer"},
    {"decode_attention_backend": "fa3"}, {"prefill_attention_backend": "fa4"},
    {"device": "cpu"}, {"dtype": "float16"}, {"kv_cache_dtype": "fp8_e5m2"},
    {"tp_size": 4}, {"pp_size": 2}, {"dp_size": 2}, {"attn_cp_size": 2},
    {"dcp_size": 2}, {"enable_prefill_cp": True}, {"enable_two_batch_overlap": True},
    {"speculative_algorithm": "EAGLE"}, {"enable_lora": True},
    {"disaggregation_mode": "prefill"}, {"enable_lmcache": True},
    {"enable_hierarchical_cache": True}, {"cpu_offload_gb": 1},
])
def test_unported_runtime_path_is_rejected_before_weight_load(updates):
    args = {"attention_backend": "trtllm_mla", **updates}
    with pytest.raises(ValueError):
        xing_config().validate_runtime_support(SimpleNamespace(**args))
