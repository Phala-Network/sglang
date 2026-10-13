"""Pure CPU checks of the real admission policy; no CUDA/import substitutes."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/configs/request_static_yarn.py"
)
spec = importlib.util.spec_from_file_location(
    "request_static_yarn_policy_under_test", SOURCE
)
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


def config(**updates):
    rope = {
        "rope_type": "yarn",
        "factor": 4.0,
        "request_static_factors": [1, 2, 4],
        "original_max_position_embeddings": 262144,
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
        **updates,
    }
    return policy.RequestStaticYarnConfig.from_hf_config(
        SimpleNamespace(rope_parameters=rope)
    )


def request(prompt, output, *, explicit=None, namespace=None):
    return SimpleNamespace(
        origin_input_ids=range(prompt),
        sampling_params=SimpleNamespace(
            max_new_tokens=output, custom_params={"yarn_factor": explicit}
        ),
        extra_key=namespace,
        rope_profile=None,
        multimodal_inputs=None,
    )


@pytest.mark.parametrize(
    "budget,factor",
    [
        (1, 1),
        (262144, 1),
        (262145, 2),
        (524288, 2),
        (524289, 4),
        (1000000, 4),
        (1048576, 4),
    ],
)
def test_smallest_covering_profile(budget, factor):
    assert config().select(budget).factor == factor


def test_table_offsets_and_profile_identity():
    first = config()
    assert [x[1] for x in first.profiles] == [0, 1048576, 3145728]
    assert len({x[2] for x in first.profiles}) == 3
    assert first.select(1).profile_id == config().select(1).profile_id
    assert first.select(1).profile_id != config(rope_theta=1000000).select(1).profile_id


@pytest.mark.parametrize(
    "architecture",
    ["Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"],
)
def test_supported_wrapper_requires_multimodal_mode(architecture):
    config().validate_model_support([architecture], True)
    with pytest.raises(ValueError, match="requires multimodal mode"):
        config().validate_model_support([architecture], False)


@pytest.mark.parametrize(
    "architectures",
    [None, [], ["Qwen3_5ForCausalLM"], ["Qwen3_5MoeForCausalLM"], ["Qwen3ForCausalLM"]],
)
def test_unported_architecture_is_not_silently_accepted(architectures):
    with pytest.raises(ValueError, match="ConditionalGeneration wrapper"):
        config().validate_model_support(architectures, True)


def test_model_support_guard_precedes_embedded_mtp_architecture_remapping():
    # This source-wiring check runs without Torch. Behavioral model startup is
    # a separate image gate; validating the rewritten MTP arch would reject
    # the supported wrapper's own embedded draft worker.
    tree = ast.parse(SOURCE.with_name("model_config.py").read_text(encoding="utf8"))
    model_config = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelConfig"
    )
    init = next(
        node for node in model_config.body if getattr(node, "name", None) == "__init__"
    )
    calls = {
        node.func.attr: node.lineno
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("validate_model_support", "_config_draft_model")
    }
    assert calls["validate_model_support"] < calls["_config_draft_model"]


@pytest.mark.parametrize(
    "updates",
    [
        {"request_static_factors": []},
        {"request_static_factors": [2, 1, 4]},
        {"request_static_factors": [1, 2, 2, 4]},
        {"request_static_factors": [1, float("nan"), 4]},
        {"request_static_factors": [1, float("inf")]},
        {"request_static_factors": [0.5, 4]},
        {"request_static_factors": [1, 2]},
        {"rope_type": "default"},
        {"original_max_position_embeddings": 0},
    ],
)
def test_invalid_configuration_fails(updates):
    with pytest.raises(ValueError):
        config(**updates)


def test_prompt_plus_output_and_expanded_mm_admission():
    req = request(262140, 5)
    policy.admit_request_static_yarn(req, config(), 1000000)
    assert req.rope_profile.factor == 2
    assert req.rope_profile.budget == 262145


def test_no_reselection_after_chunking_retraction_or_custom_param_change():
    req = request(262140, 5)
    policy.admit_request_static_yarn(req, config(), 1000000)
    selected, namespace = req.rope_profile, req.extra_key
    req.origin_input_ids = range(300000)
    req.sampling_params.custom_params["yarn_factor"] = 4
    policy.admit_request_static_yarn(req, config(), 1000000)
    assert req.rope_profile is selected
    assert req.extra_key == namespace
    with pytest.raises(Exception):
        selected.factor = 4


def test_explicit_output_budget_is_not_clipped_to_make_it_fit():
    with pytest.raises(ValueError, match="server context"):
        policy.admit_request_static_yarn(request(999999, 2), config(), 1000000)
    with pytest.raises(ValueError, match="profile capacity"):
        policy.admit_request_static_yarn(
            request(262140, 5, explicit=1), config(), 1000000
        )
    with pytest.raises(ValueError):
        config().select(1048577)


@pytest.mark.parametrize("explicit", [0, 3, True, "2", float("nan")])
def test_invalid_explicit_factor_rejected(explicit):
    with pytest.raises(ValueError):
        config().select(32, explicit)


def test_explicit_factor_is_test_control_and_none_reserves_full_context():
    req = request(10, 10, explicit=4)
    policy.admit_request_static_yarn(req, config(), 1000000)
    assert req.rope_profile.factor == 4
    req = request(10, None)
    policy.admit_request_static_yarn(req, config(), 1000000)
    assert req.rope_profile.factor == 4
    assert req.rope_profile.budget == 1000000


def test_cache_namespace_keeps_user_identity_and_isolates_profiles():
    reqs = [request(100, 10, explicit=f, namespace="tenant|x") for f in [1, 2, 4]]
    for req in reqs:
        policy.admit_request_static_yarn(req, config(), 1000000)
    assert len({req.extra_key for req in reqs}) == 3
    assert all(policy.json.loads(req.extra_key)[0] == "tenant|x" for req in reqs)


def test_feature_off_is_unchanged_and_override_not_silently_ignored():
    req = request(1000001, 10)
    policy.admit_request_static_yarn(req, None, 1000000)
    assert req.rope_profile is None
    with pytest.raises(ValueError, match="requires"):
        policy.admit_request_static_yarn(request(10, 1, explicit=2), None, 1000000)
