from types import SimpleNamespace

import pytest

from sglang.kernels.ops.communication import all_reduce_fusion
from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe as mxfp4_moe
from sglang.srt.models.deepseek_v2 import _select_fused_finalize_comm_key
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("num_tokens", [1, 48, 96])
def test_small_batches_keep_default_plane(num_tokens):
    assert (
        _select_fused_finalize_comm_key(
            num_tokens,
            is_nextn=True,
            is_target_verify=False,
            medium_enabled=False,
        )
        == all_reduce_fusion.DEFAULT_COMM_KEY
    )


@pytest.mark.parametrize("num_tokens", [97, 288, 336, 384])
def test_medium_target_batches_use_separate_plane(num_tokens):
    assert (
        _select_fused_finalize_comm_key(
            num_tokens,
            is_nextn=False,
            is_target_verify=True,
            medium_enabled=True,
        )
        == all_reduce_fusion.DSV41_MEDIUM_COMM_KEY
    )


@pytest.mark.parametrize("num_tokens", [97, 288, 336, 384])
@pytest.mark.parametrize(
    "is_nextn,is_target_verify,medium_enabled",
    [
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ],
)
def test_medium_path_is_target_only_and_opt_in(
    num_tokens, is_nextn, is_target_verify, medium_enabled
):
    assert (
        _select_fused_finalize_comm_key(
            num_tokens,
            is_nextn=is_nextn,
            is_target_verify=is_target_verify,
            medium_enabled=medium_enabled,
        )
        is None
    )


@pytest.mark.parametrize("num_tokens", [0, 385, 512])
def test_out_of_policy_batches_fall_back(num_tokens):
    assert (
        _select_fused_finalize_comm_key(
            num_tokens,
            is_nextn=False,
            is_target_verify=True,
            medium_enabled=True,
        )
        is None
    )


def test_communicator_registry_is_namespaced(monkeypatch):
    monkeypatch.setattr(all_reduce_fusion, "_COMM_MAP", {})
    small = SimpleNamespace(world_size=4)
    medium = SimpleNamespace(world_size=4)

    all_reduce_fusion.register_comm(small, comm_key=all_reduce_fusion.DEFAULT_COMM_KEY)
    all_reduce_fusion.register_comm(
        medium, comm_key=all_reduce_fusion.DSV41_MEDIUM_COMM_KEY
    )

    assert all_reduce_fusion.get_registered_comm(4) is small
    assert (
        all_reduce_fusion.get_registered_comm(
            4, all_reduce_fusion.DSV41_MEDIUM_COMM_KEY
        )
        is medium
    )
    with pytest.raises(AssertionError):
        all_reduce_fusion.register_comm(
            SimpleNamespace(world_size=4),
            comm_key=all_reduce_fusion.DSV41_MEDIUM_COMM_KEY,
        )


def test_medium_comm_initialization_and_selection_are_observable(monkeypatch):
    logs = []
    created = []

    class FakeComm:
        def __init__(self, group=None, device=None, **kwargs):
            created.append(dict(kwargs))
            self.group = group
            self.device = device
            self.world_size = 4
            self.disabled = False
            self.max_push_size = kwargs.get("max_push_size", 1024 * 1024)
            self.config = SimpleNamespace(
                num_push_blocks=kwargs.get("max_push_blocks", 96)
            )
            self.obj = SimpleNamespace(world_size=4)

    default_comm = FakeComm(group=object(), device=object())
    tp_group = SimpleNamespace(ca_comm=default_comm, world_size=4)

    monkeypatch.setattr(mxfp4_moe, "_fused_finalize_all_reduce_comms", {})
    monkeypatch.setattr(mxfp4_moe, "_fused_finalize_all_reduce_probed", set())
    monkeypatch.setattr(mxfp4_moe, "_fused_finalize_all_reduce_selected", set())
    monkeypatch.setattr(mxfp4_moe, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        mxfp4_moe, "log_info_on_rank0", lambda logger, msg: logs.append(msg)
    )
    monkeypatch.setattr(
        "sglang.srt.distributed.device_communicators.custom_all_reduce_v2.CustomAllReduceV2",
        FakeComm,
    )
    monkeypatch.setattr(
        "sglang.kernels.ops.communication.mp.register_comm_cleanup", lambda comm: None
    )
    monkeypatch.setattr(all_reduce_fusion, "_COMM_MAP", {})
    monkeypatch.setattr("torch.cuda.is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
        lambda: False,
    )

    created_before_init = len(created)
    assert mxfp4_moe.initialize_fused_finalize_all_reduce_comm(
        all_reduce_fusion.DEFAULT_COMM_KEY
    )
    assert mxfp4_moe.initialize_fused_finalize_all_reduce_comm(
        all_reduce_fusion.DSV41_MEDIUM_COMM_KEY
    )
    assert mxfp4_moe.initialize_fused_finalize_all_reduce_comm(
        all_reduce_fusion.DEFAULT_COMM_KEY
    )
    assert mxfp4_moe.initialize_fused_finalize_all_reduce_comm(
        all_reduce_fusion.DSV41_MEDIUM_COMM_KEY
    )
    assert len(created) == created_before_init + 1

    class FakeQuantMethod:
        flashinfer_mxfp4_moe_precision = "default"

    monkeypatch.setattr(mxfp4_moe, "Mxfp4FlashinferTrtllmMoEMethod", FakeQuantMethod)
    experts = SimpleNamespace(
        quant_method=FakeQuantMethod(),
        should_fuse_routed_scaling_factor_in_topk=True,
    )
    assert mxfp4_moe.should_use_fuse_finalize_all_reduce(
        experts,
        96,
        5120,
        comm_key=all_reduce_fusion.DEFAULT_COMM_KEY,
    )
    assert mxfp4_moe.should_use_fuse_finalize_all_reduce(
        experts,
        288,
        5120,
        comm_key=all_reduce_fusion.DSV41_MEDIUM_COMM_KEY,
    )
    assert not mxfp4_moe.should_use_fuse_finalize_all_reduce(
        experts,
        385,
        5120,
        comm_key=all_reduce_fusion.DSV41_MEDIUM_COMM_KEY,
    )
    medium = mxfp4_moe._fused_finalize_all_reduce_comms[
        all_reduce_fusion.DSV41_MEDIUM_COMM_KEY
    ]
    medium.max_push_size = 2 * 1024 * 1024
    assert not mxfp4_moe.should_use_fuse_finalize_all_reduce(
        experts,
        288,
        5120,
        comm_key=all_reduce_fusion.DSV41_MEDIUM_COMM_KEY,
    )
    assert any("initialized comm_key=default" in item for item in logs)
    assert any("initialized comm_key=dsv41_medium" in item for item in logs)
    assert any("selected comm_key=dsv41_medium rows=288" in item for item in logs)


def test_capture_first_medium_probe_can_retry_eagerly(monkeypatch):
    class FakeComm:
        def __init__(self, group=None, device=None, **kwargs):
            self.group = group
            self.device = device
            self.world_size = 4
            self.disabled = False
            self.max_push_size = kwargs.get("max_push_size", 1024 * 1024)
            self.config = SimpleNamespace(
                num_push_blocks=kwargs.get("max_push_blocks", 96)
            )
            self.obj = SimpleNamespace(world_size=4)

    default_comm = FakeComm(group=object(), device=object())
    tp_group = SimpleNamespace(ca_comm=default_comm, world_size=4)
    capture = {"active": True}

    monkeypatch.setattr(mxfp4_moe, "_fused_finalize_all_reduce_comms", {})
    monkeypatch.setattr(mxfp4_moe, "_fused_finalize_all_reduce_probed", set())
    monkeypatch.setattr(mxfp4_moe, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(mxfp4_moe, "log_info_on_rank0", lambda logger, msg: None)
    monkeypatch.setattr(
        "sglang.srt.distributed.device_communicators.custom_all_reduce_v2.CustomAllReduceV2",
        FakeComm,
    )
    monkeypatch.setattr(
        "sglang.kernels.ops.communication.mp.register_comm_cleanup", lambda comm: None
    )
    monkeypatch.setattr(all_reduce_fusion, "_COMM_MAP", {})
    monkeypatch.setattr(
        "torch.cuda.is_current_stream_capturing", lambda: capture["active"]
    )

    key = all_reduce_fusion.DSV41_MEDIUM_COMM_KEY
    assert not mxfp4_moe.initialize_fused_finalize_all_reduce_comm(key)
    assert key not in mxfp4_moe._fused_finalize_all_reduce_probed

    capture["active"] = False
    assert mxfp4_moe.initialize_fused_finalize_all_reduce_comm(key)
    assert key in mxfp4_moe._fused_finalize_all_reduce_probed
