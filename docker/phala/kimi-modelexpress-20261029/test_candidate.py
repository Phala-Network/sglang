"""CPU regressions against the installed ModelExpress package, no model load."""

import datetime
import json
import os
import pathlib
from concurrent import futures
from types import SimpleNamespace

import grpc
import pytest
import torch
from modelexpress import p2p_pb2
from modelexpress.engines.sglang.checkpoint_cache import release_native_checkpoint_cache
from modelexpress.ucx_utils import _resolve_nic_pin, apply_nic_pin_for_device


@pytest.mark.parametrize(
    "rank,expected",
    [
        (0, "mlx5_0:1,mlx5_10:1"),
        (1, "mlx5_1:1,mlx5_11:1"),
    ],
)
def test_dual_rail_groups(monkeypatch, rank, expected):
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "mlx5_0:1,mlx5_10:1;mlx5_1:1,mlx5_11:1")
    monkeypatch.setenv("UCX_NET_DEVICES", "wrong:1")
    apply_nic_pin_for_device(rank)
    assert os.environ["UCX_NET_DEVICES"] == expected


@pytest.mark.parametrize(
    "mapping,rank",
    [
        ("mlx5_0:1,mlx5_10:1;", 0),
        ("mlx5_0:1,mlx5_0:1;mlx5_1:1", 0),
        ("mlx5_0:0;mlx5_1:1", 0),
        ("mlx5_0:1;mlx5_1:1", 2),
        ("mlx5_0:1;mlx5_1:1", -1),
        ("mlx5_0:1;all", 0),
    ],
)
def test_invalid_group_fails_before_agent_creation(monkeypatch, mapping, rank):
    monkeypatch.setenv("MX_RDMA_NIC_PIN", mapping)
    with pytest.raises(ValueError):
        _resolve_nic_pin(rank)


def test_legacy_mapping_and_off_preserved(monkeypatch):
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "mlx5_0:1,mlx5_1:1")
    assert _resolve_nic_pin(1) == "mlx5_1:1"
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "off")
    assert _resolve_nic_pin(0) is None


def test_disabled_cleanup_has_no_collective(monkeypatch):
    def forbidden():
        raise AssertionError("disabled cleanup entered distributed path")

    monkeypatch.setattr(torch.distributed, "is_initialized", forbidden)
    release_native_checkpoint_cache(SimpleNamespace(), SimpleNamespace())


def test_empty_p2p_load_does_not_drop_files(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    release_native_checkpoint_cache(
        SimpleNamespace(native_checkpoint_files=[]),
        SimpleNamespace(
            model_loader_extra_config={"drop_checkpoint_cache_after_model_load": True}
        ),
    )


def _mixed_rank_worker(rank, init_path, results_dir, all_p2p):
    """Real Gloo collectives, real engine cleanup, mocked file advice only."""
    import torch.distributed as dist
    from sglang.srt import runtime_context
    from sglang.srt.model_loader import weight_utils

    dist.init_process_group(
        "gloo",
        init_method="file://" + init_path,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=25),
    )
    world = SimpleNamespace(
        rank=rank, world_size=2, cpu_group=dist.group.WORLD, barrier=dist.barrier
    )
    parallel = SimpleNamespace(world_group=world, nnodes=1)
    runtime_context.get_parallel = lambda: parallel
    weight_utils.get_parallel = lambda: parallel
    advised = []
    weight_utils._drop_file_cache_after_load = advised.append
    # Rank 0 represents RDMA success; rank 1 represents native fallback.
    files = [] if rank == 0 or all_p2p else ["/weights/b", "/weights/a", "/weights/a"]
    adapter = SimpleNamespace(native_checkpoint_files=files)
    config = SimpleNamespace(
        model_loader_extra_config={"drop_checkpoint_cache_after_model_load": True}
    )
    try:
        release_native_checkpoint_cache(adapter, config)
        pathlib.Path(results_dir, str(rank) + ".json").write_text(json.dumps(advised))
        assert adapter.native_checkpoint_files == []
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("all_p2p", [False, True])
def test_mixed_rank_cleanup_does_not_deadlock(tmp_path, all_p2p):
    torch.multiprocessing.spawn(
        _mixed_rank_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path), all_p2p),
        nprocs=2,
        join=True,
    )
    advised = sum(
        [json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(2)], []
    )
    assert sorted(advised) == ([] if all_p2p else ["/weights/a", "/weights/b"])


def test_protobuf7_grpc_roundtrip():
    original = p2p_pb2.SourceIdentity(
        model_name="test/model", revision="fixed", tensor_parallel_size=8
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    handler = grpc.unary_unary_rpc_method_handler(
        lambda request, context: request,
        request_deserializer=p2p_pb2.SourceIdentity.FromString,
        response_serializer=p2p_pb2.SourceIdentity.SerializeToString,
    )
    server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler("mx.test", {"Echo": handler}),)
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            call = channel.unary_unary(
                "/mx.test/Echo",
                request_serializer=p2p_pb2.SourceIdentity.SerializeToString,
                response_deserializer=p2p_pb2.SourceIdentity.FromString,
            )
            assert call(original, timeout=5) == original
    finally:
        server.stop(0).wait()
