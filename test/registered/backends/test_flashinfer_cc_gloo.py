"""Eight-process CPU/Gloo failure-ordering test; does not test GPU kernels."""

from datetime import timedelta
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import torch.distributed as dist
import torch.multiprocessing as mp

from test_flashinfer_cc_contract import FI_ROOT, functions


def rank_main(rank, world_size, port, fail_rank, result_dir):
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}", timeout=timedelta(seconds=45))
    events = []

    class Backend:
        def allgather(self, value):
            events.append("cpu_vote")
            values = [None] * world_size
            dist.all_gather_object(values, value, group=dist.group.WORLD)
            return values

    def allocate(*args, **kwargs):
        events.append("allocate")
        if rank == fail_rank:
            raise RuntimeError("injected local OOM")
        return object()

    def rendezvous(*args, **kwargs):
        events.append("rendezvous")
        return SimpleNamespace(get_buffer=lambda *a, **k: SimpleNamespace(data_ptr=lambda: 123))

    fake_torch = SimpleNamespace(dtype=object, device=object, Tensor=object,
        empty=lambda *a, **k: SimpleNamespace(element_size=lambda: 2))
    ns = functions(FI_ROOT / "comm/torch_symmetric_memory.py", ["_alloc_symm_buffer_bytes"], {
        "torch": fake_torch, "Any": object,
        "_enable_symm_mem_for_group": lambda name: None,
        "symm_mem": SimpleNamespace(empty=allocate, rendezvous=rendezvous),
    })
    failed = False
    try:
        ns["_alloc_symm_buffer_bytes"](128, world_size, "bf16", "cuda", "tp",
            comm_backend=Backend())
    except RuntimeError as exc:
        failed = True
        assert "injected local OOM" in str(exc), str(exc)
    assert failed == (fail_rank >= 0), (rank, failed, events)
    assert events == (["allocate", "cpu_vote"] if fail_rank >= 0
                      else ["allocate", "cpu_vote", "rendezvous"]), events
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps({
        "rank": rank, "fail_rank": fail_rank, "failed_collectively": failed, "events": events}))
    dist.destroy_process_group()


if __name__ == "__main__":
    root = Path(os.environ.get("CC_TEST_RESULTS", "/tmp/cc-gloo-results"))
    for fail_rank in (-1, 0, 7):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        result_dir = root / f"fail-rank-{fail_rank}"
        result_dir.mkdir(parents=True, exist_ok=True)
        mp.spawn(rank_main, args=(8, port, fail_rank, str(result_dir)), nprocs=8, join=True)
        results = [json.loads(p.read_text()) for p in sorted(result_dir.glob("rank-*.json"))]
        assert len(results) == 8
        print(json.dumps({"scenario": fail_rank, "ranks": results, "passed": True}), flush=True)
