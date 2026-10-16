"""PyTorch regressions for graph-owned accept buffers and delayed consumers."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import msgspec
import sglang
import torch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FIELDS = (
    "correct_len",
    "bonus",
    "cap_trim_lens",
    "commit_lens",
    "new_seq_lens",
    "out_tokens",
)


def load_read_accept():
    # Load the installed implementation without importing unrelated CUDA-only
    # kernels. Both the result structure and method are the actual source AST.
    path = (
        Path(sglang.__file__).parent
        / "srt/speculative/dspark_components/dspark_verify.py"
    )
    tree = ast.parse(path.read_text())
    result = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AcceptOuts"
    )
    owner = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DsparkVerifyEpilogue"
    )
    method = next(
        n
        for n in owner.body
        if isinstance(n, ast.FunctionDef) and n.name == "read_accept"
    )
    namespace = {"msgspec": msgspec, "torch": torch}
    exec(
        compile(ast.Module(body=[result, method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["read_accept"]


def buffers(device="cpu"):
    values = {}
    for index, field in enumerate(FIELDS):
        shape = (8, 6) if field == "out_tokens" else (8,)
        dtype = (
            torch.int32 if field in ("cap_trim_lens", "commit_lens") else torch.int64
        )
        values[field + "_buf"] = (
            torch.arange(
                48 if len(shape) == 2 else 8, dtype=dtype, device=device
            ).reshape(shape)
            + index * 100
        )
    return SimpleNamespace(**values)


class TestAcceptResultOwnership(unittest.TestCase):
    def test_all_fields_survive_next_graph_buffer_overwrite(self):
        read_accept = load_read_accept()
        for bs in (0, 1, 4, 8):
            with self.subTest(bs=bs):
                epilogue = buffers()
                expected = {
                    field: getattr(epilogue, field + "_buf")[:bs].clone()
                    for field in FIELDS
                }
                result = read_accept(epilogue, bs)
                flat_tokens = result.out_tokens.reshape(-1)
                for field in FIELDS:
                    source = getattr(epilogue, field + "_buf")
                    actual = getattr(result, field)
                    self.assertEqual(actual.dtype, source.dtype)
                    self.assertEqual(actual.device, source.device)
                    self.assertEqual(actual.shape, expected[field].shape)
                    if bs:
                        self.assertNotEqual(actual.data_ptr(), source.data_ptr())
                    source.fill_(-1000)
                    torch.testing.assert_close(actual, expected[field], rtol=0, atol=0)
                torch.testing.assert_close(
                    flat_tokens, expected["out_tokens"].reshape(-1), rtol=0, atol=0
                )

    def test_rank_consumers_observe_same_iteration_despite_delayed_copy(self):
        read_accept = load_read_accept()
        ranks = [buffers(), buffers()]
        for rank in ranks:
            rank.commit_lens_buf.fill_(3)
        first, delayed = (read_accept(rank, 4) for rank in ranks)
        immediate_host = first.commit_lens.clone()
        ranks[1].commit_lens_buf.fill_(5)  # next graph before rank1's D2H consumer
        delayed_host = delayed.commit_lens.clone()
        torch.testing.assert_close(immediate_host, delayed_host, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA test target")
    def test_cuda_delayed_consumer_storage_survives_next_overwrite(self):
        read_accept = load_read_accept()
        epilogue = buffers("cuda")
        forward = torch.cuda.Stream()
        copy = torch.cuda.Stream()
        forward.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(forward):
            result = read_accept(epilogue, 8)
        copy.wait_stream(forward)
        host = torch.empty((8,), dtype=torch.int32, pin_memory=True)
        with torch.cuda.stream(copy):
            torch.cuda._sleep(5_000_000)
            host.copy_(result.commit_lens, non_blocking=True)
            result.commit_lens.record_stream(copy)
        with torch.cuda.stream(forward):
            epilogue.commit_lens_buf.fill_(-1000)
        copy.synchronize()
        torch.testing.assert_close(host, torch.arange(8, dtype=torch.int32) + 300)


if __name__ == "__main__":
    unittest.main()
