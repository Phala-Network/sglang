"""Source-level partition/adapter checks; no kernels or real collective tested."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
INDEXER = ROOT / "python/sglang/srt/layers/attention/dsv4/v41_indexer"


def load(nodes, ns):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<actual-source>", "exec"), ns)


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    @property
    def shape(self):
        return self.data.shape

    def new_empty(self, shape):
        return Tensor(np.empty(shape, dtype=self.data.dtype))

    def contiguous(self):
        return self

    def __getitem__(self, key):
        return Tensor(self.data[key])


ns = {}
common = ast.parse(
    (ROOT / "python/sglang/srt/utils/common.py").read_text(encoding="utf-8")
)
load(
    [
        n
        for n in common.body
        if isinstance(n, ast.FunctionDef) and n.name in {"ceil_align", "ceil_div"}
    ],
    ns,
)
tree = ast.parse((INDEXER / "types.py").read_text(encoding="utf-8"))
row = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RowShard")
# Keep exact methods; replace msgspec's field-only constructor for CPU isolation.
row.bases = []
row.keywords = []
load([row], ns)
Row = ns["RowShard"]


def shard(**kwargs):
    obj = Row()
    obj.__dict__.update(kwargs)
    return obj


tree = ast.parse(
    (ROOT / "python/sglang/srt/layers/attention/deepseek_v4_backend.py").read_text(
        encoding="utf-8"
    )
)
backend = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4AttnBackend"
)
methods = [
    n
    for n in backend.body
    if isinstance(n, ast.FunctionDef)
    and n.name in {"_indexer_row_shard", "_make_low_ratio_prefill_indexer_inputs"}
]
gate_ns = dict(
    RowShard=shard,
    _INDEXER_TP_SHARD_MIN_ROWS=512,
    PrefillInputs=lambda **kwargs: NS(**kwargs),
    _as_int_list=list,
    torch=NS(int32="int32", tensor=lambda value, **kwargs: value),
)
load(
    [
        ast.ClassDef(
            name="Backend", bases=[], keywords=[], body=methods, decorator_list=[]
        )
    ],
    gate_ns,
)


class TestRowShardCPU(unittest.TestCase):
    def test_partitions_and_gather(self):
        cases = 0
        for world in [1, 2, 4]:
            for heads in [1, 2, 8, 16, 32, 64, 128]:
                for rows in [0, 1, 7, 511, 512, 513, 8191, 8192, 8193]:
                    with self.subTest(world=world, heads=heads, rows=rows):
                        shards = [
                            shard(rank=r, world=world, row_align=128 // heads)
                            for r in range(world)
                        ]
                        parts = [s.local_rows(rows) for s in shards]
                        self.assertEqual(
                            [i for p in parts for i in range(p.start, p.stop)],
                            list(range(rows)),
                        )
                        per = shards[0].rows_per_rank(rows)
                        self.assertEqual(per % (128 // heads), 0)
                        contributions = []
                        for part in parts:
                            local = np.full((per, 2), -999, dtype=np.int32)
                            local[: part.stop - part.start] = np.arange(
                                part.start, part.stop
                            )[:, None]
                            contributions.append(local)
                        full = np.concatenate(contributions)
                        for rank, s in enumerate(shards):

                            def gather(output, local):
                                self.assertEqual(output.shape, (world * per, 2))
                                np.testing.assert_array_equal(
                                    local.data, contributions[rank]
                                )
                                output.data[:] = full

                            s.group = NS(all_gather_into_tensor=gather)
                            out = s.gather(Tensor(contributions[rank]), rows)
                            np.testing.assert_array_equal(
                                out.data, np.repeat(np.arange(rows)[:, None], 2, axis=1)
                            )
                        cases += 1
        self.assertEqual(cases, 189)

    def test_gate_and_cp(self):
        obj = gate_ns["Backend"]()
        obj.forward_metadata = NS(
            late_layer_tail=None, core_metadata=NS(page_table=None)
        )
        layer = NS(indexer=NS(n_heads=8), layer_id=1, compress_ratio=1, freqs_cis=None)
        for enabled in [False, True]:
            gate_ns["envs"] = NS(
                SGLANG_OPT_DSV4_INDEXER_TP_SHARD=NS(get=lambda: enabled)
            )
            for world in [1, 2]:
                gate_ns["get_parallel"] = lambda: NS(
                    attn_tp_size=world, attn_tp_rank=0, attn_tp_group="attention-tp"
                )
                for rows in [511, 512, 513, 8192]:
                    for cp in [False, True]:
                        batch = NS(
                            extend_seq_lens_cpu=[rows],
                            extend_seq_lens=[rows],
                            req_pool_indices=[0],
                            seq_lens_cpu=[rows],
                        )
                        inputs = obj._make_low_ratio_prefill_indexer_inputs(
                            layer,
                            NS(shape=(rows,), device=None),
                            None,
                            None,
                            None,
                            batch,
                            [rows] if cp else None,
                        )
                        self.assertEqual(
                            inputs.row_shard is not None,
                            enabled and world > 1 and rows >= 512 and not cp,
                        )
                        if inputs.row_shard is not None:
                            self.assertEqual(inputs.row_shard.group, "attention-tp")
                            self.assertEqual(inputs.row_shard.row_align, 16)

    def test_score_slices_preserve_request_offsets(self):
        source = ast.parse((INDEXER / "scoring.py").read_text(encoding="utf-8"))
        func = next(
            n
            for n in source.body
            if isinstance(n, ast.FunctionDef) and n.name == "score_tiles"
        )
        seen = []

        def kernel(**kwargs):
            seen.append(kwargs)
            yield slice(0, len(kwargs["starts"])), None

        score_ns = dict(
            flat_index_logits_tiles=kernel, _DEEP_GEMM_SCORE_BUDGET_BYTES=2 << 30
        )
        load([func], score_ns)
        data = NS(
            num_rows=7,
            q_fp4=np.arange(7),
            q_sf=np.arange(7) + 10,
            weights=np.arange(7) + 20,
            request_starts=np.array([0, 0, 0, 100, 100, 100, 100]),
            compress_lens=np.arange(7) + 1,
            lens_per_request=[3, 4],
        )
        for rows in [slice(2, 5), slice(7, 7), None]:
            list(
                score_ns["score_tiles"](data, ("k", "scale"), width_align=4, rows=rows)
            )
            actual = seen[-1]
            sl = slice(0, 7) if rows is None else rows
            for key, attr in [
                ("starts", "request_starts"),
                ("lengths", "compress_lens"),
                ("weights", "weights"),
            ]:
                np.testing.assert_array_equal(actual[key], getattr(data, attr)[sl])
            np.testing.assert_array_equal(actual["q"][0], data.q_fp4[sl])
            np.testing.assert_array_equal(actual["q"][1], data.q_sf[sl])
            self.assertEqual(actual["context_lengths"], [3, 4])
        # Both producers must preserve control's flattened-K offset convention.
        for name, path in [
            ("dense_prefill_topk", INDEXER / "scoring.py"),
            ("select_prefill_rows", INDEXER / "sparse_table.py"),
        ]:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            fn = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            calls = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "topk_transform_ragged_v2"
            ]
            offsets = next(k.value for k in calls[0].keywords if k.arg == "out_offsets")
            self.assertEqual(ast.unparse(offsets), "starts[tile]")


if __name__ == "__main__":
    unittest.main()
