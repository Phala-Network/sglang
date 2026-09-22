"""Production source-method CPU tests, not a native import or GPU qualification."""

import ast
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRT = ROOT / "python/sglang/srt"


def source_functions(path, names, namespace=None):
    namespace = dict(namespace or {})
    namespace.setdefault("torch", torch)
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    for name in names:
        body = parsed.body
        for part in name.split("."):
            node = next(n for n in body if getattr(n, "name", None) == part)
            body = getattr(node, "body", [])
        selected.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            )
        ]
        + selected,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@contextmanager
def injected_modules(modules):
    # Restore only injected keys; lazy torch modules must retain registrations.
    absent = object()
    saved = {name: sys.modules.get(name, absent) for name in modules}
    try:
        sys.modules.update(modules)
        yield
    finally:
        for name, module in saved.items():
            if module is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class QuantizationContracts(unittest.TestCase):
    def setUp(self):
        self.convert = source_functions(
            SRT / "layers/quantization/fp8_utils.py",
            ["block_fp8_scale_to_mxfp8_e8m0"],
            {"ceil_div": lambda n, d: (n + d - 1) // d},
        )["block_fp8_scale_to_mxfp8_e8m0"]

    def test_scale_expansion_bit_exact_partial_row(self):
        actual = self.convert(torch.tensor([[1, 2], [0.5, 4]]), (33, 64), [32, 32])
        self.assertEqual(actual.shape, (33, 2))
        self.assertTrue(
            torch.equal(actual[:32], torch.tensor([[127, 128]]).expand(32, 2))
        )
        self.assertEqual(actual[-1].tolist(), [126, 129])

    def test_invalid_scales_and_layout(self):
        for value in (float("inf"), -float("inf"), float("nan"), 0, -1, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.convert(torch.tensor([[value]]), (32, 32), [32, 32])
        for shape, blocks in (((32, 33), [32, 32]), ((32, 32), [32, 128])):
            with self.assertRaises(ValueError):
                self.convert(torch.ones(1, 1), shape, blocks)
        with self.assertRaises(ValueError):
            self.convert(torch.ones(2, 1), (32, 32), [32, 32])

    def test_finite_guard_ablation(self):
        # Execute the same source with only the finite guard removed in memory.
        path = SRT / "layers/quantization/fp8_utils.py"
        fn = next(
            n
            for n in ast.parse(path.read_text(encoding="utf-8")).body
            if getattr(n, "name", None) == "block_fp8_scale_to_mxfp8_e8m0"
        )
        removed = 0
        for node in ast.walk(fn):
            if isinstance(node, ast.BoolOp):
                before = len(node.values)
                node.values = [
                    v for v in node.values if "isfinite" not in ast.unparse(v)
                ]
                removed += before - len(node.values)
        self.assertEqual(removed, 1)
        ns = {"torch": torch, "ceil_div": lambda n, d: (n + d - 1) // d}
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias("annotations")], level=0
                ),
                fn,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
        ablated = ns[fn.name](torch.tensor([[float("inf")]]), (32, 32), [32, 32])
        self.assertTrue(bool((ablated == 255).all()))
        with self.assertRaises(ValueError):
            self.convert(torch.tensor([[float("inf")]]), (32, 32), [32, 32])

    def test_mxfp8_hardware_backend_layout_gates(self):
        class Backend:
            def __init__(self, name):
                self.name = name

            def is_flashinfer_cutlass(self):
                return self.name == "cutlass"

            def is_flashinfer_cutedsl(self):
                return self.name == "cute"

            def is_flashinfer(self):
                return self.name in ("cutlass", "cute")

            def is_unsupported(self):
                return self.name == "unsupported"

        state = dict(blackwell=True, available=True, backend=Backend("cute"))
        ns = source_functions(
            SRT / "layers/quantization/fp8_utils.py",
            ["resolve_block_fp8_mxfp8_backend", "can_serve_block_fp8_as_mxfp8"],
            dict(
                _is_cuda=True,
                get_fp8_gemm_runner_backend=lambda: state["backend"],
                get_platform=lambda: NS(is_blackwell=state["blackwell"]),
                is_flashinfer_available=lambda: state["available"],
                resolve_mxfp8_dense_gemm_backend=lambda: state["backend"],
                Mxfp8DenseGemmBackend=NS(UNSUPPORTED=Backend("unsupported")),
            ),
        )
        gate = ns["can_serve_block_fp8_as_mxfp8"]
        self.assertTrue(gate([32, 32], "ue8m0"))
        for blocks, fmt in (
            (None, "ue8m0"),
            ([32], "ue8m0"),
            ([32, 128], "ue8m0"),
            ([32, 32], None),
        ):
            self.assertFalse(gate(blocks, fmt))
        for field in ("blackwell", "available"):
            state[field] = False
            self.assertFalse(gate([32, 32], "ue8m0"))
            state[field] = True
        ns["_is_cuda"] = False
        self.assertFalse(gate([32, 32], "ue8m0"))
        ns["_is_cuda"] = True
        for backend in ("auto", "triton", "trtllm", "deep_gemm"):
            state["backend"] = Backend(backend)
            self.assertFalse(gate([32, 32], "ue8m0"))

    def test_hopper_empty_large_dtype_shape_fallbacks(self):
        calls = []

        def result(name):
            def run(x, weight, *args, **kwargs):
                calls.append(name)
                return torch.zeros(*x.shape[:-1], weight.shape[0], dtype=x.dtype)

            return run

        fn = source_functions(
            SRT / "layers/quantization/fp8_utils.py",
            ["flashinfer_deepgemm_w8a8_block_fp8_linear_with_fallback"],
            dict(
                deep_gemm_wrapper=NS(ENABLE_JIT_DEEPGEMM=True),
                deepgemm_w8a8_block_fp8_linear_with_fallback=result("deepgemm"),
                triton_w8a8_block_fp8_linear=result("triton"),
                fp8_blockscale_gemm_sm90=result("hopper"),
            ),
        )["flashinfer_deepgemm_w8a8_block_fp8_linear_with_fallback"]
        for rows, dtype, n, k, expected in (
            (0, torch.bfloat16, 64, 128, "deepgemm"),
            (32, torch.bfloat16, 64, 128, "deepgemm"),
            (1, torch.bfloat16, 64, 128, "hopper"),
            (31, torch.bfloat16, 64, 128, "hopper"),
            (1, torch.float16, 64, 128, "triton"),
            (1, torch.bfloat16, 63, 128, "triton"),
            (1, torch.bfloat16, 64, 127, "triton"),
        ):
            fn(
                torch.zeros(rows, k, dtype=dtype),
                torch.zeros(n, k),
                [128, 128],
                torch.ones(1, 1),
            )
            self.assertEqual(calls[-1], expected)

    def test_routed_abi_identity_and_negative_dtype_guards(self):
        fn = source_functions(
            SRT / "layers/moe/moe_runner/flashinfer_trtllm.py",
            ["_make_deferred_finalize_output"],
            {"FlashInferTrtllmDeferredFinalizeOutput": NS},
        )["_make_deferred_finalize_output"]
        for dtype in (torch.bfloat16, torch.float32):
            weights = torch.tensor([[0.25657165, 0.24222815]], dtype=dtype)
            result = (
                torch.zeros(2, 8, dtype=torch.bfloat16),
                weights,
                torch.arange(2, dtype=torch.int32),
            )
            self.assertIs(
                fn(result, top_k=2, expected_expert_weights_dtype=dtype).expert_weights,
                weights,
            )
            other = torch.float32 if dtype == torch.bfloat16 else torch.bfloat16
            with self.assertRaisesRegex(RuntimeError, "dtype mismatch"):
                fn(result, top_k=2, expected_expert_weights_dtype=other)
        with self.assertRaisesRegex(RuntimeError, "only supports"):
            fn(result, top_k=2, expected_expert_weights_dtype=torch.float16)


class CacheContracts(unittest.TestCase):
    def test_transfer_boundaries_and_layout(self):
        ns = source_functions(
            SRT / "mem_cache/deepseek_v4_compress_state.py",
            ["c4_state_transfer_indices", "request_scoped_state_transfer_indices"],
            {"np": np},
        )
        fn = ns["request_scoped_state_transfer_indices"]
        for ratio in (2, 128):
            for online in (False, True):
                self.assertEqual(
                    fn(3, ratio, ratio=ratio, online=online, ring_size=ratio * 2).size,
                    0,
                )
                self.assertEqual(
                    fn(
                        3, ratio + 1, ratio=ratio, online=online, ring_size=ratio * 2
                    ).tolist(),
                    [3 if online else 7],
                )
        with self.assertRaises(AssertionError):
            fn(3, 1, ratio=128, online=False, ring_size=129)
        for ring in (0, 4, 9):
            with self.assertRaises(ValueError):
                ns["c4_state_transfer_indices"](1, 13, ring_size=ring)
        self.assertEqual(
            ns["c4_state_transfer_indices"](1, 13, ring_size=8).tolist(),
            [8, 9, 10, 11, 12],
        )

    def test_backup_guards_no_hit_accounting(self):
        fn = source_functions(
            SRT / "mem_cache/unified_radix_cache.py",
            ["UnifiedRadixCache._backup_completed_write_through_chunk"],
        )["_backup_completed_write_through_chunk"]
        actions = []
        tree = NS(
            enable_hicache=True,
            is_root=lambda n: n == 0,
            is_backuped=lambda n: n == 2,
            node_by_id=lambda n: n,
            _build_backup_kv_action=lambda n: ("backup", n),
        )
        cache = NS(
            tree_core=tree,
            cache_controller=NS(write_policy="write_through"),
            _tree_core_backend="python",
            buffer_pipeline=None,
            linker=None,
            _apply_cache_actions=actions.extend,
        )
        fn(cache, NS(last_device_node=1), chunked=True)
        self.assertEqual(actions, [("backup", 1)])
        actions.clear()
        for node in (None, 0, 2):
            fn(cache, NS(last_device_node=node), chunked=True)
        fn(cache, NS(last_device_node=1), chunked=False)
        for field, bad in (
            ("_tree_core_backend", "rust"),
            ("buffer_pipeline", object()),
            ("linker", object()),
            ("cache_controller", None),
        ):
            original = getattr(cache, field)
            setattr(cache, field, bad)
            fn(cache, NS(last_device_node=1), chunked=True)
            setattr(cache, field, original)
        cache.cache_controller.write_policy = "write_through_selective"
        fn(cache, NS(last_device_node=1), chunked=True)
        cache.cache_controller.write_policy = "write_through"
        tree.enable_hicache = False
        fn(cache, NS(last_device_node=1), chunked=True)
        self.assertEqual(actions, [])

    def test_engram_verify_commits_only_accepted_prefix(self):
        fn = source_functions(
            SRT / "layers/engram.py",
            ["EngramHasher.commit_after_verify"],
            {"_cuda_kernels": lambda t: False},
        )["commit_after_verify"]
        state = NS(
            max_ngram_size=4,
            history=torch.tensor([[1, 2, 3], [7, 8, 9], [4, 5, 6]], dtype=torch.int32),
        )
        fn(
            state,
            torch.tensor([[10, 11, 12], [20, 21, 22]]),
            torch.tensor([0, 2]),
            torch.tensor([1, 2]),
        )
        self.assertEqual(state.history.tolist(), [[2, 3, 10], [7, 8, 9], [6, 20, 21]])


class DenseContracts(unittest.TestCase):
    def test_dense_tile_publish_consume_underfill_and_budget(self):
        candidate = source_functions(
            SRT / "layers/attention/dsv4/candidate_indexer.py",
            [
                "_candidate_block_topk",
                "select_candidate_block_ids",
                "candidate_block_mask",
                "mask_topk_scores",
            ],
            {"F": F},
        )
        dense = source_functions(
            SRT / "layers/attention/dsv4/dense_prefill_indexer.py",
            ["dense_prefill_topk", "_select_tile"],
            dict(candidate, PrefillCandidateBlocks=NS, _SCORE_BUDGET_BYTES=64),
        )
        calls = []
        logits = torch.tensor(
            [
                [8, 7, 6, 5, 4, 3, 2, 1],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [4, 8, 3, 7, 2, 6, 1, 5],
                [1, 4, 7, 2, 5, 8, 3, 6],
                [8, 1, 7, 2, 6, 3, 5, 4],
            ],
            dtype=torch.float32,
        )
        starts, lengths, ids = (
            torch.tensor([0, 0, 0, 10, 10]),
            torch.tensor([5, 7, 8, 6, 8]),
            torch.arange(5),
        )

        def mqa(q, kv, weights, start, end, unused, width):
            calls.append((len(q[0]), width))
            result = logits[q[0]].clone()
            return result.masked_fill_(
                torch.arange(width)[None] >= (end - start)[:, None], -torch.inf
            )

        def topk(scores, lens, out_offsets, out_indices):
            out_indices.copy_(
                (
                    scores.topk(out_indices.shape[-1], dim=-1).indices
                    + out_offsets[:, None]
                ).to(torch.int32)
            )

        kernels, dg = (
            ModuleType("sglang.kernels.ops.attention.dsv4"),
            ModuleType("deep_gemm"),
        )
        kernels.topk_transform_ragged_v2, dg.fp8_fp4_mqa_logits = topk, mqa
        kwargs = dict(
            q=(ids, ids),
            kv=(None, None),
            weights=torch.ones(5),
            starts=starts,
            lengths=lengths,
            request_lengths=[(3, 8), (2, 8)],
            topk=3,
            candidate_topk_blocks=1,
            candidate_block_size=2,
        )
        with injected_modules(
            {"deep_gemm": dg, "sglang.kernels.ops.attention.dsv4": kernels}
        ):
            selected, blocks = dense["dense_prefill_topk"](
                **kwargs, publish_candidates=True, candidates=None
            )
            consumed, none = dense["dense_prefill_topk"](
                **kwargs, publish_candidates=False, candidates=blocks
            )
        self.assertIsNone(none)
        self.assertEqual(selected.shape, (5, 3))
        self.assertEqual([b.shape for b in blocks.request_blocks], [(3, 1), (2, 1)])
        self.assertTrue(all(n * w * 4 <= 64 for n, w in calls))
        self.assertEqual(len(calls), 6)
        mask = candidate["candidate_block_mask"](torch.cat(blocks.request_blocks), 8, 2)
        for row in range(5):
            valid = consumed[row][consumed[row] >= 0] - starts[row]
            self.assertTrue(bool(mask[row, valid.long()].all()))
            self.assertTrue(bool((valid < lengths[row]).all()))
            self.assertIn(-1, consumed[row].tolist())


class CommunicationContracts(unittest.TestCase):
    def test_allreduce_rejects_wrong_group_and_unaligned_mnnvl(self):
        workspace = NS(
            initialized=True,
            workspace=object(),
            world_size=4,
            group=("group", None),
            backend="mnnvl",
            is_buffer_size_sufficient=lambda **kwargs: True,
        )
        ns = source_functions(
            SRT / "layers/flashinfer_comm_fusion.py",
            ["can_use_flashinfer_allreduce"],
            dict(
                _flashinfer_allreduce_unavailable=False,
                _flashinfer_comm=object(),
                _get_workspace_manager=lambda attn: workspace,
            ),
        )
        fn = ns["can_use_flashinfer_allreduce"]
        kwargs = dict(
            use_attn_tp_group=False,
            expected_world_size=4,
            expected_group_key=("group", None),
        )
        self.assertTrue(fn(torch.zeros(2, 8, dtype=torch.bfloat16), **kwargs))
        self.assertFalse(fn(torch.zeros(2, 7, dtype=torch.bfloat16), **kwargs))
        workspace.group = ("different_group_same_size", None)
        self.assertFalse(fn(torch.zeros(2, 8, dtype=torch.bfloat16), **kwargs))
        workspace.group = ("group", None)
        ns["_flashinfer_allreduce_unavailable"] = True
        self.assertFalse(fn(torch.zeros(2, 8, dtype=torch.bfloat16), **kwargs))

    def test_separate_planes_and_target_only_medium_guards(self):
        ns = source_functions(
            ROOT / "python/sglang/kernels/ops/communication/all_reduce_fusion.py",
            ["register_comm", "get_registered_comm"],
            {"DEFAULT_COMM_KEY": "default", "_COMM_MAP": {}},
        )
        small, medium = NS(world_size=4), NS(world_size=4)
        ns["register_comm"](small)
        ns["register_comm"](medium, "dsv41_medium")
        self.assertIs(ns["get_registered_comm"](4), small)
        self.assertIs(ns["get_registered_comm"](4, "dsv41_medium"), medium)
        with self.assertRaises(AssertionError):
            ns["register_comm"](NS(world_size=4))
        module = ModuleType("sglang.kernels.ops.communication")
        module.all_reduce_fusion = NS(
            DEFAULT_COMM_KEY="default", DSV41_MEDIUM_COMM_KEY="dsv41_medium"
        )
        fn = source_functions(
            SRT / "models/deepseek_v2.py",
            ["_select_fused_finalize_comm_key"],
            {
                "_FUSED_FINALIZE_ALL_REDUCE_MAX_TOKENS": 96,
                "_FUSED_FINALIZE_ALL_REDUCE_MEDIUM_MAX_TOKENS": 384,
            },
        )["_select_fused_finalize_comm_key"]
        with injected_modules({"sglang.kernels.ops.communication": module}):
            for rows in (97, 288, 384):
                self.assertEqual(
                    fn(
                        rows, is_nextn=False, is_target_verify=True, medium_enabled=True
                    ),
                    "dsv41_medium",
                )
                for nextn, target, enabled in (
                    (True, True, True),
                    (False, False, True),
                    (False, True, False),
                ):
                    self.assertIsNone(
                        fn(
                            rows,
                            is_nextn=nextn,
                            is_target_verify=target,
                            medium_enabled=enabled,
                        )
                    )
            for rows in (0, 385):
                self.assertIsNone(
                    fn(rows, is_nextn=False, is_target_verify=True, medium_enabled=True)
                )


class ConsumerContracts(unittest.TestCase):
    def test_vision_feature_span_and_weight_remapping(self):
        ns = source_functions(
            SRT / "models/deepseek_v4.py",
            [
                "DeepseekV4ForCausalLM.get_image_feature",
                "DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format",
            ],
            {
                "GPU_PLAN_KEY": "gpu_plan",
                "image_token_types": lambda h, w: torch.tensor([0, 1, 1, 2, 3]),
            },
        )
        seen = []
        owner = NS(
            image_start=torch.tensor([10.0, 10.0]),
            image_end=torch.tensor([30.0, 30.0]),
            image_newline=torch.tensor([20.0, 20.0]),
            tp_size=4,
            config=NS(vision_downsample_ratio=1, hidden_size=2),
            vision=lambda pixels, h, w: seen.append((h, w)) or pixels,
            aligner=lambda pixels, h, w: pixels,
        )
        item = NS(
            reconstruct=lambda device, **kw: seen.append(kw),
            n_vit_h=1,
            n_vit_w=2,
            feature=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            model_specific_data={},
        )
        spans = ns["get_image_feature"](owner, [item])
        self.assertEqual(
            spans[0].tolist(), [[10, 10], [1, 2], [3, 4], [20, 20], [30, 30]]
        )
        self.assertEqual(seen, [{"ipc_consumer_count": 4}, (1, 2)])
        remap = ns["remap_weight_name_to_dpsk_hf_format"]
        if isinstance(remap, staticmethod):
            remap = remap.__func__
        self.assertEqual(
            remap("vision.blocks.0.attn.wqkv.weight"),
            "vision.blocks.0.attn.qkv_proj.weight",
        )
        self.assertEqual(remap("aligner.w1.weight"), "aligner.w1.weight")
        self.assertEqual(
            remap("layers.0.engram.wkv.scale"),
            "model.layers.0.engram.wkv.weight_scale_inv",
        )

    def test_low_ratio_hicache_builds_actual_pool_entries(self):
        names = {}
        for ratio in (1, 2):
            for suffix in ("", "_INDEXER", "_INDEXER_SCALE"):
                name = f"DEEPSEEK_V4_C{ratio}{suffix}"
                names[name] = name
        ns = source_functions(
            SRT / "mem_cache/hybrid_cache/hybrid_pool_assembler.py",
            ["_dsv4_low_ratio_entries"],
            dict(
                PoolName=NS(**names),
                build_pool_entry=lambda **kw: NS(**kw),
                DeepSeekV4PagedHostPool=lambda **kw: NS(**kw),
                get_memory=lambda: NS(hicache_mem_layout="layer_first"),
                _get_allocator_type=lambda: "cpu_fixture",
            ),
        )
        kv = NS(
            sources_by_ratio={1: [0], 2: [1]},
            start_layer=0,
            kv_pools={
                r: NS(kv_buffer=[torch.zeros(4, 64 // r, dtype=torch.uint8)])
                for r in (1, 2)
            },
            index_pools={
                r: NS(
                    page_size=2,
                    index_k_with_scale_buffer=None,
                    index_k_payload_buffer=[
                        torch.zeros(16 // r, 2, 4, dtype=torch.uint8)
                    ],
                    index_k_scale_buffer=[torch.zeros(16 // r, 2, 1)],
                )
                for r in (1, 2)
            },
        )
        entries = ns["_dsv4_low_ratio_entries"](kv, 8, 8, 2)
        self.assertEqual(len(entries), 6)
        self.assertEqual({e.name for e in entries}, set(names))
        for entry in entries:
            self.assertEqual(entry.host_pool.num_host_pages, 8)
            self.assertTrue(entry.host_pool.page_aligned_only)
            self.assertEqual(entry.host_pool.device_buffers[0].shape[0], 4)
        kv.sources_by_ratio = {}
        self.assertEqual(ns["_dsv4_low_ratio_entries"](kv, 8, 8, 2), [])

    def test_python_sidecar_backup_load_prefetch_follow_full_indices(self):
        phases = NS(BACKUP_HOST="backup", LOAD_BACK="load", PREFETCH="prefetch")
        pool_names = NS(KV="kv", SWA="swa", MAMBA="mamba")
        fn = source_functions(
            SRT / "mem_cache/unified_radix_cache.py",
            ["UnifiedRadixCache._build_sidecar_transfers"],
            dict(
                CacheTransferPhase=phases,
                PoolName=pool_names,
                PoolTransfer=lambda **kw: NS(**kw),
            ),
        )["_build_sidecar_transfers"]
        names = [
            f"c{ratio}{suffix}"
            for ratio in (1, 2)
            for suffix in ("", "_indexer", "_indexer_scale")
        ]
        cache = NS(
            sidecar_pool_specs=[
                NS(pool_name=name, indices_from_pool="kv", hit_policy="all")
                for name in names
            ]
        )
        transfer = NS(
            device_indices=torch.tensor([1, 2]),
            host_indices=torch.tensor([9, 10]),
            keys=["key"],
        )
        for phase in (phases.BACKUP_HOST, phases.LOAD_BACK, phases.PREFETCH):
            result = fn(cache, phase, transfer, {})
            self.assertEqual([r.name for r in result], names)
            self.assertTrue(all(r.indices_from_pool == "kv" for r in result))
            self.assertTrue(all(r.keys is transfer.keys for r in result))
        transfer.device_indices = transfer.host_indices = None
        self.assertEqual(fn(cache, phases.BACKUP_HOST, transfer, {}), [])
        self.assertEqual(fn(cache, phases.LOAD_BACK, transfer, {}), [])
        self.assertEqual(len(fn(cache, phases.PREFETCH, transfer, {})), 6)

    def test_bitmap_host_capacity_source_boundary_simulation(self):
        # Source/host arithmetic only: the CUDA sort/atomic bitmap is not executed.
        path = (
            ROOT
            / "python/sglang/kernels/jit/csrc/deepseek_v4/candidate_block_table.cuh"
        )
        text = path.read_text(encoding="utf-8")
        for guard in (
            "CandidateBlockTableConfig<4>",
            "CandidateBlockTableConfig<8>",
            "num_pages <= LargeC::kMaxBlocks / blocks_per_page",
            "page_table.size(1)",
            "candidate row exceeds page table",
            "selected candidates must be distinct and complete",
            "kVecWords = 4",
        ):
            self.assertIn(guard, text)
        page_size = 256
        legacy_blocks, large_blocks = 4 * 1024 * 32, 8 * 1024 * 32
        for tokens, expected_words in ((1048576, 4), (1048577, 8), (2097152, 8)):
            pages = (tokens + page_size - 1) // page_size
            addressable = pages * (page_size // 8)
            self.assertLessEqual(addressable, large_blocks)
            self.assertEqual(4 if addressable <= legacy_blocks else 8, expected_words)
        # Ablating the large specialization loses the speculative scratch block.
        self.assertGreater((1048577 + 7) // 8, legacy_blocks)
        self.assertGreater(((2097153 + 255) // 256) * 32, large_blocks)


if __name__ == "__main__":
    unittest.main()
