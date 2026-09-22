"""Dependency-free CPU/source checks, not CUDA numerical/capture acceptance.

Run directly with Python. Execute selected production AST with scalar pointer
fixtures; Triton compilation, GPU concurrency and performance remain untested.
"""

import ast
import math
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[5]
SRT = ROOT / "python/sglang/srt"
ALIGN = SRT / "layers/moe/moe_runner/triton_utils/moe_align_block_size.py"
FUSED = SRT / "layers/moe/fused_moe_triton/fused_marlin_moe.py"
DENSE = SRT / "layers/quantization/marlin_utils_fp4.py"
GRAPH = SRT / "model_executor/model_runner_components/cuda_graph_setup.py"


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def load_functions(path, names, namespace):
    nodes = [
        node
        for node in tree(path).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


class Pointer:
    def __init__(self, data, offset=0):
        self.data = data
        self.offset = offset
        self.device = "cpu"

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset)

    def numel(self):
        return len(self.data)


def allocate(shape, value=0, **kwargs):
    return Pointer([value] * (shape if isinstance(shape, int) else math.prod(shape)))


class ScalarTL:
    program = 0
    cdiv = staticmethod(lambda x, y: (x + y - 1) // y)

    def program_id(self, axis):
        return self.program

    @staticmethod
    def load(ptr):
        assert 0 <= ptr.offset < len(ptr.data)
        return ptr.data[ptr.offset]

    @staticmethod
    def store(ptr, value):
        assert 0 <= ptr.offset < len(ptr.data)
        ptr.data[ptr.offset] = value


class Kernel:
    def __init__(self, function, tl):
        self.function = function
        self.tl = tl

    def __getitem__(self, grid):
        def launch(*args):
            # Reverse launch order also exercises exclusive chunk-prefix ranks.
            for program in reversed(range(grid[0])):
                self.tl.program = program
                self.function(*args)

        return launch


def alignment_namespace():
    tl = ScalarTL()
    namespace = {
        "torch": SimpleNamespace(
            full=allocate, zeros=allocate, empty=allocate, int32="int32"
        ),
        "triton": SimpleNamespace(cdiv=tl.cdiv),
        "tl": tl,
        "_is_cuda": True,
        "SMALL_NUMEL_LIMIT": 64,
        "_CUDA_SMALL_BATCH_MAX_BUCKETS": 64,
        "_SGLANG_EXPERIMENTAL_LORA_OPTI": False,
        "sgl_moe_align_block_size": Mock(),
        "moe_align_small_numel": Mock(),
    }
    kernels = [
        "_deterministic_align_count",
        "_deterministic_align_chunk_prefix",
        "_deterministic_align_expert_prefix",
        "_deterministic_align_scatter",
    ]
    load_functions(
        ALIGN,
        [*kernels, "_moe_align_block_size_deterministic", "moe_align_block_size"],
        namespace,
    )
    for name in kernels:
        namespace[name] = Kernel(namespace[name], tl)
    return namespace


def assignment_expression(path, name):
    return next(
        node.value
        for node in ast.walk(tree(path))
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    )


class MarlinBreakableSourceContract(unittest.TestCase):
    def test_scalar_alignment_matches_stable_reference(self):
        namespace = alignment_namespace()
        rng = random.Random(42)
        for experts, entries, block in [
            (128, 0, 8),
            (128, 6, 8),
            (128, 486, 8),
            (128, 2022, 32),
            (32, 256, 64),
            (128, 49152, 64),
        ]:
            routes = [rng.randrange(-1, experts) for _ in range(entries)]
            for ignore in (False, True):
                with self.subTest(experts=experts, entries=entries, ignore=ignore):
                    actual = namespace["_moe_align_block_size_deterministic"](
                        Pointer(routes), block, experts, ignore
                    )
                    expected_ids, expected_experts = [], []
                    for expert in range(0 if ignore else -1, experts):
                        ids = [i for i, route in enumerate(routes) if route == expert]
                        padded = ScalarTL.cdiv(len(ids), block) * block
                        expected_ids.extend(ids + [entries] * (padded - len(ids)))
                        expected_experts.extend([expert] * (padded // block))
                    count = actual[2].data[0]
                    self.assertEqual(count, len(expected_ids))
                    self.assertEqual(actual[0].data[:count], expected_ids)
                    self.assertEqual(actual[1].data[: count // block], expected_experts)

    def test_dispatch_defaults_and_stable_tiny_path(self):
        for cuda, entries, experts, ignore, deterministic, expected in [
            (True, 65, 128, False, False, "generic"),
            (True, 64, 128, False, False, "tiny"),
            (True, 64, 128, False, True, "tiny"),
            (True, 65, 128, False, True, "stable"),
            (True, 64, 63, False, True, "stable"),
            (True, 64, 64, False, True, "tiny"),
            (True, 64, 128, True, True, "stable"),
            (False, 64, 128, False, True, "stable"),
            (False, 64, 128, False, False, "generic"),
        ]:
            with self.subTest(case=(cuda, entries, experts, ignore, deterministic)):
                ns = alignment_namespace()
                ns["_is_cuda"] = cuda
                stable = ns["_moe_align_block_size_deterministic"] = Mock()
                kwargs = {"deterministic": True} if deterministic else {}
                ns["moe_align_block_size"](
                    Pointer([0] * entries), 8, experts, ignore, **kwargs
                )
                self.assertEqual(stable.called, expected == "stable")
                self.assertEqual(ns["moe_align_small_numel"].called, expected == "tiny")
                self.assertEqual(
                    ns["sgl_moe_align_block_size"].called, expected == "generic"
                )

    def test_marlin_requests_stable_alignment_and_fp32_both_gemms(self):
        calls = [node for node in ast.walk(tree(FUSED)) if isinstance(node, ast.Call)]
        align = [
            node
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "moe_align_block_size"
        ]
        self.assertEqual(len(align), 1)
        self.assertTrue(
            any(
                k.arg == "deterministic" and ast.literal_eval(k.value)
                for k in align[0].keywords
            )
        )
        gemms = [
            node
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "moe_wna16_marlin_gemm"
        ]
        self.assertEqual(len(gemms), 2)
        for gemm in gemms:
            kwargs = {k.arg: k.value for k in gemm.keywords}
            self.assertTrue(ast.literal_eval(kwargs["use_fp32_reduce"]))
            self.assertEqual(kwargs["use_atomic_add"].id, "use_atomic_add")

    def test_fp4_atomic_gate_preserves_other_marlin_defaults(self):
        expr = compile(
            ast.Expression(assignment_expression(FUSED, "use_atomic_add")),
            str(FUSED),
            "eval",
        )
        for dtype in ("half", "bfloat16"):
            for sm in (8, 9, 12):
                for mxfp4, nvfp4 in ((False, False), (True, False), (False, True)):
                    ns = {
                        "hidden_states": SimpleNamespace(dtype=dtype, device="cuda"),
                        "torch": SimpleNamespace(
                            half="half",
                            cuda=SimpleNamespace(
                                get_device_capability=lambda _: (sm, 0)
                            ),
                        ),
                        "is_mxfp4_marlin": mxfp4,
                        "is_nvfp4_marlin": nvfp4,
                    }
                    self.assertEqual(
                        eval(expr, ns),
                        (dtype == "half" or sm >= 9) and not (mxfp4 or nvfp4),
                    )

    def test_dense_fp32_disables_atomic_without_changing_opt_out(self):
        expr = compile(
            ast.Expression(assignment_expression(DENSE, "use_atomic_add")),
            str(DENSE),
            "eval",
        )
        for fp32 in (False, True):
            for heuristic in (False, True):
                chooser = Mock(return_value=heuristic)
                self.assertEqual(
                    eval(
                        expr,
                        {
                            "use_fp32_reduce": fp32,
                            "should_use_atomic_add_reduce": chooser,
                            "reshaped_x": SimpleNamespace(size=lambda _: 6),
                            "padded_size_n": 1856,
                            "padded_size_k": 2688,
                            "input": SimpleNamespace(device="cuda", dtype="bfloat16"),
                        },
                    ),
                    not fp32 and heuristic,
                )
                self.assertEqual(chooser.called, not fp32)

    def test_only_breakable_bypasses_gqa_gate_and_capture_uses_gate(self):
        ns = {}
        load_functions(SRT / "model_executor/cuda_graph_config.py", ["Backend"], ns)
        load_functions(
            GRAPH,
            [
                "has_standard_gqa_for_all_local_layers",
                "should_disable_prefill_graph_for_nonstandard_gqa",
            ],
            ns,
        )
        for backend in ns["Backend"].ALL:
            for start, end in ((0, 46), (23, 46), (23, 23)):
                for count in (0, end - start, end - start + 1):
                    self.assertEqual(
                        ns["should_disable_prefill_graph_for_nonstandard_gqa"](
                            prefill_backend=backend,
                            attention_layer_count=count,
                            start_layer=start,
                            end_layer=end,
                        ),
                        backend != ns["Backend"].BREAKABLE and count < end - start,
                    )
        capture = next(
            n
            for n in tree(GRAPH).body
            if isinstance(n, ast.FunctionDef) and n.name == "capture_prefill_graph"
        )
        guards = [
            n
            for n in ast.walk(capture)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Call)
            and isinstance(n.test.func, ast.Name)
            and n.test.func.id == "should_disable_prefill_graph_for_nonstandard_gqa"
        ]
        self.assertEqual(len(guards), 1)
        kwargs = {k.arg: k.value for k in guards[0].test.keywords}
        self.assertEqual(kwargs["prefill_backend"].id, "prefill_backend")
        self.assertTrue(any(isinstance(n, ast.Return) for n in guards[0].body))


if __name__ == "__main__":
    unittest.main()
