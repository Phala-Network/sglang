"""CPU source-call dispatch tests, not CUDA kernel accuracy/performance evidence.

CUDA device metadata is simulated. Actual CPU torch multiplication verifies
operand routing only; ggml kernel calls are spies, never GPU executions.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType

from test_qwen_gguf_migration_cpu import execute, source_nodes


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/layers/quantization/gguf.py"
)


def load_dispatch():
    namespace = dict(torch=torch, gguf=gguf, WeightType=WeightType)
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {
        "UNQUANTIZED_TYPES",
        "STANDARD_QUANT_TYPES",
        "KQUANT_TYPES",
        "IMATRIX_QUANT_TYPES",
        "DEQUANT_TYPES",
        "MMVQ_QUANT_TYPES",
        "MMQ_QUANT_TYPES",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        )
    ]
    nodes += [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_use_bf16_gguf_prefill",
            "dequantize_gguf_weight",
            "fused_mul_mat_gguf",
            "_ordered_gguf_shard_ids",
        }
    ]
    execute(nodes, namespace)
    execute(
        source_nodes("layers/quantization/gguf.py", {"apply"}, "GGUFLinearMethod"),
        namespace,
    )
    return namespace


class CudaMetadataCPUActivation:
    """Only the device tag is simulated; matrix operands stay on the CPU."""

    def __init__(self, rows=128, dtype=torch.bfloat16, device="cuda", columns=32):
        self.data = torch.ones(rows, columns, dtype=dtype)
        self.shape, self.dtype = self.data.shape, self.data.dtype
        self.device = SimpleNamespace(type=device)

    def __matmul__(self, other):
        return self.data @ other


class TestQwenGGUFPrefillDispatch(unittest.TestCase):
    def setUp(self):
        self.ns = load_dispatch()
        self.guard = self.ns["_use_bf16_gguf_prefill"]
        self.dispatch = self.ns["fused_mul_mat_gguf"]
        self.packed = torch.zeros(3, 34, dtype=torch.uint8)
        self.dequant = self.ns["ggml_dequantize"] = Mock(
            return_value=torch.ones(3, 32, dtype=torch.bfloat16)
        )
        self.mmq = self.ns["ggml_mul_mat_a8"] = Mock(return_value="mmq")
        self.mmvq = self.ns["ggml_mul_mat_vec_a8"] = Mock(return_value="mmvq")

    def test_exact_row_dtype_device_and_tensor_type_guards(self):
        for rows, expected in (
            (0, False),
            (1, False),
            (20, False),
            (127, False),
            (128, True),
            (16384, True),
        ):
            with self.subTest(rows=rows):
                self.assertEqual(
                    self.guard(CudaMetadataCPUActivation(rows), WeightType.Q8_0),
                    expected,
                )
        for dtype, device, quant in (
            (torch.float16, "cuda", WeightType.Q8_0),
            (torch.float32, "cuda", WeightType.Q8_0),
            (torch.bfloat16, "cpu", WeightType.Q8_0),
            (torch.bfloat16, "musa", WeightType.Q8_0),
            (torch.bfloat16, "cuda", WeightType.Q4_K),
            (torch.bfloat16, "cuda", WeightType.Q8_1),
            (torch.bfloat16, "cuda", WeightType.IQ4_NL),
        ):
            with self.subTest(dtype=dtype, device=device, quant=quant):
                self.assertFalse(
                    self.guard(
                        CudaMetadataCPUActivation(dtype=dtype, device=device), quant
                    )
                )

    def test_qualified_path_calls_real_dequant_shape_and_matmul(self):
        activation = CudaMetadataCPUActivation()
        actual = self.dispatch(
            activation, self.packed, WeightType.Q8_0, qwen_bf16_q8_prefill=True
        )
        self.dequant.assert_called_once_with(
            self.packed, WeightType.Q8_0, 3, 32, torch.bfloat16
        )
        torch.testing.assert_close(
            actual, torch.full((128, 3), 32, dtype=torch.bfloat16)
        )
        self.mmq.assert_not_called()
        self.mmvq.assert_not_called()

    def test_default_does_not_enable_qwen_path_for_other_models(self):
        result = self.dispatch(
            CudaMetadataCPUActivation(), self.packed, WeightType.Q8_0
        )
        self.assertEqual(result, "mmq")
        self.dequant.assert_not_called()

    def test_decode_small_batch_and_nonmatching_paths_keep_original_kernels(self):
        for rows, dtype, device, quant, expected in (
            (1, torch.bfloat16, "cuda", WeightType.Q8_0, "mmvq"),
            (20, torch.bfloat16, "cuda", WeightType.Q8_0, "mmq"),
            (127, torch.bfloat16, "cuda", WeightType.Q8_0, "mmq"),
            (128, torch.float16, "cuda", WeightType.Q8_0, "mmq"),
            (128, torch.bfloat16, "cpu", WeightType.Q8_0, "mmq"),
            (128, torch.bfloat16, "cuda", WeightType.Q4_K, "mmq"),
            (128, torch.bfloat16, "cuda", WeightType.Q8_1, "mmq"),
        ):
            with self.subTest(rows=rows, dtype=dtype, device=device, quant=quant):
                self.assertEqual(
                    self.dispatch(
                        CudaMetadataCPUActivation(rows, dtype, device),
                        self.packed,
                        quant,
                        qwen_bf16_q8_prefill=True,
                    ),
                    expected,
                )
        self.dequant.assert_not_called()

    def test_empty_and_unquantized_short_circuit_remain_unchanged(self):
        empty = self.dispatch(
            torch.zeros(0, 32), self.packed, WeightType.Q8_0, qwen_bf16_q8_prefill=True
        )
        self.assertEqual(tuple(empty.shape), (0, 3))
        actual = self.dispatch(
            torch.ones(2, 3),
            torch.ones(4, 3),
            WeightType.BF16,
            qwen_bf16_q8_prefill=True,
        )
        torch.testing.assert_close(actual, torch.full((2, 4), 3.0))
        self.dequant.assert_not_called()
        self.mmq.assert_not_called()
        self.mmvq.assert_not_called()

    def test_imatrix_keeps_existing_dequant_fallback(self):
        # Existing I-matrix fallback still dequantizes; it is not the new path.
        actual = self.dispatch(
            CudaMetadataCPUActivation(),
            self.packed,
            WeightType.IQ4_NL,
            qwen_bf16_q8_prefill=True,
        )
        self.dequant.assert_called_once()
        self.assertEqual(self.dequant.call_args.args[1], WeightType.IQ4_NL)
        self.assertEqual(tuple(actual.shape), (128, 3))

    def test_linear_apply_propagates_scope_flag_without_changing_bias(self):
        self.packed.shard_id = []
        layer = SimpleNamespace(
            qweight=self.packed,
            qweight_type=SimpleNamespace(weight_type=WeightType.Q8_0),
        )
        method = SimpleNamespace(
            quant_config=SimpleNamespace(_qwen_bf16_q8_prefill=True)
        )
        result = self.ns["apply"](
            method,
            layer,
            CudaMetadataCPUActivation(),
            torch.ones(3, dtype=torch.bfloat16),
        )
        torch.testing.assert_close(
            result, torch.full((128, 3), 33, dtype=torch.bfloat16)
        )
        method.quant_config = SimpleNamespace()
        self.assertEqual(
            self.ns["apply"](method, layer, CudaMetadataCPUActivation()), "mmq"
        )

    def test_merged_shards_use_actual_per_shard_types(self):
        packed = torch.zeros(5, 34, dtype=torch.uint8)
        packed.shard_id = [1, 0]
        packed.shard_offset_map = {0: (0, 2, 34), 1: (2, 5, 34)}
        layer = SimpleNamespace(
            qweight=packed,
            qweight_type=SimpleNamespace(
                shard_weight_type={0: WeightType.Q8_0, 1: WeightType.Q4_K}
            ),
        )
        trace = []
        self.ns["fused_mul_mat_gguf"] = lambda x, w, t, **kw: (
            trace.append((tuple(w.shape), t, kw)) or torch.full((2, w.shape[0]), int(t))
        )
        result = self.ns["apply"](
            SimpleNamespace(quant_config=SimpleNamespace(_qwen_bf16_q8_prefill=True)),
            layer,
            torch.zeros(2, 32),
        )
        self.assertEqual(
            trace,
            [
                ((2, 34), WeightType.Q8_0, {"qwen_bf16_q8_prefill": True}),
                ((3, 34), WeightType.Q4_K, {"qwen_bf16_q8_prefill": True}),
            ],
        )
        self.assertEqual(tuple(result.shape), (2, 5))
        self.assertEqual(result[0].tolist(), [8, 8, 12, 12, 12])

    def test_loader_only_enables_supported_qwen_gguf(self):
        method = source_nodes(
            "model_loader/loader.py", {"load_model"}, "GGUFModelLoader"
        )[0]
        enable = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Assign)
                and "_qwen_bf16_q8_prefill" in ast.unparse(child)
                for child in node.body
            )
        )
        for architecture, expected in (
            ("qwen3_5", True),
            ("qwen3_5_text", True),
            ("gemma4", False),
            ("muse_glimmer", False),
            ("qwen2", False),
            ("qwen3_next", False),
        ):
            quant = SimpleNamespace(_qwen_bf16_q8_prefill=False)
            execute(
                [enable],
                {
                    "model_config": SimpleNamespace(
                        hf_config=SimpleNamespace(model_type=architecture)
                    ),
                    "quant_config": quant,
                },
            )
            self.assertEqual(quant._qwen_bf16_q8_prefill, expected)

    def test_quant_config_defaults_to_original_dispatch(self):
        nodes = source_nodes("layers/quantization/gguf.py", {"__init__"}, "GGUFConfig")
        cls = ast.ClassDef(
            name="GGUFConfig",
            bases=[ast.Name(id="QuantizationConfig", ctx=ast.Load())],
            keywords=[],
            body=nodes,
            decorator_list=[],
        )
        namespace = {
            "QuantizationConfig": type("QuantizationConfig", (), {}),
            "_is_hip": False,
        }
        execute([cls], namespace)
        self.assertFalse(namespace["GGUFConfig"]()._qwen_bf16_q8_prefill)

    def test_current_native_wrapper_call_signature(self):
        native_call = Mock(return_value="native-result")
        namespace = {
            "torch": SimpleNamespace(
                ops=SimpleNamespace(
                    sgl_kernel=SimpleNamespace(
                        ggml_dequantize=SimpleNamespace(default=native_call)
                    )
                )
            )
        }
        execute(
            source_nodes(
                "../kernels/aot/python/sgl_kernel/quantization/gguf.py",
                {"ggml_dequantize"},
            ),
            namespace,
        )
        actual = namespace["ggml_dequantize"](
            self.packed, WeightType.Q8_0, 3, 32, torch.bfloat16
        )
        self.assertEqual(actual, "native-result")
        native_call.assert_called_once_with(
            self.packed, WeightType.Q8_0, 3, 32, torch.bfloat16
        )


if __name__ == "__main__":
    unittest.main()
