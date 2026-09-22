"""Execute the actual weight-load prefix with CPU stubs; no GPU imports."""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py"
)


def load_weight_method(trace):
    parsed = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in parsed.body
        if isinstance(n, ast.ClassDef) and n.name == "Mxfp4FlashinferTrtllmMoEMethod"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "process_weights_after_loading"
    )
    namespace = {
        "Module": object,
        "_pad_intermediate_size": lambda layer: trace.append("pad"),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace[fn.name], fn


class ReorderReached(Exception):
    pass


class WeightLoadContract(unittest.TestCase):
    def test_actual_method_orders_loader_then_padding_then_reorder(self):
        trace = []
        method, _ = load_weight_method(trace)
        quant_utils = ModuleType("sglang.srt.layers.quantization.utils")

        def reorder(*args):
            trace.append("reorder")
            raise ReorderReached

        quant_utils.reorder_w1w3_to_w3w1 = reorder
        owner = SimpleNamespace(
            _fp8=SimpleNamespace(
                process_weights_after_loading=lambda layer: trace.append("fp8-load")
            )
        )
        layer = SimpleNamespace(
            w13_weight=SimpleNamespace(data=None),
            w13_weight_scale_inv=SimpleNamespace(data=None),
        )
        with patch.dict(sys.modules, {quant_utils.__name__: quant_utils}):
            with self.assertRaises(ReorderReached):
                method(owner, layer)
        self.assertEqual(trace, ["fp8-load", "pad", "reorder"])

    def test_mega_moe_bypass_never_pads_or_shuffles(self):
        trace = []
        method, _ = load_weight_method(trace)
        quant_utils = ModuleType("sglang.srt.layers.quantization.utils")
        quant_utils.reorder_w1w3_to_w3w1 = lambda *args: self.fail("unexpected reorder")
        owner = SimpleNamespace(
            _fp8=SimpleNamespace(
                process_weights_after_loading=lambda layer: trace.append("fp8-load")
            )
        )
        layer = SimpleNamespace(_mega_moe_weights_built=True)
        with patch.dict(sys.modules, {quant_utils.__name__: quant_utils}):
            method(owner, layer)
        self.assertEqual(trace, ["fp8-load"])

    def test_both_scale_tensors_keep_ue8m0_conversion_before_byte_shuffle(self):
        _, fn = load_weight_method([])
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        converted = {
            call.func.value.id
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "to"
            and isinstance(call.func.value, ast.Name)
            and len(call.args) == 1
            and ast.unparse(call.args[0]) == "torch.float8_e8m0fnu"
        }
        self.assertEqual(converted, {"w13_scale", "w2_scale"})
        converted_lines = [
            call.lineno
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "to"
            and call.args
            and ast.unparse(call.args[0]) == "torch.float8_e8m0fnu"
        ]
        byte_views = [
            call.lineno
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "view"
            and call.args
            and ast.unparse(call.args[0]) == "torch.uint8"
        ]
        self.assertLess(max(converted_lines), min(byte_views))


if __name__ == "__main__":
    unittest.main()
