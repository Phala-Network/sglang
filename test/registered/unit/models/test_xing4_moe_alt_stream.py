"""CPU checks of actual Xing stream construction and upstream MoE dispatch.

Only storage and GPU execution are substituted. These checks do not qualify
CUDA graph edges, numerical accuracy, response content, or throughput.
"""

from __future__ import annotations

import ast
import copy
import itertools
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / "python/sglang/srt/models/xing4_0.py"
DEEPSEEK = ROOT / "python/sglang/srt/models/deepseek_v2.py"


def method(source, class_name, method_name):
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name))


def actual_class(name, methods, namespace):
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            ast.ClassDef(name=name, bases=[ast.Attribute(value=ast.Name(id="nn", ctx=ast.Load()), attr="Module", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[]),
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<actual-source-methods>", "exec"), namespace)
    return namespace[name]


class StorageOnly:
    pass


class TestXingModelOwnedMoeStream(unittest.TestCase):
    def test_cpu_and_model_owned_stream_lifetime(self):
        source = MODEL.read_text()
        events = []

        def stream():
            token = object()
            events.append(token)
            return token

        def make_layers(count, factory, **kwargs):
            return [factory(i, f"{kwargs['prefix']}.{i}") for i in range(count)], 0, count

        namespace = dict(
            nn=NS(Module=StorageOnly), torch=NS(cuda=NS(Stream=stream)),
            get_pp_group=lambda: NS(is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1),
            VocabParallelEmbedding=lambda *args, **kwargs: object(), PPMissingLayer=object,
            make_layers=make_layers, Xing4_0DecoderLayer=lambda *args, **kwargs: NS(**kwargs),
            add_prefix=lambda name, prefix: f"{prefix}.{name}" if prefix else name,
            RMSNorm=lambda *args, **kwargs: object(), _is_cuda=False,
        )
        cls = actual_class("Xing4_0Model", [method(source, "Xing4_0Model", "__init__")], namespace)
        config = NS(vocab_size=128, hidden_size=3584, num_hidden_layers=40, rms_norm_eps=1e-6, hc_mult=4)
        cpu = cls(config)
        self.assertIsNone(cpu.moe_alt_stream)
        self.assertEqual(len(events), 0)
        namespace["_is_cuda"] = True
        first, second = cls(config), cls(config)
        self.assertEqual(len(events), 2)
        self.assertIsNot(first.moe_alt_stream, second.moe_alt_stream)
        self.assertTrue(all(layer.moe_alt_stream is first.moe_alt_stream for layer in first.layers))
        self.assertTrue(all(layer.moe_alt_stream is second.moe_alt_stream for layer in second.layers))
        self.assertEqual(source.count("torch.cuda.Stream()"), 1)

    def test_attention_stream_and_legacy_moe_fallback(self):
        constructor = method(MODEL.read_text(), "Xing4_0DecoderLayer", "__init__")

        def named_call(name):
            return next(node for node in ast.walk(constructor) if isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id == name) or (isinstance(node.func, ast.Attribute) and node.func.attr == name)))

        attention = next(arg.value for arg in named_call("DeepseekV2AttentionMLA").keywords if arg.arg == "alt_stream")
        moe = next(arg.value for arg in named_call("DeepseekV2MoE").keywords if arg.arg == "alt_stream")
        legacy, dedicated = object(), object()
        for old, new in itertools.product((None, legacy), (None, dedicated)):
            namespace = {"alt_stream": old, "moe_alt_stream": new}
            self.assertIs(eval(compile(ast.Expression(attention), "<attention-arg>", "eval"), namespace), old)
            self.assertIs(eval(compile(ast.Expression(moe), "<moe-arg>", "eval"), namespace), new if new is not None else old)

    def test_existing_upstream_dispatch_boundaries(self):
        fake_mega = ModuleType("sglang.srt.layers.moe.mega_moe")
        fake_mega.should_use_mega_moe = lambda *args: False
        fake_mega.forward_mega_moe = lambda *args, **kwargs: "mega"
        state = dict(capture=False, bcg=False)
        namespace = dict(
            nn=NS(Module=StorageOnly), _enable_pcg_dsv2_dual_stream=False,
            _is_hip=False, get_is_capture_mode=lambda: state["capture"],
            is_in_breakable_cuda_graph=lambda: state["bcg"],
        )
        source = DEEPSEEK.read_text()
        cls = actual_class("DispatchProbe", [method(source, "DeepseekV2MoE", "_can_dual_stream_graph"), method(source, "DeepseekV2MoE", "forward")], namespace)
        probe = cls()
        probe.gate = NS(e_score_correction_bias_vl=None)
        probe._enable_a2a_moe = False
        probe.forward_normal_dual_stream = lambda *args, **kwargs: "dual"
        probe.forward_normal = lambda *args, **kwargs: "serial"
        probe.forward_deepep = lambda *args, **kwargs: "a2a"
        cases = itertools.product((0, 1, 2, 4, 6, 8, 16, 32, 256), (0, 1), (False, True), (False, True), (False, True))
        with patch.dict(sys.modules, {fake_mega.__name__: fake_mega}):
            for rows, fused, capture, bcg, has_stream in cases:
                with self.subTest(rows=rows, fused=fused, capture=capture, bcg=bcg, has_stream=has_stream):
                    probe.alt_stream = object() if has_stream else None
                    probe.num_fused_shared_experts = fused
                    state.update(capture=capture, bcg=bcg)
                    selected = probe.forward(NS(shape=(rows, 3584)))
                    expected = "dual" if rows > 0 and fused == 0 and capture and not bcg and has_stream else "serial"
                    self.assertEqual(selected, expected)


if __name__ == "__main__":
    unittest.main()
