"""Real CPU tensor/reader tests with source-method loader/model integration.

The model constructors/distributed/native imports are not loaded. The integration
fixture executes the actual GGUF loader, weight iterator, Qwen body/head loaders
and identity audit on a small CPU parameter graph. It is not full-model/GPU
acceptance. Historical donors: 81881bd7c1, 39eedc17eb and 788a898375.
"""

import ast
import glob
import importlib.util
import logging
import os
import re
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gguf
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, SRT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NAMES_MODULE = "sglang.srt.model_loader.gguf_name_maps"
VISION_MODULE = "sglang.srt.model_loader.gguf_vision"
NAMES = load_file(NAMES_MODULE, "model_loader/gguf_name_maps.py")
VISION = load_file(VISION_MODULE, "model_loader/gguf_vision.py")


def source_nodes(relative, names, class_name=None):
    tree = ast.parse((SRT / relative).read_text(encoding="utf-8"))
    if class_name:
        tree = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"Missing real source methods: {relative}: {names}")
    return nodes


def execute(nodes, namespace):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            "real_sglang_source_methods",
            "exec",
        ),
        namespace,
    )
    return namespace


def config(model_type="qwen3_5_text"):
    return SimpleNamespace(
        model_type=model_type,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        tie_word_embeddings=False,
        language_model_only=False,
    )


def base_namespace():
    namespace = dict(
        torch=torch,
        nn=torch.nn,
        glob=glob,
        os=os,
        logger=logging.getLogger(__name__),
        _MODEL_PREFIX="model.",
        _is_cpu=False,
        _is_amx_available=False,
        QWEN3_5_KV_SCALE_MAPPER=SimpleNamespace(apply=lambda weights: weights),
        get_layer_id=lambda name: (
            int(re.search(r"layers\.(\d+)", name)[1])
            if re.search(r"layers\.(\d+)", name)
            else None
        ),
    )
    return execute(
        source_nodes(
            "model_loader/weight_utils.py",
            {
                "gguf_quant_weights_iterator",
                "default_weight_loader",
            },
        ),
        namespace,
    )


class TestQwenGGUFTransform(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.transform = NAMES.Qwen3_5GGUFWeightTransform(self.cfg)
        self.order = [0, 2, 4, 1, 3, 5]

    def test_state_conversion_and_head_order(self):
        expected = torch.arange(6, dtype=torch.float32) / 4
        stored = expected.reshape(2, 3).T.reshape(-1)
        torch.testing.assert_close(
            self.transform("model.layers.0.linear_attn.A_log", -stored.exp(), None),
            expected,
        )
        torch.testing.assert_close(
            self.transform("model.layers.0.linear_attn.dt_bias", stored, None), expected
        )

    def test_folded_norm_excludes_raw_gated_norm(self):
        weight = torch.tensor([1.25, 0.75])
        for name in (
            "model.norm.weight",
            "model.layers.3.self_attn.q_norm.weight",
            "model.layers.3.self_attn.k_norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ):
            torch.testing.assert_close(self.transform(name, weight, None), weight - 1)
        self.assertIs(
            self.transform("model.layers.0.linear_attn.norm.weight", weight, None),
            weight,
        )

    def test_qkv_conv_and_other_gdn_row_permutations(self):
        weight = torch.arange(320 * 4).reshape(320, 4)
        expected = torch.cat(
            (weight[:128], weight[128:].reshape(6, 32, 4)[self.order].reshape(192, 4))
        )
        torch.testing.assert_close(
            self.transform("model.layers.0.linear_attn.in_proj_qkv.qweight", weight, 8),
            expected,
        )
        torch.testing.assert_close(
            self.transform("model.layers.0.linear_attn.conv1d.weight", weight, None),
            expected.unsqueeze(1),
        )
        for suffix, rows, block in (
            ("in_proj_z", 192, 32),
            ("in_proj_b", 6, 1),
            ("in_proj_a", 6, 1),
        ):
            data = torch.arange(rows * 4).reshape(rows, 4)
            expected = data.reshape(6, block, 4)[self.order].reshape(rows, 4)
            torch.testing.assert_close(
                self.transform(f"model.layers.0.linear_attn.{suffix}.qweight", data, 8),
                expected,
            )

    def test_q8_output_columns_are_exact_bytes(self):
        weight = torch.arange(2 * 6 * 34).remainder(256).to(torch.uint8).reshape(2, 204)
        expected = weight.reshape(2, 6, 34)[:, self.order].reshape(2, 204)
        torch.testing.assert_close(
            self.transform("model.layers.0.linear_attn.out_proj.qweight", weight, 8),
            expected,
        )
        real = torch.arange(2 * 192).reshape(2, 192)
        expected = real.reshape(2, 6, 32)[:, self.order].reshape(2, 192)
        torch.testing.assert_close(
            self.transform("model.layers.0.linear_attn.out_proj.weight", real, None),
            expected,
        )

    def test_bf16_payload_and_type_iterator(self):
        weight = torch.tensor([[1.5, -2, 0.25]], dtype=torch.bfloat16)
        values = [
            ("model.embed_tokens.qweight_type", torch.tensor(30)),
            ("model.embed_tokens.qweight", weight.view(torch.uint8)),
        ]
        actual = list(NAMES.apply_gguf_weight_transform(iter(values), self.transform))
        torch.testing.assert_close(actual[1][1], weight, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "scalar"):
            list(
                NAMES.apply_gguf_weight_transform(
                    [("x.qweight_type", torch.tensor([8, 8]))], self.transform
                )
            )
        with self.assertRaisesRegex(ValueError, "missing GGUF weight type"):
            list(NAMES.apply_gguf_weight_transform([values[1]], self.transform))

    def test_invalid_decay_geometry_and_quant_block_fail_closed(self):
        for invalid in (
            torch.zeros(6),
            torch.full((6,), -float("inf")),
            torch.full((6,), float("nan")),
        ):
            with self.assertRaises(ValueError):
                self.transform("model.layers.0.linear_attn.A_log", invalid, None)
        for key, value in (("linear_num_key_heads", 0), ("linear_num_value_heads", 5)):
            cfg = config()
            setattr(cfg, key, value)
            with self.assertRaises(ValueError):
                NAMES.Qwen3_5GGUFWeightTransform(cfg)
        with self.assertRaises(NotImplementedError):
            self.transform(
                "model.layers.0.linear_attn.out_proj.qweight",
                torch.zeros(2, 210, dtype=torch.uint8),
                14,
            )
        with self.assertRaises(ValueError):
            self.transform(
                "model.embed_tokens.qweight", torch.zeros(1, 3, dtype=torch.uint8), 30
            )

    def test_other_architectures_are_not_transformed(self):
        for name in ("gemma4", "qwen2", "qwen3_next", "qwen3_5_moe", "muse_glimmer"):
            self.assertIsNone(
                NAMES.get_gguf_weight_transform(SimpleNamespace(model_type=name))
            )
        self.assertIs(
            NAMES.GGUF_HF_NAME_MAP_BUILDERS["muse_glimmer"],
            NAMES.build_muse_glimmer_name_map,
        )

    def test_completeness_uses_identity_not_equal_values(self):
        model = torch.nn.Module()
        model.register_parameter("first", torch.nn.Parameter(torch.ones(2)))
        model.register_parameter("alias", model.first)
        model.register_parameter(
            "equal_but_distinct", torch.nn.Parameter(torch.ones(2))
        )
        self.assertEqual(
            NAMES.get_missing_gguf_parameters(model, {"alias"}), ["equal_but_distinct"]
        )
        self.assertEqual(
            NAMES.get_missing_gguf_parameters(model, {"alias", "equal_but_distinct"}),
            [],
        )
        self.assertEqual(
            NAMES.get_missing_gguf_parameters(model, {"unknown"}),
            ["equal_but_distinct", "first"],
        )

    def test_real_transformers_meta_name_map(self):
        from transformers import Qwen3_5TextConfig

        cfg = Qwen3_5TextConfig(
            num_hidden_layers=2,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            vocab_size=32,
            full_attention_interval=2,
        )
        mapping = NAMES.build_qwen3_5_name_map(cfg)
        self.assertEqual(mapping["blk.0.ssm_a"], "model.layers.0.linear_attn.A_log")
        self.assertEqual(
            mapping["blk.0.ssm_dt.bias"], "model.layers.0.linear_attn.dt_bias"
        )
        self.assertEqual(mapping["output.weight"], "lm_head.weight")
        self.assertEqual(
            mapping["blk.1.attn_q.weight"], "model.layers.1.self_attn.q_proj.weight"
        )


class TestQwenGGUFVision(unittest.TestCase):
    def setUp(self):
        self.vision = SimpleNamespace(
            temporal_patch_size=2,
            deepstack_visual_indexes=[],
            hidden_size=4,
            intermediate_size=8,
            spatial_merge_size=2,
            in_channels=3,
            patch_size=2,
            num_position_embeddings=16,
            out_hidden_size=12,
            depth=1,
            num_heads=2,
        )
        self.config = SimpleNamespace(vision_config=self.vision)
        self.mapping = VISION.qwen35_vision_tensor_map(self.config)
        self.metadata = {
            "general.architecture": "clip",
            "clip.projector_type": "qwen3vl_merger",
            "clip.vision.block_count": 1,
            "clip.vision.embedding_length": 4,
            "clip.vision.attention.head_count": 2,
            "clip.vision.feed_forward_length": 8,
            "clip.vision.patch_size": 2,
            "clip.vision.spatial_merge_size": 2,
            "clip.vision.projection_dim": 12,
        }
        self.tensors = [
            self.tensor(
                name,
                torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)
                / 64,
            )
            for name, (_, shape) in self.mapping.items()
        ]
        self.reader = SimpleNamespace(
            tensors=self.tensors,
            get_field=lambda key: (
                SimpleNamespace(contents=lambda: self.metadata[key])
                if key in self.metadata
                else None
            ),
        )

    @staticmethod
    def tensor(name, value):
        bf16 = value.dtype == torch.bfloat16
        return SimpleNamespace(
            name=name,
            shape=tuple(reversed(value.shape)),
            tensor_type=gguf.GGMLQuantizationType.BF16
            if bf16
            else gguf.GGMLQuantizationType.F32,
            data=value.view(torch.uint8).numpy() if bf16 else value.numpy(),
        )

    def load(self):
        with patch.object(gguf, "GGUFReader", return_value=self.reader):
            return dict(
                VISION.qwen35_vision_weights_iterator(
                    "fixture-mmproj.gguf", self.config
                )
            )

    def test_temporal_conv3d_matches_exact_converter_slices(self):
        torch.manual_seed(63)
        weight = torch.randn(4, 3, 2, 2, 2)
        self.tensors[0] = self.tensor(
            "v.patch_embd.weight", weight[:, :, 0].contiguous()
        )
        self.tensors[1] = self.tensor(
            "v.patch_embd.weight.1", weight[:, :, 1].contiguous()
        )
        loaded = self.load()
        self.assertEqual(len(loaded), 21)
        actual_weight = loaded["visual.patch_embed.proj.weight"]
        torch.testing.assert_close(actual_weight, weight, rtol=0, atol=0)
        inputs = torch.randn(2, 3, 2, 4, 4)
        expected = F.conv2d(inputs[:, :, 0], weight[:, :, 0], stride=2) + F.conv2d(
            inputs[:, :, 1], weight[:, :, 1], stride=2
        )
        torch.testing.assert_close(
            F.conv3d(inputs, actual_weight, stride=(2, 2, 2)).squeeze(2), expected
        )

    def test_bf16_exact_bytes(self):
        weight = torch.tensor(
            [[1.5, -2, 0.03125], [3.5, -0.5, 64]], dtype=torch.bfloat16
        )
        torch.testing.assert_close(
            VISION._read_unquantized_tensor(self.tensor("bf16", weight), (2, 3)),
            weight,
            rtol=0,
            atol=0,
        )

    def test_missing_extra_duplicate_and_bad_shape_fail(self):
        missing = self.tensors.pop()
        with self.assertRaisesRegex(ValueError, "missing="):
            self.load()
        self.tensors.append(missing)
        self.tensors.append(self.tensor("unexpected", torch.ones(2)))
        with self.assertRaisesRegex(ValueError, "extra="):
            self.load()
        self.tensors.pop()
        self.tensors.append(self.tensors[-1])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.load()
        self.tensors.pop()
        self.tensors[-1].shape = (123,)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.load()

    def test_metadata_geometry_and_quantized_vision_fail(self):
        self.metadata["clip.projector_type"] = "qwen2vl_merger"
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            self.load()
        for key, value in (
            ("deepstack_visual_indexes", [0]),
            ("temporal_patch_size", 3),
        ):
            old = getattr(self.vision, key)
            setattr(self.vision, key, value)
            with self.assertRaisesRegex(ValueError, "no deepstack"):
                VISION.qwen35_vision_tensor_map(self.config)
            setattr(self.vision, key, old)
        value = self.tensor("quantized", torch.ones(2, 32))
        value.tensor_type = gguf.GGMLQuantizationType.Q8_0
        with self.assertRaisesRegex(ValueError, "only F32/F16/BF16"):
            VISION._read_unquantized_tensor(value, (2, 32))


def load_loader_namespace(model=None):
    namespace = base_namespace()
    namespace.update(
        BaseModelLoader=type("BaseModelLoader", (), {}),
        _initialize_model=lambda *args: model,
        _get_quantization_config=lambda *args: SimpleNamespace(get_name=lambda: "gguf"),
        set_default_torch_dtype=lambda *args: nullcontext(),
        device_loading_context=lambda *args: nullcontext(),
        get_gguf_extra_tensor_names=lambda *args: [],
    )
    return execute(
        source_nodes("model_loader/loader.py", {"GGUFModelLoader"}), namespace
    )


class TestQwenGGUFLoadChain(unittest.TestCase):
    def setUp(self):
        # Restore only our two injected modules. patch.dict(sys.modules) would
        # also remove lazy Torch imports, leaving C++ dispatcher registrations
        # alive and causing duplicate registration on the next import.
        for name, module in ((NAMES_MODULE, NAMES), (VISION_MODULE, VISION)):
            previous = sys.modules.get(name)
            sys.modules[name] = module
            if previous is None:
                self.addCleanup(sys.modules.pop, name, None)
            else:
                self.addCleanup(sys.modules.__setitem__, name, previous)

    def test_snapshot_selection_rejects_ambiguity(self):
        Loader = load_loader_namespace()["GGUFModelLoader"]
        loader = object.__new__(Loader)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                loader._prepare_weights(directory)
            target = Path(directory) / "model.gguf"
            target.touch()
            self.assertEqual(loader._prepare_weights(directory), str(target))
            (Path(directory) / "mmproj-bf16.gguf").touch()
            self.assertEqual(loader._prepare_weights(directory, True), str(target))
            with self.assertRaises(ValueError):
                loader._prepare_weights(directory)
            (Path(directory) / "draft.gguf").touch()
            with self.assertRaises(ValueError):
                loader._prepare_weights(directory, True)

    def test_actual_iterator_reader_and_text_body_head_loaders(self):
        namespace = base_namespace()
        execute(
            source_nodes("models/qwen3_5.py", {"load_weights"}, "Qwen3_5ForCausalLM"),
            namespace,
        )
        Body = type(
            "Body", (torch.nn.Module,), {"load_weights": namespace["load_weights"]}
        )
        execute(
            source_nodes(
                "models/qwen3_5_text.py", {"load_weights"}, "Qwen3_5ForCausalLM"
            ),
            namespace,
        )
        Text = type(
            "Text", (torch.nn.Module,), {"load_weights": namespace["load_weights"]}
        )
        model = Text()
        model.config = config()
        model.quant_config = SimpleNamespace(get_name=lambda: "gguf")
        model.pp_group = SimpleNamespace(is_last_rank=True)
        model.model = Body()
        model.model.norm = torch.nn.Module()
        model.model.norm.weight = torch.nn.Parameter(
            torch.zeros(2), requires_grad=False
        )
        model.lm_head = torch.nn.Module()
        model.lm_head.qweight = torch.nn.Parameter(
            torch.zeros(2, 2, dtype=torch.bfloat16), requires_grad=False
        )
        model.lm_head.qweight_type = torch.nn.Parameter(
            torch.zeros((), dtype=torch.int64), requires_grad=False
        )
        namespace = load_loader_namespace(model)
        Loader = namespace["GGUFModelLoader"]
        loader = object.__new__(Loader)
        loader.load_config = SimpleNamespace()
        loader._get_gguf_weights_map = lambda _: {
            "output_norm.weight": "model.norm.weight",
            "output.weight": "lm_head.weight",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.gguf")
            writer = gguf.GGUFWriter(path, "qwen35")
            norm = torch.tensor([1.25, 0.75])
            head = torch.tensor([[1.5, -2], [0.25, 8]], dtype=torch.bfloat16)
            writer.add_tensor("output_norm.weight", norm.numpy())
            writer.add_tensor(
                "output.weight",
                head.view(torch.uint8).numpy(),
                raw_dtype=gguf.GGMLQuantizationType.BF16,
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            cfg = SimpleNamespace(
                hf_config=config(), model_path=path, dtype=torch.bfloat16
            )
            loaded = loader.load_model(
                model_config=cfg, device_config=SimpleNamespace(device="cpu")
            )
            self.assertIs(loaded, model)
            torch.testing.assert_close(model.model.norm.weight, norm - 1)
            torch.testing.assert_close(model.lm_head.qweight, head, rtol=0, atol=0)
            self.assertEqual(model.lm_head.qweight_type.item(), 30)
            model.register_parameter("unloaded", torch.nn.Parameter(torch.ones(2)))
            with self.assertRaisesRegex(RuntimeError, "unloaded"):
                loader.load_model(
                    model_config=cfg, device_config=SimpleNamespace(device="cpu")
                )
            namespace["_get_quantization_config"] = lambda *args: SimpleNamespace(
                get_name=lambda: "fp8"
            )
            with self.assertRaisesRegex(ValueError, "requires GGUF quantization"):
                loader.load_model(
                    model_config=cfg, device_config=SimpleNamespace(device="cpu")
                )

    def test_architecture_mismatch_and_other_architecture_passthrough(self):
        namespace = load_loader_namespace()
        sentinel = iter([("sentinel", torch.ones(1))])
        namespace["gguf_quant_weights_iterator"] = lambda *args: sentinel
        loader = object.__new__(namespace["GGUFModelLoader"])
        cfg = SimpleNamespace(hf_config=config())
        reader = SimpleNamespace(
            get_field=lambda _: SimpleNamespace(contents=lambda: "qwen3next")
        )
        with patch.object(gguf, "GGUFReader", return_value=reader):
            with self.assertRaisesRegex(ValueError, "architecture=qwen35"):
                loader._get_weights_iterator("fixture.gguf", {}, cfg)
        for architecture in ("gemma4", "qwen2", "muse_glimmer"):
            cfg.hf_config = SimpleNamespace(model_type=architecture)
            self.assertIs(
                loader._get_weights_iterator("fixture.gguf", {}, cfg), sentinel
            )

    def test_actual_multimodal_loader_chains_visual_and_text_weights(self):
        vision_fixture = TestQwenGGUFVision()
        vision_fixture.setUp()
        cfg = config("qwen3_5")
        cfg.text_config = config()
        cfg.vision_config = vision_fixture.vision
        namespace = base_namespace()
        execute(
            source_nodes(
                "models/qwen3_5.py", {"load_weights"}, "Qwen3_5ForConditionalGeneration"
            ),
            namespace,
        )
        Model = type(
            "VisionLanguage",
            (torch.nn.Module,),
            {"load_weights": namespace["load_weights"]},
        )
        model = Model()
        model.config = cfg.text_config
        model.pp_group = SimpleNamespace(is_last_rank=True)

        def add_parameter(path, shape):
            parent = model
            parts = path.split(".")
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    parent.add_module(part, torch.nn.Module())
                parent = getattr(parent, part)
            parent.register_parameter(
                parts[-1], torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
            )

        add_parameter("model.norm.weight", (2,))
        for target, shape in vision_fixture.mapping.values():
            if target == "visual.patch_embed.proj.weight":
                shape = (shape[0], shape[1], 2, *shape[2:])
            if target not in dict(model.named_parameters()):
                add_parameter(target, shape)
        namespace = load_loader_namespace(model)
        loader = object.__new__(namespace["GGUFModelLoader"])
        loader.load_config = SimpleNamespace()
        loader._get_gguf_weights_map = lambda _: {
            "output_norm.weight": "model.norm.weight"
        }
        main_reader = SimpleNamespace(
            get_field=lambda _: SimpleNamespace(contents=lambda: "qwen35"),
            tensors=[
                vision_fixture.tensor("output_norm.weight", torch.tensor([1.25, 0.75]))
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "model.gguf"
            model_file.touch()
            model_cfg = SimpleNamespace(
                hf_config=cfg, model_path=str(model_file), dtype=torch.float32
            )
            with self.assertRaisesRegex(ValueError, "exactly one mmproj"):
                loader.load_model(
                    model_config=model_cfg, device_config=SimpleNamespace(device="cpu")
                )
            projector = Path(directory) / "mmproj-bf16.gguf"
            projector.touch()
            with patch.object(
                gguf,
                "GGUFReader",
                side_effect=lambda path: (
                    vision_fixture.reader if "mmproj-" in str(path) else main_reader
                ),
            ):
                loaded = loader.load_model(
                    model_config=model_cfg, device_config=SimpleNamespace(device="cpu")
                )
            self.assertIs(loaded, model)
            expected = vision_fixture.load()
            for name, value in expected.items():
                torch.testing.assert_close(
                    dict(model.named_parameters())[name], value, rtol=0, atol=0
                )
            torch.testing.assert_close(
                model.model.norm.weight, torch.tensor([0.25, -0.25])
            )
            (Path(directory) / "mmproj-duplicate.gguf").touch()
            with self.assertRaisesRegex(ValueError, "exactly one mmproj"):
                loader.load_model(
                    model_config=model_cfg, device_config=SimpleNamespace(device="cpu")
                )

    def test_non_gguf_head_does_not_consume_quantized_names(self):
        namespace = base_namespace()
        execute(
            source_nodes(
                "models/qwen3_5_text.py", {"load_weights"}, "Qwen3_5ForCausalLM"
            ),
            namespace,
        )
        param = torch.nn.Parameter(torch.zeros(1))
        model = SimpleNamespace(
            config=SimpleNamespace(tie_word_embeddings=False),
            quant_config=SimpleNamespace(get_name=lambda: "fp8"),
            pp_group=SimpleNamespace(is_last_rank=True),
            named_parameters=lambda: iter([("lm_head.qweight", param)]),
            model=SimpleNamespace(
                load_weights=lambda weights: set(name for name, _ in weights)
            ),
        )
        loaded = namespace["load_weights"](
            model, iter([("lm_head.qweight", torch.ones(1))])
        )
        self.assertEqual(loaded, set())
        self.assertEqual(param.item(), 0)

    def test_embedding_hook_only_receives_gguf_for_supported_qwen(self):
        namespace = base_namespace()
        namespace.update(
            VocabParallelEmbedding=lambda *args, **kwargs: kwargs,
            PPMissingLayer=lambda: "missing",
            is_dp_attention_enabled=lambda: False,
            add_prefix=lambda suffix, prefix: (
                f"{prefix}.{suffix}" if prefix else suffix
            ),
        )
        execute(
            source_nodes(
                "models/qwen3_5.py", {"_build_embed_tokens"}, "Qwen3_5ForCausalLM"
            ),
            namespace,
        )
        build = namespace["_build_embed_tokens"]
        init = source_nodes("models/qwen3_5.py", {"__init__"}, "Qwen3_5ForCausalLM")[0]
        selection = next(
            node
            for node in init.body
            if isinstance(node, ast.If)
            and "config.model_type" in ast.unparse(node.test)
        )
        for architecture, quant, expected in (
            ("qwen3_5_text", "gguf", True),
            ("qwen3_5", "gguf", True),
            ("qwen3_5_text", "fp8", False),
            ("gemma4", "gguf", False),
            ("qwen4_exp", "gguf", False),
            ("qwen2", "fp8", False),
        ):
            with self.subTest(architecture=architecture, quant=quant):
                subject = SimpleNamespace(pp_group=SimpleNamespace(is_first_rank=True))
                subject._build_embed_tokens = lambda *args: build(subject, *args)
                cfg = SimpleNamespace(
                    model_type=architecture, vocab_size=4, hidden_size=2
                )
                qcfg = SimpleNamespace(get_name=lambda: quant)
                execute(
                    [selection],
                    {
                        **namespace,
                        "self": subject,
                        "config": cfg,
                        "quant_config": qcfg,
                        "prefix": "model",
                    },
                )
                self.assertEqual("quant_config" in subject.embed_tokens, expected)
                if expected:
                    self.assertIs(subject.embed_tokens["quant_config"], qcfg)
                    self.assertEqual(
                        subject.embed_tokens["prefix"], "model.embed_tokens"
                    )
        # A subclass's old one-argument hook must still work.
        subject = SimpleNamespace(_build_embed_tokens=lambda cfg: "subclass")
        execute(
            [selection],
            {
                **namespace,
                "self": subject,
                "config": SimpleNamespace(model_type="qwen4_exp"),
                "quant_config": None,
                "prefix": "",
            },
        )
        self.assertEqual(subject.embed_tokens, "subclass")

    def test_packed_gguf_type_binding_and_scalar_slot(self):
        namespace = base_namespace()
        namespace.update(
            BlockQuantScaleParameter=type("Block", (), {}),
            PerTensorScaleParameter=type("Scale", (), {}),
            set_weight_attrs=lambda p, attrs: [
                setattr(p, key, value) for key, value in attrs.items()
            ],
        )
        nodes = source_nodes(
            "models/qwen3_5.py",
            {
                "_override_weight_loader",
                "_bind_packed_weight_loaders",
                "_get_split_sizes_for_param",
                "_make_packed_weight_loader",
            },
            "Qwen3_5GatedDeltaNet",
        )
        cls = ast.ClassDef(
            name="GDN", bases=[], keywords=[], body=nodes, decorator_list=[]
        )
        execute([cls], namespace)
        execute(
            source_nodes(
                "layers/linear.py", {"weight_loader"}, "MergedColumnParallelLinear"
            ),
            namespace,
        )
        module = SimpleNamespace(output_sizes=[4, 4, 12, 12])
        param = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
        param.is_gguf_weight_type = True
        param.shard_weight_type = {}
        param.weight_loader = lambda p, value, shard: namespace["weight_loader"](
            module, p, value, shard
        )
        module.qweight_type = param
        namespace["GDN"]()._bind_packed_weight_loaders(module)
        param.weight_loader(param, torch.tensor(8), (0, 1, 2))
        torch.testing.assert_close(param.data, torch.tensor([8.0, 8.0, 8.0, 0.0]))
        self.assertEqual(param.shard_weight_type, {0: 8, 1: 8, 2: 8})
        packed = torch.nn.Parameter(
            torch.empty(0, dtype=torch.uint8), requires_grad=False
        )
        packed.is_gguf_weight = True
        packed.output_dim = 0
        packed.shard_id, packed.shard_id_map, packed.data_container = [], {}, []
        module.tp_size, module.tp_rank = 1, 0
        packed.weight_loader = lambda p, value, shard: namespace["weight_loader"](
            module, p, value, shard
        )
        namespace["GDN"]()._bind_packed_weight_loaders(
            SimpleNamespace(qweight=packed, output_sizes=module.output_sizes)
        )
        data = torch.arange(40, dtype=torch.uint8).reshape(20, 2)
        packed.weight_loader(packed, data, (0, 1, 2))
        self.assertEqual(packed.shard_id, [0, 1, 2])
        self.assertEqual(
            [tuple(value.shape) for value in packed.data_container],
            [(4, 2), (4, 2), (12, 2)],
        )
        torch.testing.assert_close(
            torch.cat(packed.data_container), data, rtol=0, atol=0
        )
        non_gguf = torch.nn.Parameter(torch.zeros(1))
        original = lambda *args: None
        non_gguf.weight_loader = original
        namespace["GDN"]()._bind_packed_weight_loaders(
            SimpleNamespace(qweight=non_gguf)
        )
        self.assertIs(non_gguf.weight_loader, original)


if __name__ == "__main__":
    unittest.main()
