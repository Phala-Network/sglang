"""CPU contracts for local Qwen3.5 GGUF selection and native config loading."""

import ast
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"


def source_function(relative, name, namespace):
    tree = ast.parse((SRT / relative).read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(relative), "exec"), namespace)
    return namespace[name]


class FakeConfig:
    model_type = "qwen3_5_text"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeMultiConfig(FakeConfig):
    model_type = "qwen3_5"

    def __init__(self, text_config, vision_config, **kwargs):
        super().__init__(
            text_config=FakeConfig(**text_config),
            vision_config=FakeConfig(**vision_config),
            **kwargs,
        )


class FakeReader:
    records = {}

    def __init__(self, path):
        meta, shapes = self.records[str(path)]
        self.fields = {
            key: SimpleNamespace(contents=lambda value=value: value)
            for key, value in meta.items()
        }
        self.tensors = [
            SimpleNamespace(name=name, shape=shape) for name, shape in shapes.items()
        ]


def text_metadata():
    return {
        "general.architecture": "qwen35",
        "qwen35.block_count": 4,
        "qwen35.full_attention_interval": 2,
        "qwen35.ssm.state_size": 32,
        "qwen35.ssm.inner_size": 192,
        "qwen35.ssm.group_count": 2,
        "qwen35.ssm.conv_kernel": 4,
        "qwen35.rope.dimension_count": 16,
        "qwen35.rope.freq_base": 1000000.0,
        "qwen35.attention.key_length": 64,
        "qwen35.attention.head_count": 8,
        "qwen35.attention.head_count_kv": 2,
        "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen35.embedding_length": 512,
        "qwen35.feed_forward_length": 1536,
        "qwen35.context_length": 8192,
        "tokenizer.ggml.eos_token_id": 42,
    }


def vision_metadata():
    return {
        "general.architecture": "clip",
        "clip.projector_type": "qwen3vl_merger",
        "clip.vision.block_count": 2,
        "clip.vision.embedding_length": 128,
        "clip.vision.feed_forward_length": 384,
        "clip.vision.attention.head_count": 4,
        "clip.vision.patch_size": 16,
        "clip.vision.spatial_merge_size": 2,
        "clip.vision.projection_dim": 512,
    }


class TestQwenGGUFConfig(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.weight = self.directory / "qwen35-Q8_0.gguf"
        self.weight.write_bytes(b"GGUF")
        self.mmproj = self.directory / "mmproj-model-f16.gguf"
        self.reader_modules = {
            "gguf": types.SimpleNamespace(GGUFReader=FakeReader),
            "sglang.srt.configs.qwen3_5": types.SimpleNamespace(
                Qwen3_5Config=FakeMultiConfig,
                Qwen3_5TextConfig=FakeConfig,
            ),
        }
        FakeReader.records = {
            str(self.weight): (
                text_metadata(),
                {"token_embd.weight": (512, 8192)},
            )
        }
        self.resolve = source_function(
            "utils/hf_transformers/common.py",
            "resolve_local_gguf_directory",
            {"Path": Path},
        )
        self.build = source_function(
            "utils/hf_transformers/gguf_native.py",
            "build_qwen3_5_gguf_config",
            {"Path": Path, "PretrainedConfig": FakeConfig},
        )

    def build_config(self):
        with patch.dict(sys.modules, self.reader_modules):
            return self.build(str(self.weight))

    def test_directory_selects_model_not_projector(self):
        self.mmproj.write_bytes(b"GGUF")
        self.assertEqual(self.resolve(self.directory), str(self.weight))
        (self.directory / "config.json").write_text("{}", encoding="utf-8")
        self.assertIsNone(self.resolve(self.directory))

    def test_argument_hook_rewrites_model_and_tokenizer_paths(self):
        args = SimpleNamespace(
            model_path=str(self.directory),
            tokenizer_path=str(self.directory),
            revision=None,
            speculative_draft_model_path=None,
        )
        shim = types.SimpleNamespace(
            resolve_local_gguf_directory=self.resolve,
            resolve_hf_gguf_reference=lambda *args, **kwargs: None,
        )
        resolve_hook = source_function(
            "arg_groups/model_path_hook.py",
            "resolve_hf_gguf_model_path",
            {
                "resolving_view": lambda server_args: server_args,
                "declare_resolution": lambda server_args, _, **changes: (
                    server_args.__dict__.update(changes)
                ),
                "logger": logging.getLogger(__name__),
            },
        )
        with patch.dict(sys.modules, {"sglang.srt.utils.hf_transformers_utils": shim}):
            resolve_hook(args)
        self.assertEqual(args.model_path, str(self.weight))
        self.assertEqual(args.tokenizer_path, str(self.weight))

    def test_ambiguous_and_projector_only_directories_fail(self):
        other = self.directory / "qwen35-Q4.gguf"
        other.write_bytes(b"GGUF")
        with self.assertRaisesRegex(ValueError, "exactly one model GGUF"):
            self.resolve(self.directory)
        other.unlink()
        self.weight.unlink()
        self.mmproj.write_bytes(b"GGUF")
        with self.assertRaisesRegex(ValueError, "found 0"):
            self.resolve(self.directory)

    def test_text_config_uses_metadata_and_weight_shape(self):
        config = self.build_config()
        self.assertEqual(config.model_type, "qwen3_5_text")
        self.assertEqual(config.vocab_size, 8192)
        self.assertEqual(config.linear_num_key_heads, 2)
        self.assertEqual(config.linear_num_value_heads, 6)
        self.assertEqual(config.partial_rotary_factor, 0.25)
        self.assertEqual(config.dtype, "bfloat16")
        self.assertIsNone(config.pad_token_id)
        self.assertEqual(config.layer_types, ["linear_attention", "full_attention"] * 2)
        self.assertEqual(config.architectures, ["Qwen3_5ForCausalLM"])

    def test_live_27b_block_count_excludes_mtp_layer(self):
        meta = FakeReader.records[str(self.weight)][0]
        meta.update(
            {
                "qwen35.block_count": 65,
                "qwen35.nextn_predict_layers": 1,
                "qwen35.full_attention_interval": 4,
                "qwen35.rope.dimension_count": 64,
                "qwen35.attention.key_length": 256,
                "qwen35.rope.freq_base": 10000000.0,
                "qwen35.rope.dimension_sections": [11, 11, 10, 0],
                "qwen35.ssm.state_size": 128,
                "qwen35.ssm.inner_size": 6144,
                "qwen35.ssm.group_count": 16,
            }
        )
        config = self.build_config()
        self.assertEqual(config.num_hidden_layers, 64)
        self.assertEqual(len(config.layer_types), 64)
        self.assertEqual(config.layer_types[-1], "full_attention")
        self.assertEqual(config.linear_num_value_heads, 48)
        self.assertEqual(config.rope_parameters["mrope_section"], [11, 11, 10])
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 0.25)
        self.assertEqual(config.rope_parameters["rope_type"], "default")

    def test_invalid_mtp_count_fails_closed(self):
        meta = FakeReader.records[str(self.weight)][0]
        meta["qwen35.nextn_predict_layers"] = meta["qwen35.block_count"]
        with self.assertRaisesRegex(ValueError, "nextn_predict_layers"):
            self.build_config()

    def test_vision_projector_builds_multimodal_config(self):
        self.mmproj.write_bytes(b"GGUF")
        tokens = [f"token-{i}" for i in range(8192)]
        for token_id, token in (
            (8053, "<|vision_start|>"),
            (8054, "<|vision_end|>"),
            (8056, "<|image_pad|>"),
            (8057, "<|video_pad|>"),
        ):
            tokens[token_id] = token
        FakeReader.records[str(self.weight)][0]["tokenizer.ggml.tokens"] = tokens
        FakeReader.records[str(self.weight)][0]["qwen35.rope.dimension_sections"] = [
            2,
            2,
            4,
            0,
        ]
        FakeReader.records[str(self.mmproj)] = (
            vision_metadata(),
            {
                "v.patch_embd.weight": (16, 16, 3, 128),
                "v.position_embd.weight": (128, 256),
            },
        )
        config = self.build_config()
        self.assertEqual(config.model_type, "qwen3_5")
        self.assertEqual(config.text_config.hidden_size, 512)
        self.assertEqual(config.vision_config.num_position_embeddings, 256)
        self.assertEqual(config.vision_config.deepstack_visual_indexes, [])
        self.assertEqual(config.rope_scaling["mrope_section"], [2, 2, 4])
        self.assertTrue(config.rope_scaling["mrope_interleaved"])
        self.assertEqual(config.image_token_id, 8056)
        self.assertEqual(config.video_token_id, 8057)
        self.assertEqual(config.vision_start_token_id, 8053)
        self.assertEqual(config.vision_end_token_id, 8054)
        self.assertIsNone(config.text_config.pad_token_id)
        self.assertEqual(config.architectures, ["Qwen3_5ForConditionalGeneration"])

    def test_multimodal_config_requires_image_marker(self):
        self.mmproj.write_bytes(b"GGUF")
        FakeReader.records[str(self.mmproj)] = (
            vision_metadata(),
            {
                "v.patch_embd.weight": (16, 16, 3, 128),
                "v.position_embd.weight": (128, 256),
            },
        )
        FakeReader.records[str(self.weight)][0]["tokenizer.ggml.tokens"] = []
        with self.assertRaisesRegex(ValueError, "<\\|image_pad\\|>"):
            self.build_config()

    def test_processor_uses_tokenizer_config_for_separate_gguf_weights(self):
        selected = []
        tokenizer = SimpleNamespace(chat_template="synthetic")
        processor = SimpleNamespace(tokenizer=tokenizer)
        def no_op(*args, **kwargs):
            return None
        load = source_function(
            "utils/hf_transformers/processor.py",
            "get_processor",
            {
                "resolve_runai_obj_uri": lambda path: path,
                "_normalize_image_processor_backend": lambda backend, use_fast: "auto",
                "is_mistral_model": lambda path: False,
                "check_gguf_file": lambda path: path == str(self.weight),
                "AutoConfig": SimpleNamespace(
                    from_pretrained=lambda path, **kwargs: (
                        selected.append(path) or SimpleNamespace(model_type="qwen3_5")
                    )
                ),
                "_is_deepseek_ocr_model": lambda config: False,
                "_is_deepseek_ocr2_model": lambda config: False,
                "_CUSTOMIZED_MM_PROCESSOR": {},
                "AutoProcessor": SimpleNamespace(
                    from_pretrained=lambda path, *args, **kwargs: processor
                ),
                "_apply_image_processor_backend": lambda proc, *args: proc,
                "PreTrainedTokenizerBase": type("PreTrainedTokenizerBase", (), {}),
                "get_tokenizer_from_processor": lambda proc: proc.tokenizer,
                "_TOKENIZERS_BACKEND": "TokenizersBackend",
                "_install_tokenizer_warnings_filter": no_op,
                "patch_mistral_common_tokenizer": no_op,
                "_fix_special_tokens_pattern": no_op,
                "_fix_added_tokens_encoding": no_op,
                "attach_additional_stop_token_ids": no_op,
            },
        )
        processor_path = str(self.directory / "fp8-processor")
        self.assertIs(load(processor_path, model_name=str(self.weight)), processor)
        self.assertEqual(selected, [processor_path])
        selected.clear()
        self.assertIs(load(processor_path, model_name="ordinary-model"), processor)
        self.assertEqual(selected, ["ordinary-model"])

    def test_missing_geometry_fails_closed(self):
        del FakeReader.records[str(self.weight)][0]["qwen35.ssm.group_count"]
        with self.assertRaisesRegex(ValueError, "qwen35.ssm.group_count"):
            self.build_config()
        self.assertIsNone(self.resolve(self.directory / "does-not-exist"))


if __name__ == "__main__":
    unittest.main()
