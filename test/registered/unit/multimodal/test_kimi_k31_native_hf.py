"""Real K3.1 metadata entrypoints with pinned, data-only checkpoint fixtures.

Set KIMI_K31_TEST_DATA to a directory containing target/, draft/, and
manifest.json. Image qualification must provide the fixture explicitly; the
ordinary CPU suite can skip this integration check when its BPE data is absent.
"""

import copy
import hashlib
import importlib
import inspect
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

EXPECTED_DATA = {
    "target/config.json": (
        7303,
        "d95037631e3f23bf527a9f476f25209b231b46853a84877b9c56d5672b8d1202",
    ),
    "target/tokenizer_config.json": (
        3448,
        "9fc322181b8ec44236e148916aea4b6c2467b66ded7bfaff0efe623ad750b852",
    ),
    "target/preprocessor_config.json": (
        1011,
        "4be333605990c53a816e586dee9d5dd545afb7a59947c17f8f7ef26b4782668e",
    ),
    "target/tiktoken.model": (
        2795286,
        "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    ),
    "draft/config.json": (
        2050,
        "b59ceb231cbec61f287968de7ecdb8520c786f0a813088b7636753fd423e537a",
    ),
}


class TestKimiK31NativeHF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = os.environ.get("KIMI_K31_TEST_DATA")
        if not fixture:
            raise unittest.SkipTest("KIMI_K31_TEST_DATA data fixture is not configured")
        cls.fixture = Path(fixture).resolve()
        cls.target = str(cls.fixture / "target")
        cls.draft = str(cls.fixture / "draft")
        cls.blocked = []
        module_cache = Path(os.environ.get("HF_MODULES_CACHE", "/tmp/hf/modules"))
        protected = (cls.fixture, module_cache.resolve())

        def audit(event, args):
            if event != "exec":
                return
            filename = getattr(args[0], "co_filename", "")
            path = Path(filename).resolve()
            if "transformers_modules" in path.parts or any(
                path.is_relative_to(root) for root in protected
            ):
                cls.blocked.append({"entry": "exec", "file": filename})
                raise RuntimeError("checkpoint/cache Python execution is forbidden")

        sys.addaudithook(audit)

        def deny(*args, **kwargs):
            cls.blocked.append({"entry": "HF dynamic model-code loader"})
            raise RuntimeError("HF dynamic model-code loading is forbidden")

        # Cover lazy imports as well as aliases imported by Transformers.
        for name in (
            "transformers.dynamic_module_utils",
            "transformers.models.auto.configuration_auto",
            "transformers.models.auto.tokenization_auto",
            "transformers.models.auto.processing_auto",
            "transformers.models.auto.image_processing_auto",
        ):
            module = importlib.import_module(name)
            for attr in ("get_class_from_dynamic_module", "get_cached_module_file"):
                if hasattr(module, attr):
                    guard = patch.object(module, attr, deny)
                    guard.start()
                    cls.addClassCleanup(guard.stop)

        from sglang.srt.utils.hf_transformers import (
            get_config,
            get_processor,
            get_tokenizer,
        )

        cls.rows = []
        cls.processors = {}
        cls.tokenizers = {}
        for trust in (False, True):
            config = get_config(cls.target, trust_remote_code=trust)
            draft = get_config(cls.draft, trust_remote_code=trust)
            tokenizer = get_tokenizer(
                cls.target,
                trust_remote_code=trust,
                revision="6d0a0202956796543364d666ab19c6ad1a494ee1",
            )
            processor = get_processor(
                cls.target,
                trust_remote_code=trust,
                tokenizer_revision="6d0a0202956796543364d666ab19c6ad1a494ee1",
                image_processor_backend="auto" if trust else "pil",
            )
            cls.tokenizers[trust] = tokenizer
            cls.processors[trust] = processor
            for entry, obj in (
                ("config", config),
                ("draft_config", draft),
                ("tokenizer", tokenizer),
                ("processor", processor),
                ("processor.tokenizer", processor.tokenizer),
                ("processor.image_processor", processor.image_processor),
            ):
                cls.rows.append(
                    {
                        "entry": entry,
                        "trust_remote_code": trust,
                        "class": type(obj).__name__,
                        "module": type(obj).__module__,
                        "file": inspect.getfile(type(obj)),
                    }
                )
        print("NATIVE_HF_SELECTION=" + json.dumps(cls.rows), flush=True)

    def test_pinned_data_only_fixture(self):
        manifest = json.loads((self.fixture / "manifest.json").read_text())
        self.assertEqual(
            manifest["target_revision"], "6d0a0202956796543364d666ab19c6ad1a494ee1"
        )
        self.assertEqual(
            manifest["draft_revision"], "d0d8bc2e3c6d0a195d2e00a49017e71ca8d881f1"
        )
        for row in manifest["files"]:
            data = (self.fixture / row["path"]).read_bytes()
            self.assertEqual((row["bytes"], row["sha256"]), EXPECTED_DATA[row["path"]])
            self.assertEqual(
                (len(data), hashlib.sha256(data).hexdigest()),
                EXPECTED_DATA[row["path"]],
            )
        self.assertEqual({row["path"] for row in manifest["files"]}, set(EXPECTED_DATA))
        self.assertEqual(
            sorted(p.suffix for p in self.fixture.rglob("*") if p.is_file()),
            [".json", ".json", ".json", ".json", ".json", ".model"],
        )
        config = json.loads((Path(self.target) / "config.json").read_text())
        self.assertIn("AutoConfig", config["auto_map"])
        tokenizer = json.loads(
            (Path(self.target) / "tokenizer_config.json").read_text()
        )
        self.assertIn("AutoTokenizer", tokenizer["auto_map"])
        processor = json.loads(
            (Path(self.target) / "preprocessor_config.json").read_text()
        )
        self.assertIn("AutoProcessor", processor["auto_map"])
        self.assertIn("AutoImageProcessor", processor["auto_map"])

    def test_native_classes_win_with_both_trust_settings(self):
        expected = {
            "config": "sglang.srt.configs.kimi_k3",
            "draft_config": "transformers.models.qwen3.configuration_qwen3",
            "tokenizer": "sglang.srt.multimodal.kimi_k31_hf.tokenization_kimi",
            "processor": "sglang.srt.multimodal.kimi_k31_hf.kimi_k3_processor",
            "processor.tokenizer": "sglang.srt.multimodal.kimi_k31_hf.tokenization_kimi",
            "processor.image_processor": (
                "sglang.srt.multimodal.kimi_k31_hf.kimi_k3_vision_processing"
            ),
        }
        for row in self.rows:
            self.assertEqual(row["module"], expected[row["entry"]])
            self.assertFalse(Path(row["file"]).resolve().is_relative_to(self.fixture))
            self.assertNotIn("transformers_modules", Path(row["file"]).parts)
        self.assertEqual(self.blocked, [])
        for module in tuple(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename:
                self.assertFalse(Path(filename).resolve().is_relative_to(self.fixture))
                self.assertNotIn("transformers_modules", Path(filename).parts)
        package = Path(inspect.getfile(type(self.tokenizers[False]))).parent
        sources = json.loads((package / "SOURCES.json").read_text())
        self.assertEqual(
            sources["revision"], "6d0a0202956796543364d666ab19c6ad1a494ee1"
        )
        for row in sources["modules"] + [sources["license"]]:
            data = (package / row["name"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])
            self.assertEqual(
                hashlib.sha1(
                    b"blob " + str(len(data)).encode() + b"\0" + data
                ).hexdigest(),
                row["git_blob"],
            )

    def test_utf8_roundtrip(self):
        text = '你好 🌏 café Ελληνικά\nTools: {"city":"杭州"}\tend'
        for tokenizer in self.tokenizers.values():
            ids = tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(tokenizer.decode(ids, skip_special_tokens=False), text)

    def test_xtml_reasoning_and_tool_result_order(self):
        messages = [
            {"role": "user", "content": "Check the two cities."},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Inspect both tools first.",
                "tool_calls": [
                    {
                        "id": "opaque-a",
                        "type": "function",
                        "function": {
                            "name": "weather_a",
                            "arguments": '{"city":"杭州"}',
                        },
                    },
                    {
                        "id": "opaque-b",
                        "type": "function",
                        "function": {
                            "name": "weather_b",
                            "arguments": {"city": "Paris"},
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "opaque-b", "content": "result-B"},
            {"role": "tool", "tool_call_id": "opaque-a", "content": "result-A"},
        ]
        original = copy.deepcopy(messages)
        for processor in self.processors.values():
            rendered = processor.apply_chat_template(messages, tokenize=False)
            self.assertIn("Inspect both tools first.", rendered)
            self.assertIn("thinking_effort=max", rendered)
            self.assertIn("weather_a", rendered)
            self.assertIn("weather_b", rendered)
            self.assertLess(rendered.index("result-A"), rendered.index("result-B"))
            ids = processor.apply_chat_template(messages, tokenize=True)
            self.assertEqual(processor.decode(ids, skip_special_tokens=False), rendered)
        self.assertEqual(messages, original)

    def test_rgba_image_preprocessing_and_resolution_prompt(self):
        import numpy as np
        import torch
        from PIL import Image

        rgba = np.zeros((28, 56, 4), dtype=np.uint8)
        rgba[..., 0] = 251
        rgba[..., 1] = 17
        rgba[..., 2] = 63
        rgba[..., 3] = 0
        rgba[4:20, 8:40, 3] = 255
        image = Image.fromarray(rgba, "RGBA")
        outputs = []
        for processor in self.processors.values():
            result = processor(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this."},
                            {"type": "image", "image": image},
                        ],
                    }
                ],
                return_tensors="pt",
            )
            self.assertEqual(result["grid_thws"].tolist(), [[1, 2, 4]])
            self.assertEqual(list(result["pixel_values"].shape), [8, 3, 14, 14])
            self.assertTrue(torch.isfinite(result["pixel_values"]).all())
            torch.testing.assert_close(
                result["pixel_values"][0, :, 0, 0], torch.ones(3)
            )
            torch.testing.assert_close(
                result["pixel_values"][0, :, 0, 8], torch.full((3,), 180 / 127.5 - 1)
            )
            torch.testing.assert_close(
                result["pixel_values"][0, :, 4, 8],
                torch.tensor([251.0, 17.0, 63.0]) / 127.5 - 1,
            )
            self.assertIn(
                "<|media_begin|>image 56x28<|media_content|><|media_pad|><|media_end|>",
                processor.decode(result["input_ids"][0].tolist()),
            )
            self.assertIs(processor.media_processor, processor.image_processor)
            outputs.append(result)
        torch.testing.assert_close(
            outputs[0]["pixel_values"], outputs[1]["pixel_values"]
        )
        self.assertEqual(self.blocked, [])


if __name__ == "__main__":
    unittest.main()
