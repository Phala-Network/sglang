"""Actual protocol and serving-method CPU tests, not a native serving import.

Pydantic/OpenAI types, DS encoder and async worker are real. Serving infrastructure
is replaced at the boundary; AST extraction executes unmodified source methods.
"""

import ast
import asyncio
import copy
import importlib.util
import ipaddress
import json
import logging
import re
import sys
import types
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from urllib.parse import unquote_to_bytes, urlparse, urlsplit

from jsonschema import Draft202012Validator, SchemaError
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with patch.dict(sys.modules, packages):
        spec.loader.exec_module(module)
    return module


utils = types.ModuleType("sglang.utils")
utils.convert_json_schema_to_str = json.dumps
packages = {}
for package, package_path in (
    ("sglang", SRT.parent),
    ("sglang.srt", SRT),
    ("sglang.srt.phala_compat", SRT / "phala_compat"),
):
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(package_path)]
    packages[package] = namespace

with patch.dict(sys.modules, {"sglang.utils": utils}):
    protocol = load_module("ds_test_protocol", SRT / "entrypoints/openai/protocol.py")


def execute(nodes, namespace):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module([future, *nodes], type_ignores=[])),
            "<actual source>",
            "exec",
        ),
        namespace,
    )


def source(path):
    return ast.parse((SRT / path).read_text(encoding="utf-8")).body


namespace = dict(
    vars(protocol),
    re=re,
    copy=copy,
    Enum=Enum,
    logger=logging.getLogger(__name__),
    envs=NS(SGLANG_DEFAULT_THINKING=NS(get=lambda: False)),
    ipaddress=ipaddress,
    urlparse=urlparse,
    urlsplit=urlsplit,
    unquote_to_bytes=unquote_to_bytes,
    _allowed_media_domains=frozenset(),
    get_mm=lambda: NS(media_url_max_file_size_mb=1),
    get_model=lambda: NS(context_length=4096),
    get_serving=lambda: NS(allow_auto_truncate=False),
    Draft202012Validator=Draft202012Validator,
    SchemaError=SchemaError,
    normalize_json_schema_types=lambda value: value,
)
execute(
    [
        n
        for n in source("utils/common.py")
        if getattr(n, "name", None)
        in {"_normalize_media_domain", "_assert_media_url_allowed"}
    ],
    namespace,
)
chat_nodes = source("entrypoints/openai/serving_chat.py")
owner = next(n for n in chat_nodes if getattr(n, "name", None) == "OpenAIServingChat")
methods = {
    "_apply_dsv41_reasoning_off",
    "_validate_request",
    "_validate_media_content",
    "_all_tools",
    "_effective_tools",
    "_allowed_tool_names",
    "_effective_tool_choice",
    "_request_tools_for_prompt",
    "_filter_message_tools_for_prompt",
    "_convert_to_internal_request_async",
    "_qwen35_reasoning_effort_token_range",
    "_apply_jinja_template",
    "_resolve_dsv41_reasoning_effort",
}
owner.bases = []
owner.body = [n for n in owner.body if getattr(n, "name", None) in methods]
execute(
    [
        n
        for n in chat_nodes
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and (t.id == "_MEDIA_CONTENT_PART_TYPES" or t.id.startswith("_QWEN35_"))
            for t in n.targets
        )
    ]
    + [
        n
        for n in chat_nodes
        if getattr(n, "name", None)
        in {"ThinkingMode", "normalize_assistant_tool_call_arguments"}
    ]
    + [owner],
    namespace,
)
Serving = namespace["OpenAIServingChat"]
Request = protocol.ChatCompletionRequest
worker = load_module(
    "ds_test_worker", SRT / "managers/async_dynamic_batch_tokenizer.py"
)
encoder = load_module("ds_test_encoder", SRT / "entrypoints/openai/encoding_dsv41.py")
execute(
    [
        n
        for n in source("entrypoints/openai/chat_encoding.py")
        if getattr(n, "name", None)
        in {"parse_dsv41_reasoning_effort", "dsv41_tool_payload"}
    ],
    namespace,
)
namespace["encoding_dsv41"] = encoder
namespace["chat_encoding"] = NS(
    parse_dsv41_reasoning_effort=namespace["parse_dsv41_reasoning_effort"],
    dsv41_tool_payload=namespace["dsv41_tool_payload"],
)


def serving(spec="dsv41", multimodal=False, image=False, audio=False, video=False):
    obj = Serving()
    obj.chat_encoding_spec = spec
    obj.template_manager = NS(reasoning_config=None)
    obj._grammar_backend = "none"
    obj._dsv41_default_reasoning_effort = "high" if spec == "dsv41" else None
    obj.tokenizer_manager = NS(
        model_config=NS(
            is_multimodal=multimodal,
            is_image_understandable_model=image,
            is_audio_understandable_model=audio,
        ),
        mm_processor=NS(mm_tokens=NS(video_token_id=0 if video else None)),
    )
    return obj


def request(**kwargs):
    return Request(
        model="client-alias", messages=[{"role": "user", "content": "hi"}], **kwargs
    )


def media_request(kind, url):
    holder = {"url": url}
    if kind == "input_audio":
        holder = {"data": url, "format": "wav"}
    return Request(
        model="client-alias",
        messages=[{"role": "user", "content": [{"type": kind, kind: holder}]}],
    )


class SharedProtocolTests(unittest.TestCase):
    def test_integer_transport_and_model_guard(self):
        for value in (1, 75, 100):
            req = request(reasoning_effort=value)
            self.assertIs(type(req.reasoning_effort), int)
            self.assertIsNone(serving()._validate_request(req))
            self.assertEqual(
                namespace["parse_dsv41_reasoning_effort"](req.reasoning_effort), value
            )
            for spec in (
                None,
                "dsv4",
                "dsv32",
                "inkling",
                "kimi_k3",
                "muse",
                "qwen3_5",
            ):
                with self.subTest(value=value, spec=spec):
                    self.assertIn("DeepSeek-V4.1", serving(spec)._validate_request(req))

    def test_invalid_and_other_protocol_numeric_contracts(self):
        for value in (
            True,
            False,
            -1,
            101,
            1.0,
            75.0,
            "75",
            "ultra",
            {},
            [],
            float("nan"),
        ):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                request(reasoning_effort=value)
        self.assertIsNone(request(reasoning_effort="default").reasoning_effort)
        from pydantic import TypeAdapter

        adapter = TypeAdapter(protocol.ReasoningEffortType)
        with self.assertRaises(ValidationError):
            adapter.validate_python(75)
        for value in (0.0, 0.5, 0.99):
            self.assertEqual(adapter.validate_python(value), value)
            self.assertIsNone(
                serving("inkling")._validate_request(request(reasoning_effort=value))
            )

    def test_nested_precedence_exclusion_and_template_override_preserved(self):
        req = request(
            reasoning_effort=75,
            reasoning={"effort": 0.5, "exclude": True},
            chat_template_kwargs={"reasoning_effort": 99},
        )
        self.assertEqual(req.reasoning_effort, 0.5)
        self.assertTrue(req.reasoning_exclude)
        self.assertTrue(req.chat_template_kwargs["thinking"])
        self.assertEqual(req.chat_template_kwargs["reasoning_effort"], 99)
        self.assertEqual(request(reasoning={"effort": 75}).reasoning_effort, 75)
        for nested in (True, False, 1.0, "75", "ultra"):
            with self.subTest(nested=nested), self.assertRaises(ValidationError):
                request(reasoning_effort=75, reasoning={"effort": nested})

    def test_qwen_budget_tiers_remain_bounded_and_distinct(self):
        obj = serving("qwen3_5")
        obj._uses_qwen35_chat_template = lambda: True
        with patch.dict(namespace, get_serving=lambda: NS(enable_strict_thinking=True)):
            ranges = [
                obj._qwen35_reasoning_effort_token_range(tier, 4096)
                for tier in ("low", "medium", "xhigh")
            ]
            self.assertLess(ranges[0][1], ranges[1][1])
            self.assertLess(ranges[1][1], ranges[2][1])
            for tier in ("low", "medium", "high", "xhigh", "max"):
                self.assertLessEqual(
                    obj._qwen35_reasoning_effort_token_range(tier, 4096, 75)[1], 75
                )
            self.assertIsNone(obj._qwen35_reasoning_effort_token_range(75, 4096))
            self.assertIsNone(obj._qwen35_reasoning_effort_token_range(0.5, 4096))

    def test_media_capabilities_text_and_modalities(self):
        image = media_request("image_url", "https://example.com/a.png")
        self.assertIn("text input", serving()._validate_request(image))
        ds = serving(multimodal=True, image=True)
        self.assertIsNone(ds._validate_request(image))
        for kind in ("audio_url", "video_url", "input_audio"):
            req = media_request(
                kind, "YQ==" if kind == "input_audio" else "https://example.com/a"
            )
            self.assertIn("not supported", ds._validate_request(req))
        omni = serving("qwen3_5", multimodal=True, image=True, audio=True, video=True)
        for kind in ("image_url", "audio_url", "video_url", "input_audio"):
            req = media_request(
                kind, "YQ==" if kind == "input_audio" else "https://example.com/a"
            )
            self.assertIsNone(omni._validate_request(req))

    def test_media_uri_and_domain_checks(self):
        obj = serving(multimodal=True, image=True)
        for url in (
            "C:/secret.png",
            "file:///tmp/a",
            "YWJj",
            "https:///a",
            "http://[bad",
            "data:image/png;base64",
        ):
            with self.subTest(url=url):
                self.assertIn(
                    "Invalid", obj._validate_request(media_request("image_url", url))
                )
        with patch.dict(
            namespace, _allowed_media_domains=frozenset({"allowed.example"})
        ):
            self.assertIn(
                "Invalid",
                obj._validate_request(
                    media_request("image_url", "https://other.example/a")
                ),
            )
            self.assertIsNone(
                obj._validate_request(
                    media_request("image_url", "https://allowed.example/a")
                )
            )
        with patch.dict(
            namespace,
            _assert_media_url_allowed=lambda _: (_ for _ in ()).throw(
                OSError("OS fault")
            ),
        ):
            with self.assertRaises(OSError):
                obj._validate_request(
                    media_request("image_url", "https://example.com/a")
                )

    def test_inline_size_and_malformed_base64(self):
        obj = serving(multimodal=True, image=True, audio=True)
        exact = "YWFh" * (1024 * 1024 // 3) + "YQ=="
        self.assertIsNone(
            obj._validate_request(
                media_request("image_url", "data:image/png;base64," + exact)
            )
        )
        self.assertIn(
            "size limit",
            obj._validate_request(media_request("input_audio", exact[:-4] + "YWE=")),
        )
        for invalid in ("A", "====", "YQ==AAAA", "%59Q=="):
            self.assertIn(
                "Invalid",
                obj._validate_request(
                    media_request("image_url", "data:image/png;base64," + invalid)
                ),
            )
        self.assertIsNone(
            obj._validate_request(
                media_request("image_url", "data:image/png," + "%41" * (1024 * 1024))
            )
        )
        self.assertIn(
            "size limit",
            obj._validate_request(
                media_request(
                    "image_url", "data:image/png," + "%41" * (1024 * 1024 + 1)
                )
            ),
        )

    def test_tools_none_filters_both_carriers_without_mutation(self):
        tool = {
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object"}},
        }
        req = Request(
            model="alias",
            tools=[tool],
            tool_choice="none",
            messages=[
                {"role": "system", "content": "sys", "tools": [tool]},
                {"role": "user", "content": "hi"},
            ],
        )
        obj = serving()
        messages = [m.model_dump() for m in req.messages]
        original = copy.deepcopy(messages)
        self.assertEqual(obj._request_tools_for_prompt(req), [])
        obj._filter_message_tools_for_prompt(messages, req)
        self.assertNotIn("tools", messages[0])
        self.assertEqual([m.model_dump() for m in req.messages], original)
        prompt = encoder.encode_messages(messages, thinking_mode="chat")
        self.assertNotIn('"name": "f"', prompt)

    def test_native_ds_prompt_receives_integer_and_filters_both_tool_carriers(self):
        obj = serving()
        obj.tool_call_parser = obj.reasoning_parser = "deepseekv41"
        obj.template_manager = NS(
            jinja_template_content_format="openai", reasoning_config=None
        )
        obj._fold_qwen35_system_messages = lambda messages: messages
        obj._apply_qwen35_reasoning_effort_guidance = lambda messages, effort: messages
        obj._expose_qwen35_reasoning_tool_history = lambda messages: None
        obj._encode_messages = lambda *args, **kwargs: None
        obj._handle_last_assistant_message = lambda messages, req: (messages, "")
        obj.tokenizer_manager.tokenizer = NS(encode=lambda text: list(text.encode()))
        tool = {
            "type": "function",
            "function": {"name": "secret_tool", "parameters": {"type": "object"}},
        }
        message_tool = copy.deepcopy(tool)
        message_tool["function"]["name"] = "secret_message_tool"
        req = Request(
            model="alias",
            reasoning_effort=75,
            tools=[tool],
            tool_choice="none",
            messages=[
                {"role": "system", "content": "sys", "tools": [message_tool]},
                {"role": "user", "content": "hi"},
            ],
        )
        self.assertIsNone(obj._validate_request(req))
        result = obj._apply_jinja_template(req, None, False)
        rendered = bytes(result.prompt_ids).decode()
        self.assertNotIn("secret_tool", rendered)
        self.assertNotIn("secret_message_tool", rendered)
        self.assertIn("75", rendered)
        self.assertTrue(req.tools)
        self.assertTrue(req.messages[0].tools)

    def test_async_dispatch_is_ds_only_and_uses_real_worker(self):
        async def run():
            batcher = worker.AsyncDynamicbatchTokenizer(NS(), 1, 0.001)
            try:
                for spec, ids, enabled in (
                    ("dsv41", None, True),
                    ("dsv41", [7], False),
                    ("dsv4", None, False),
                    (None, None, False),
                    ("inkling", None, False),
                    ("kimi_k3", None, False),
                ):
                    obj = serving(spec)
                    obj.tokenizer_manager.async_dynamic_batch_tokenizer = batcher
                    import threading

                    caller = threading.get_ident()
                    obj._convert_to_internal_request = lambda req, raw: (
                        threading.get_ident(),
                        req,
                    )
                    req = request(input_ids=ids)
                    thread, returned = await obj._convert_to_internal_request_async(req)
                    self.assertIs(returned, req)
                    self.assertEqual(thread != caller, enabled)
                obj = serving()
                obj._convert_to_internal_request = lambda req, raw: ("sync", req)
                self.assertEqual(
                    (await obj._convert_to_internal_request_async(request()))[0], "sync"
                )
            finally:
                batcher._executor.shutdown(wait=True)

        asyncio.run(run())

    def test_ablation_non_ds_integer_guard_is_required(self):
        ablated = copy.deepcopy(owner)
        method = next(n for n in ablated.body if n.name == "_validate_request")
        method.body = [
            n
            for n in method.body
            if not (
                isinstance(n, ast.If)
                and "Integer reasoning_effort budgets" in ast.unparse(n)
            )
        ]
        ns = dict(namespace)
        execute([ablated], ns)
        obj = serving("qwen3_5")
        obj.__class__ = ns["OpenAIServingChat"]
        self.assertIsNone(obj._validate_request(request(reasoning_effort=75)))
        self.assertIn(
            "DeepSeek",
            serving("qwen3_5")._validate_request(request(reasoning_effort=75)),
        )


if __name__ == "__main__":
    unittest.main()
