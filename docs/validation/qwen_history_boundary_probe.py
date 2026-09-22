"""Regression for historical Qwen parser contracts; exits 1 on any regression.

This executes real parser methods with a minimal output envelope and the actual
base initializer. It does not import or qualify the complete serving runtime.
"""

import ast
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
FUNCTION_CALL = ROOT / "python/sglang/srt/function_call"


@dataclass
class ToolCallItem:
    tool_index: int
    parameters: str
    name: str | None = None


@dataclass
class StreamingParseResult:
    normal_text: str = ""
    calls: list = field(default_factory=list)


base_tree = ast.parse((FUNCTION_CALL / "base_format_detector.py").read_text())
base = next(node for node in base_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "BaseFormatDetector")
base.bases = []
base.body = [node for node in base.body
             if isinstance(node, ast.FunctionDef) and node.name == "__init__"]
source = FUNCTION_CALL / "qwen3_coder_detector.py"
tree = ast.parse(source.read_text())
detector = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "Qwen3CoderDetector")
body = [
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    base,
    detector,
]
namespace = {
    "json": json, "re": re, "logger": logging.getLogger(__name__),
    "ToolCallItem": ToolCallItem, "StreamingParseResult": StreamingParseResult,
    "get_schema_properties": lambda parameters: parameters.get("properties", {}),
    "envs": SimpleNamespace(SGLANG_FORWARD_UNKNOWN_TOOLS=SimpleNamespace(get=lambda: False)),
}
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
             str(source), "exec"), namespace)
subject = namespace["Qwen3CoderDetector"]
tools = [SimpleNamespace(type="function", function=SimpleNamespace(name="probe", parameters={}))]
partial = "<tool_call><function=probe>"
unknown = "<tool_call><function=missing></function></tool_call>"
complete = "<tool_call><function=probe></function></tool_call>"
cases = {
    "truncated_stream_emits_no_call": subject().parse_streaming_increment(partial, tools).calls == [],
    "truncated_nonstream_emits_no_call": subject().detect_and_parse(partial, tools).calls == [],
    "unknown_function_is_not_executable": subject().detect_and_parse(unknown, tools).calls == [],
    "complete_empty_call_still_parses": len(subject().detect_and_parse(complete, tools).calls) == 1,
}
print(json.dumps({"source_method_probe": True, "cases": cases,
                  "passed": all(cases.values())}, indent=2))
raise SystemExit(0 if all(cases.values()) else 1)
