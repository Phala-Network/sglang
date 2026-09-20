import xgrammar as xgr

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector


def _tool():
    return Tool(
        type="function",
        function=Function(
            name="search_catalog",
            description="Search the product catalog",
            strict=True,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query", "max_results"],
                "additionalProperties": False,
            },
        ),
    )


def _repetition(structural_tag):
    matches = []

    def visit(value):
        if isinstance(value, dict):
            if value.get("type") == "tags_with_separator":
                matches.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(structural_tag.model_dump())
    assert len(matches) == 1
    return matches[0]


def test_required_parallel_uses_official_template_separator():
    tag = Qwen3CoderDetector().get_structural_tag(
        [_tool()],
        tool_choice="required",
        thinking_mode=False,
        parallel_tool_calls=True,
    )

    assert isinstance(tag, xgr.StructuralTag)
    assert isinstance(xgr.Grammar.from_structural_tag(tag), xgr.Grammar)
    assert _repetition(tag)["separator"] == "\n"
    assert _repetition(tag)["stop_after_first"] is False


def test_required_serial_stops_after_first_call():
    tag = Qwen3CoderDetector().get_structural_tag(
        [_tool()],
        tool_choice="required",
        thinking_mode=False,
        parallel_tool_calls=False,
    )

    assert isinstance(tag, xgr.StructuralTag)
    assert isinstance(xgr.Grammar.from_structural_tag(tag), xgr.Grammar)
    assert _repetition(tag)["separator"] == "\n"
    assert _repetition(tag)["stop_after_first"] is True


def test_required_reasoning_prefix_is_preserved():
    tag = Qwen3CoderDetector().get_structural_tag(
        [_tool()],
        tool_choice="required",
        thinking_mode=True,
        parallel_tool_calls=True,
    )
    serialized = tag.model_dump_json()

    assert "</think>" in serialized
    assert "\\n\\n" in serialized
    assert _repetition(tag)["separator"] == "\n"
