"""Privacy and boundedness checks; no tokenizer or live requests required."""

import copy
import json
from types import SimpleNamespace as NS

from sglang.srt.function_call.kimik3_constraint_diagnostics import (
    constraint_failure_summary,
)


def tool(name="private_tool", strict=False, schema=None):
    return NS(function=NS(name=name, strict=strict, parameters=schema))


def test_counts_original_strict_without_echoing_schema_or_mutating_inputs():
    schema = {
        "oneOf": [
            {
                "type": "object",
                "if": {"properties": {"private_field": {"const": "private_value"}}},
            }
        ],
        "description": "private_description",
        "$ref": "#/private_reference",
    }
    original = copy.deepcopy(schema)
    out = constraint_failure_summary(
        [tool(schema=schema), tool(strict=True)], "auto", 2
    )
    assert out["explicit_strict_observed"] == 1
    assert out["selected_tools_observed"] == 2
    assert out["root_keywords"] == {"$ref": 1, "oneOf": 1}
    assert out["oneof_branch_keywords"] == {"type": 1, "if": 1}
    assert "private" not in json.dumps(out)
    assert schema == original


def test_named_choice_counts_only_selected_original_tool():
    out = constraint_failure_summary(
        [tool("private_a", True), tool("private_b", False)],
        NS(function=NS(name="private_b")),
        2,
    )
    assert out["choice"] == "named"
    assert out["selected_tools_observed"] == 1
    assert out["explicit_strict_observed"] == 0
    assert "private" not in json.dumps(out)


def test_diagnostic_bounds_are_reported_without_changing_admission():
    out = constraint_failure_summary(
        [tool(schema={"oneOf": [{}] * 100})] * 100, "required", 2
    )
    assert out["truncated"]
    assert out["selected_tools_observed"] == 64
    assert out["oneof_branches_observed"] == 512


def test_boolean_schema_and_unknown_choice_do_not_escape_values():
    out = constraint_failure_summary(
        [tool(schema=True), tool(schema=None)], "private_choice", 0
    )
    assert out["choice"] == "other"
    assert out["root_keywords"] == {}
    assert not out["truncated"]
    assert "private" not in json.dumps(out)


def make_parser(monkeypatch):
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    monkeypatch.setenv("SGLANG_TOOL_STRICT_LEVEL", "2")
    request_tool = Tool.model_validate(
        {
            "type": "function",
            "function": {
                "name": "private_tool",
                "strict": False,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    )
    return FunctionCallParser([request_tool], "kimi_k3")


def test_parser_failure_reports_original_strict_and_preserves_failure(
    monkeypatch, caplog
):
    parser = make_parser(monkeypatch)

    def fail(**kwargs):
        assert kwargs["tools"][0].function.strict is True
        raise ValueError("synthetic failure")

    monkeypatch.setattr(parser.detector, "get_structural_tag", fail)
    assert parser.get_structure_constraint("required") is None
    messages = [
        r.getMessage()
        for r in caplog.records
        if r.msg.startswith("Kimi constraint failure")
    ]
    assert len(messages) == 1
    assert "'explicit_strict_observed': 0" in messages[0]
    assert "'server_level': 2" in messages[0]
    assert "private" not in messages[0]
    assert parser.tools[0].function.strict is False


def test_parser_success_adds_no_failure_diagnostics(monkeypatch, caplog):
    parser = make_parser(monkeypatch)
    assert parser.get_structure_constraint("required") is not None
    assert not [
        r for r in caplog.records if str(r.msg).startswith("Kimi constraint failure")
    ]


def test_diagnostic_failure_never_changes_rejection(monkeypatch, caplog):
    from sglang.srt.function_call import kimik3_constraint_diagnostics as diagnostics

    parser = make_parser(monkeypatch)

    def fail(**kwargs):
        raise ValueError("synthetic failure")

    def diagnostic_fail(*args):
        raise RuntimeError("private diagnostic content")

    monkeypatch.setattr(parser.detector, "get_structural_tag", fail)
    monkeypatch.setattr(diagnostics, "constraint_failure_summary", diagnostic_fail)
    assert parser.get_structure_constraint("required") is None
    messages = [
        r.getMessage()
        for r in caplog.records
        if str(r.msg).startswith("Kimi constraint failure")
    ]
    assert messages == ["Kimi constraint failure metadata unavailable"]
