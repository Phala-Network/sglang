import copy
import importlib.util
from pathlib import Path

import pytest


_ENCODER_PATH = (
    Path(__file__).parents[5]
    / "python"
    / "sglang"
    / "srt"
    / "entrypoints"
    / "openai"
    / "encoding_dsv4.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "encoding_dsv4_under_test", _ENCODER_PATH
)
encoding_dsv4 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(encoding_dsv4)


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_json_schema_uses_existing_system_once_and_preserves_input():
    messages = [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "hi"},
    ]
    original = copy.deepcopy(messages)
    result = encoding_dsv4.attach_response_format_to_control_message(
        messages,
        {
            "type": "json_schema",
            "json_schema": {"schema": SCHEMA, "name": "x", "strict": True},
        },
    )
    assert messages == original
    assert result[0]["response_format"] == SCHEMA
    assert "response_format" not in result[1]
    rendered = encoding_dsv4.encode_messages(result, "chat")
    assert rendered.count("## Response Format:") == 1
    assert '"answer"' in rendered


def test_user_only_gets_synthetic_system_and_json_object():
    result = encoding_dsv4.attach_response_format_to_control_message(
        [{"role": "user", "content": "hi"}], {"type": "json_object"}
    )
    assert result[0] == {
        "role": "system",
        "content": "",
        "response_format": {"type": "object"},
    }
    assert (
        encoding_dsv4.encode_messages(result, "chat").count("## Response Format:") == 1
    )


def test_developer_only_and_tools_share_one_control_message():
    result = encoding_dsv4.attach_response_format_to_control_message(
        [
            {
                "role": "developer",
                "content": "policy",
                "tools": [
                    {"type": "function", "function": {"name": "f", "parameters": {}}}
                ],
            }
        ],
        {"type": "json_schema", "json_schema": {"schema": SCHEMA}},
    )
    assert result[0]["response_format"] == SCHEMA
    rendered = encoding_dsv4.encode_messages(result, "chat")
    assert rendered.count("## Response Format:") == 1
    assert rendered.count('"name": "f"') == 1


def test_first_system_wins_and_text_is_preserved():
    messages = [
        {"role": "system", "content": "one"},
        {"role": "system", "content": "two"},
    ]
    result = encoding_dsv4.attach_response_format_to_control_message(
        messages, {"type": "json_schema", "json_schema": {"schema": SCHEMA}}
    )
    assert result[0]["response_format"] == SCHEMA
    assert "response_format" not in result[1]
    assert (
        encoding_dsv4.attach_response_format_to_control_message(
            messages, {"type": "text"}
        )
        == messages
    )


def test_identical_control_format_allowed_but_collisions_rejected():
    existing = [{"role": "system", "content": "x", "response_format": SCHEMA}]
    assert (
        encoding_dsv4.attach_response_format_to_control_message(
            existing, {"type": "json_schema", "json_schema": {"schema": SCHEMA}}
        )
        == existing
    )
    with pytest.raises(ValueError, match="conflicting"):
        encoding_dsv4.attach_response_format_to_control_message(
            [{"role": "system", "content": "x", "response_format": {"type": "object"}}],
            {"type": "json_schema", "json_schema": {"schema": SCHEMA}},
        )
    with pytest.raises(ValueError, match="only on"):
        encoding_dsv4.attach_response_format_to_control_message(
            [
                {"role": "system", "content": "x"},
                {"role": "user", "content": "y", "response_format": SCHEMA},
            ],
            {"type": "json_schema", "json_schema": {"schema": SCHEMA}},
        )


def test_missing_schema_rejected():
    with pytest.raises(ValueError, match="requires a schema"):
        encoding_dsv4.attach_response_format_to_control_message(
            [{"role": "user", "content": "x"}],
            {"type": "json_schema", "json_schema": {}},
        )


def test_structural_tag_and_nested_unicode_schema_are_preserved():
    messages = [
        {"role": "system", "content": "系统"},
        {"role": "user", "content": "上一轮"},
        {"role": "assistant", "content": "结果"},
        {"role": "tool", "tool_call_id": "c1", "content": "工具结果"},
        {"role": "user", "content": "继续"},
    ]
    structural = {"type": "structural_tag", "structural_tag": {"tags": []}}
    assert (
        encoding_dsv4.attach_response_format_to_control_message(messages, structural)
        == messages
    )
    schema = {
        "type": "object",
        "$defs": {"名": {"type": "string", "description": "é"}},
        "properties": {"值": {"$ref": "#/$defs/名"}},
    }
    result = encoding_dsv4.attach_response_format_to_control_message(
        messages, {"type": "json_schema", "json_schema": {"schema": schema}}
    )
    assert result[0]["response_format"] == schema
    assert result[1:] == messages[1:]
