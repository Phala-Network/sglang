"""Validate the measured final-image template, not a mounted replacement."""

import json
from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import _compile_jinja_template


@pytest.fixture(scope="module")
def template():
    return _compile_jinja_template(
        Path("/opt/phala/nemotron-lightning-chat-template.jinja").read_text()
    )


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.parametrize("kind", ["json_object", "json_schema"])
def test_json_instruction_preserves_other_rendered_content(
    template, thinking, with_tools, kind
):
    messages = [
        {"role": "system", "content": "Keep the user system instruction."},
        {"role": "user", "content": "Extract the given record."},
    ]
    kwargs = {
        "messages": messages,
        "add_generation_prompt": True,
        "enable_thinking": thinking,
    }
    if with_tools:
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "record",
                    "description": "Store a record.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            }
        ]
    original = template.render(**kwargs)
    fmt = {"type": kind}
    schema = {
        "type": "object",
        "properties": {"value": {"const": "<think>literal</think>"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    if kind == "json_schema":
        fmt["json_schema"] = {"name": "record", "strict": True, "schema": schema}
    rendered = template.render(**kwargs, response_format=fmt)
    marker = "\n\nYour final answer must be JSON only, without code fences or prose."
    assert rendered.count(marker) == 1
    start = rendered.index(marker)
    end = (
        rendered.find("# Tools", start)
        if with_tools
        else rendered.index("<|im_end|>", start)
    )
    insertion = rendered[start:end].rstrip("\n")
    assert rendered.replace(insertion, "", 1) == original
    if kind == "json_schema":
        assert (
            json.loads(insertion.split(" It must satisfy this JSON Schema: ", 1)[1])
            == schema
        )
    else:
        assert insertion == marker


def test_non_json_format_leaves_template_identical(template):
    kwargs = {
        "messages": [{"role": "user", "content": "Say ready."}],
        "add_generation_prompt": True,
    }
    assert template.render(
        **kwargs, response_format={"type": "text"}
    ) == template.render(**kwargs)
