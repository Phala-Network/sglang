"""Assistant history must preserve final JSON while dropping only reasoning prefixes."""

import json
from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import _compile_jinja_template


@pytest.fixture(scope="module")
def template():
    return _compile_jinja_template(
        Path("/opt/phala/nemotron-lightning-chat-template.jinja").read_text()
    )


@pytest.mark.parametrize(
    "value", ["plain literal", "<think>literal</think>", "before</think>after"]
)
@pytest.mark.parametrize(
    "history_kind", ["content", "separate_reasoning", "combined_reasoning"]
)
@pytest.mark.parametrize("older", [False, True])
@pytest.mark.parametrize("with_call", [False, True])
def test_final_json_history_is_never_split_on_literal_markers(
    template, value, history_kind, older, with_call
):
    answer = json.dumps({"text": value, "count": 7})
    assistant = {"role": "assistant", "content": answer}
    if history_kind == "separate_reasoning":
        assistant["reasoning_content"] = "Private analysis of the previous turn."
    elif history_kind == "combined_reasoning":
        assistant["content"] = (
            "<think>Private analysis of the previous turn.</think>" + answer
        )
    if with_call:
        assistant["tool_calls"] = [
            {
                "type": "function",
                "function": {"name": "record", "arguments": {"ok": True}},
            }
        ]
    messages = [{"role": "user", "content": "Copy the record."}, assistant]
    if older:
        messages.append({"role": "user", "content": "Read the previous complete JSON."})
    rendered = template.render(
        messages=messages,
        add_generation_prompt=False,
        truncate_history_thinking=True,
    )
    assert rendered.count(answer) == 1, rendered
    if older:
        assert "Private analysis of the previous turn." not in rendered
    elif history_kind != "content":
        assert "Private analysis of the previous turn." in rendered
