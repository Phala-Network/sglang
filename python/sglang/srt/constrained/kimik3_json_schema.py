"""K3 JSON output owns the complete native response, including reasoning.

The text parser also accepts implicit channel transitions. Deferring a JSON
matcher until one canonical think-close token sequence cannot constrain those
transitions. This format keeps the matcher active from the first token.
"""

from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)


def json_response_format(schema, thinking, max_whitespace_cnt=None):
    def const(text):
        return {"type": "const_string", "value": text}

    def sequence(*elements):
        return {"type": "sequence", "elements": list(elements)}

    def either(*elements):
        return {"type": "or", "elements": list(elements)}

    value = {"type": "json_schema", "json_schema": schema}
    if max_whitespace_cnt is not None:
        value["max_whitespace_cnt"] = max_whitespace_cnt
    closing = either(
        const(""), const(RESPONSE_CLOSE), const(RESPONSE_CLOSE + MESSAGE_CLOSE)
    )
    final = sequence(value, closing)
    response = either(final, sequence(const(RESPONSE_OPEN), final))
    if not thinking:
        return {"type": "structural_tag", "format": response}

    # A closing response marker cannot retroactively turn arbitrary reasoning
    # into unconstrained final text. It is allowed only after the JSON branch.
    excludes = [
        THINK_OPEN, THINK_CLOSE, RESPONSE_OPEN, RESPONSE_CLOSE,
        MESSAGE_CLOSE, TOOLS_OPEN, TOOLS_CLOSE,
    ]

    def reasoning_until(marker):
        return {
            "type": "tag", "begin": "",
            "content": {"type": "any_text", "excludes": excludes},
            "end": marker,
        }

    return {"type": "structural_tag", "format": either(
        sequence(reasoning_until(THINK_CLOSE), response),
        sequence(reasoning_until(RESPONSE_OPEN), final),
    )}
