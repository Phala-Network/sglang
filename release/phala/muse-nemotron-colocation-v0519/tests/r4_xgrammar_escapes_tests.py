"""Installed XGrammar must count decoded characters and enforce JSON escapes."""

import pytest
import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string


@pytest.mark.parametrize(
    "text,expected",
    [
        ('"abc"', True),
        (r'"a\nb"', True),
        (r'"a\"b"', True),
        (r'"a\\b"', True),
        (r'"a\u4e2db"', True),
        (r'"a\ud83d\ude00b"', True),
        ('"a中b"', True),
        ('"a😀b"', True),
        ('"a\tb"', False),
        ('"a\x00b"', False),
        ('"a\nb"', False),
        ('"ab"', False),
        ('"abcd"', False),
        (r'"\u4e2da"', False),
        (r'"a\nbc"', False),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_exact_decoded_length_and_json_validity(text, expected, nested):
    schema = {"type": "string", "minLength": 3, "maxLength": 3}
    if nested:
        schema = {
            "type": "object",
            "properties": {"text": schema},
            "required": ["text"],
            "additionalProperties": False,
        }
        text = '{"text":' + text + "}"
    grammar = xgr.Grammar.from_json_schema(schema)
    assert _is_grammar_accept_string(grammar, text) == expected
