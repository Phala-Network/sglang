import pytest

from sglang.srt.function_call.utils import normalize_json_schema_types


@pytest.mark.parametrize(
    "schema,expected",
    [
        (
            {"type": "object", "properties": {"x": {"type": "string"}}, "required": None},
            {"type": "object", "properties": {"x": {"type": "string"}}},
        ),
        (
            {"type": "object", "properties": None, "required": []},
            {"type": "object", "required": []},
        ),
        (
            {"type": "object", "properties": None, "required": None},
            {"type": "object"},
        ),
    ],
)
def test_normalize_json_schema_types_drops_optional_null_keywords(schema, expected):
    normalize_json_schema_types(schema)
    assert schema == expected
