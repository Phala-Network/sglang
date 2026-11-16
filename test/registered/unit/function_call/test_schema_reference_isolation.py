import urllib.request

import jsonschema.validators
import pytest
from jsonschema import Draft202012Validator

from sglang.srt.function_call.utils import coerce_argument_to_schema


def _install_urlopen_guard(monkeypatch):
    urlopen_calls = []

    def unexpected_urlopen(request, *args, **kwargs):
        urlopen_calls.append(getattr(request, "full_url", request))
        raise AssertionError("tool schema validation must not retrieve URLs")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_urlopen)
    monkeypatch.setattr(
        jsonschema.validators, "urlopen", unexpected_urlopen, raising=False
    )
    return urlopen_calls


def _assert_default_validator_attempts_urlopen(schema, expected_uri, urlopen_calls):
    with pytest.raises(Exception):
        Draft202012Validator(schema).is_valid("7")
    assert urlopen_calls == [expected_uri]
    urlopen_calls.clear()


@pytest.mark.parametrize(
    ("schema", "expected_uri"),
    [
        (
            {"$ref": "https://schema.example/static.json"},
            "https://schema.example/static.json",
        ),
        (
            {"$dynamicRef": "https://schema.example/direct.json"},
            "https://schema.example/direct.json",
        ),
        (
            {
                "allOf": [
                    {"$dynamicRef": "https://schema.example/nested.json"},
                ],
            },
            "https://schema.example/nested.json",
        ),
        (
            {
                "$id": "https://schema.example/root.json",
                "$defs": {
                    "nested": {
                        "$id": "nested.json",
                        "$dynamicRef": "remote.json",
                    },
                },
                "$ref": "#/$defs/nested",
            },
            "https://schema.example/remote.json",
        ),
    ],
    ids=["static-ref", "dynamic-ref", "nested-dynamic-ref", "id-rebased"],
)
def test_external_references_fall_back_without_urlopen(
    monkeypatch, schema, expected_uri
):
    urlopen_calls = _install_urlopen_guard(monkeypatch)

    _assert_default_validator_attempts_urlopen(schema, expected_uri, urlopen_calls)

    assert coerce_argument_to_schema("7", schema) == ("7", False)
    assert urlopen_calls == []


@pytest.mark.parametrize(
    "schema",
    [
        {
            "$defs": {"integer": {"type": "integer"}},
            "$ref": "#/$defs/integer",
        },
        {
            "$defs": {"integer": {"type": "integer"}},
            "$dynamicRef": "#/$defs/integer",
        },
        {
            "$id": "https://schema.example/root.json",
            "$defs": {
                "argument": {
                    "$id": "arguments/value.json",
                    "$defs": {"integer": {"type": "integer"}},
                    "$ref": "#/$defs/integer",
                },
            },
            "$ref": "arguments/value.json",
        },
    ],
    ids=["local-defs", "local-dynamic-ref", "local-id-rebased"],
)
def test_local_references_coerce_without_retrieval(monkeypatch, schema):
    urlopen_calls = _install_urlopen_guard(monkeypatch)

    assert coerce_argument_to_schema("7", schema) == (7, True)
    assert urlopen_calls == []
