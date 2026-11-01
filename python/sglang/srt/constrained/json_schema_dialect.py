"""Offline, fixed JSON Schema dialect declarations for grammar prevalidation.

The registry is immutable for a worker's lifetime. It is not a URI resolver or a
custom-dialect registration API. Knowing a vocabulary does not imply that the
grammar backend implements all of its assertions; its capability gate still runs.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import urldefrag

DEFAULT_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


class JSONSchemaDialectError(ValueError):
    """A dialect cannot be interpreted using the fixed local declarations."""

    def __init__(self, reason_code: str):
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class JSONSchemaDialect:
    uri: str
    keywords: frozenset[str]
    ignores_ref_siblings: bool = False
    tuple_items: bool = False
    uses_vocabularies: bool = False


_DRAFT6_KEYWORDS = frozenset(
    """$id $ref $schema additionalItems additionalProperties allOf anyOf const
    contains default definitions dependencies description enum examples
    exclusiveMaximum exclusiveMinimum format items maxItems maxLength
    maxProperties maximum minItems minLength minProperties minimum multipleOf
    not oneOf pattern patternProperties properties propertyNames required title
    type uniqueItems""".split()
)
_DRAFT7_KEYWORDS = _DRAFT6_KEYWORDS | frozenset(
    "$comment contentEncoding contentMediaType else if readOnly then writeOnly".split()
)
_DRAFT2019_KEYWORDS = (_DRAFT7_KEYWORDS - {"dependencies"}) | frozenset(
    """$anchor $defs $recursiveAnchor $recursiveRef $vocabulary contentSchema
    dependentRequired dependentSchemas deprecated maxContains minContains
    unevaluatedItems unevaluatedProperties""".split()
)
_DRAFT2020_KEYWORDS = (
    _DRAFT2019_KEYWORDS - {"additionalItems", "$recursiveRef", "$recursiveAnchor"}
) | frozenset("$dynamicAnchor $dynamicRef prefixItems".split())

# Keep definitions as legacy storage for local pointer binding. Deprecated
# meta-schema properties do not activate assertions in a later draft: dependencies
# stops at draft 7, and recursiveRef is active only in 2019-09.
_DIALECTS = MappingProxyType(
    {
        "http://json-schema.org/draft-06/schema": JSONSchemaDialect(
            "http://json-schema.org/draft-06/schema", _DRAFT6_KEYWORDS, True, True
        ),
        "http://json-schema.org/draft-07/schema": JSONSchemaDialect(
            "http://json-schema.org/draft-07/schema", _DRAFT7_KEYWORDS, True, True
        ),
        "https://json-schema.org/draft/2019-09/schema": JSONSchemaDialect(
            "https://json-schema.org/draft/2019-09/schema",
            _DRAFT2019_KEYWORDS,
            tuple_items=True,
            uses_vocabularies=True,
        ),
        DEFAULT_JSON_SCHEMA_DIALECT: JSONSchemaDialect(
            DEFAULT_JSON_SCHEMA_DIALECT,
            _DRAFT2020_KEYWORDS,
            uses_vocabularies=True,
        ),
    }
)


def _standard_vocabularies(version: str) -> Mapping[str, bool]:
    names = ["core", "applicator", "validation", "meta-data", "content"]
    names += (
        ["unevaluated", "format-annotation"] if version == "2020-12" else ["format"]
    )
    return MappingProxyType(
        {
            f"https://json-schema.org/draft/{version}/vocab/{name}": name != "format"
            for name in names
        }
    )


# Only the vocabulary declarations are needed here; no composition or retrieval
# of a meta-schema is performed. In particular, allOf/$ref do not inherit them.
_LOCAL_META_SCHEMAS = MappingProxyType(
    {
        uri: MappingProxyType(
            {
                "$vocabulary": _standard_vocabularies(
                    "2020-12" if "2020-12" in uri else "2019-09"
                )
            }
            if dialect.uses_vocabularies
            else {}
        )
        for uri, dialect in _DIALECTS.items()
    }
)
_KNOWN_VOCABULARIES = frozenset(
    uri
    for meta_schema in _LOCAL_META_SCHEMAS.values()
    for uri in meta_schema.get("$vocabulary", {})
) | {"https://json-schema.org/draft/2020-12/vocab/format-assertion"}


def validate_required_vocabularies(meta_schema: Mapping[str, Any]) -> None:
    """Check the selected meta-schema, never an ordinary schema's declaration."""
    vocabularies = meta_schema.get("$vocabulary")
    if not isinstance(vocabularies, Mapping) or not vocabularies:
        raise JSONSchemaDialectError("missing_vocabulary_declaration")
    for uri, required in vocabularies.items():
        if not isinstance(uri, str) or type(required) is not bool:
            raise JSONSchemaDialectError("invalid_vocabulary_declaration")
        if required and uri not in _KNOWN_VOCABULARIES:
            raise JSONSchemaDialectError("unknown_required_vocabulary")


def resolve_json_schema_dialect(
    declaration: Any = DEFAULT_JSON_SCHEMA_DIALECT,
) -> JSONSchemaDialect:
    """Resolve only an exact standard declaration or its scheme/empty-# alias."""
    if not isinstance(declaration, str):
        raise JSONSchemaDialectError("invalid_schema_dialect")
    uri, fragment = urldefrag(declaration)
    if fragment:
        raise JSONSchemaDialectError("unsupported_schema_dialect")
    if uri.startswith("https://json-schema.org/draft-0"):
        uri = "http://" + uri.removeprefix("https://")
    elif uri.startswith("http://json-schema.org/draft/"):
        uri = "https://" + uri.removeprefix("http://")
    dialect = _DIALECTS.get(uri)
    meta_schema = _LOCAL_META_SCHEMAS.get(uri)
    if dialect is None or meta_schema is None:
        raise JSONSchemaDialectError("unsupported_schema_dialect")
    if dialect.uses_vocabularies:
        validate_required_vocabularies(meta_schema)
    return dialect


# These are annotations/identifiers once local references have been bound. Keep
# the existing descriptive metadata intact, but do not let these names reach a
# strict tool-root assertion allowlist. format retains the service's existing
# backend capability checks; this does not add decoded-content validation.
NON_ASSERTION_DIALECT_KEYWORDS = frozenset(
    """$vocabulary $anchor $dynamicAnchor $recursiveAnchor contentEncoding
    contentMediaType contentSchema""".split()
)
