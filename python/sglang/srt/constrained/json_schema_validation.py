"""Detect JSON Schema constraints that grammar backends silently ignore."""

import re
from collections.abc import Iterator
from typing import Any
from urllib.parse import unquote


class UnsupportedJSONSchemaFeature(ValueError):
    """A schema uses constraints that the selected backend cannot preserve."""


_SINGLE_SUBSCHEMA_KEYWORDS = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SUBSCHEMA_ARRAY_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SUBSCHEMA_MAP_KEYWORDS = frozenset(
    {"$defs", "definitions", "dependentSchemas", "patternProperties", "properties"}
)

_XGRAMMAR_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "contains",
        "dependentRequired",
        "dependentSchemas",
        "else",
        "if",
        "maxContains",
        "minContains",
        "multipleOf",
        "not",
        "then",
        "uniqueItems",
    }
)
_XGRAMMAR_STRING_FORMATS = frozenset(
    {
        "date",
        "date-time",
        "duration",
        "email",
        "hostname",
        "ipv4",
        "ipv6",
        "json-pointer",
        "relative-json-pointer",
        "time",
        "uri",
        "uri-reference",
        "uri-template",
        "uuid",
    }
)

# Outlines Core 0.1.x accepts schemas containing these assertion keywords but
# does not encode them in the generated regex.
_OUTLINES_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "allOf",
        "contains",
        "dependentRequired",
        "dependentSchemas",
        "else",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "if",
        "maxContains",
        "maximum",
        "minContains",
        "minimum",
        "multipleOf",
        "not",
        "oneOf",
        "patternProperties",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
        "uniqueItems",
    }
)


def _escape_json_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def resolve_local_json_pointer(ref: str, root_schema: Any) -> Any:
    """Resolve one-resource pointers shared by preflight and K3 extraction."""
    if ref == "":
        ref = "#"
    if (
        not isinstance(ref, str)
        or not ref.startswith("#")
        or re.search(r"%(?![0-9A-Fa-f]{2})", ref)
    ):
        return None
    try:
        pointer = unquote(ref[1:], encoding="utf-8", errors="strict")
    except UnicodeError:
        return None
    if not pointer:
        return root_schema
    if not pointer.startswith("/"):
        return None
    value = root_schema
    location = "schema"
    for part in pointer[1:].split("/"):
        if (
            location == "schema"
            and isinstance(value, dict)
            and value is not root_schema
            and "$id" in value
        ):
            _raise_unsupported("xgrammar", ref, "nested $id resource scope is unsupported")
        if re.search(r"~(?![01])", part):
            return None
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and key in value:
            child = value[key]
            if location == "schema_map":
                location = "schema"
            elif location == "schema":
                if key in _SUBSCHEMA_MAP_KEYWORDS or key == "dependencies":
                    location = "schema_map"
                elif key in _SUBSCHEMA_ARRAY_KEYWORDS or (
                    key == "items" and isinstance(child, list)
                ):
                    location = "schema_array"
                elif key not in _SINGLE_SUBSCHEMA_KEYWORDS:
                    location = "data"
            else:
                location = "data"
            value = child
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", key):
            if len(key) > len(str(len(value))) or int(key) >= len(value):
                return None
            value = value[int(key)]
            location = "schema" if location == "schema_array" else "data"
        else:
            return None
    if isinstance(value, dict) and value is not root_schema and "$id" in value:
        _raise_unsupported("xgrammar", ref, "nested $id resource scope is unsupported")
    return value if isinstance(value, (bool, dict)) else None


def _iter_subschemas(
    schema: Any, pointer: str = "#", *, include_definitions: bool = True
) -> Iterator[tuple[dict[str, Any], str]]:
    """Walk schema-bearing keywords without inspecting instance-valued data."""
    if not isinstance(schema, dict):
        return

    yield schema, pointer

    for keyword in _SINGLE_SUBSCHEMA_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, (bool, dict)):
            yield from _iter_subschemas(child, f"{pointer}/{keyword}", include_definitions=include_definitions)
        elif keyword == "items" and isinstance(child, list):
            for index, item in enumerate(child):
                yield from _iter_subschemas(item, f"{pointer}/{keyword}/{index}", include_definitions=include_definitions)

    for keyword in _SUBSCHEMA_ARRAY_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list):
            for index, child in enumerate(children):
                yield from _iter_subschemas(child, f"{pointer}/{keyword}/{index}", include_definitions=include_definitions)

    for keyword in _SUBSCHEMA_MAP_KEYWORDS:
        if not include_definitions and keyword in ("$defs", "definitions"):
            continue
        children = schema.get(keyword)
        if isinstance(children, dict):
            for name, child in children.items():
                escaped_name = _escape_json_pointer(name)
                yield from _iter_subschemas(
                    child, f"{pointer}/{keyword}/{escaped_name}", include_definitions=include_definitions
                )

    dependencies = schema.get("dependencies")
    if isinstance(dependencies, dict):
        for name, child in dependencies.items():
            if isinstance(child, (bool, dict)):
                escaped_name = _escape_json_pointer(name)
                yield from _iter_subschemas(
                    child, f"{pointer}/dependencies/{escaped_name}", include_definitions=include_definitions
                )


def _raise_unsupported(backend: str, pointer: str, reason: str) -> None:
    raise UnsupportedJSONSchemaFeature(
        f"JSON Schema at {pointer} is not supported by {backend}: {reason}"
    )


def _iter_referenced_subschemas(schema: Any) -> Iterator[tuple[dict[str, Any], str]]:
    """Visit annotations only when a schema reference makes them active."""
    pending = [(schema, "#")]
    seen = set()
    while pending:
        root, pointer = pending.pop()
        if id(root) in seen:
            continue
        for node, location in _iter_subschemas(root, pointer):
            if id(node) in seen:
                continue
            seen.add(id(node))
            yield node, location
            ref = node.get("$ref")
            if isinstance(ref, str):
                target = resolve_local_json_pointer(ref, schema)
                if isinstance(target, dict) and id(target) not in seen:
                    pending.append((target, ref))


def _string_constraint_groups(schema: dict[str, Any]) -> list[str]:
    groups = []
    if "format" in schema:
        groups.append("format")
    if "pattern" in schema:
        groups.append("pattern")
    if "minLength" in schema or "maxLength" in schema:
        groups.append("minLength/maxLength")
    return groups


def _can_describe_string(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    return (
        schema_type is None
        or schema_type == "string"
        or (isinstance(schema_type, list) and "string" in schema_type)
    )


def validate_local_json_reference_scope(schema: Any) -> None:
    """The pinned converter supports one schema resource, not nested $id bases."""
    for subschema, pointer in _iter_referenced_subschemas(schema):
        if subschema is not schema and "$id" in subschema:
            _raise_unsupported("xgrammar", pointer, "nested $id resource scope is unsupported")


def validate_xgrammar_json_schema(schema: Any) -> None:
    """Reject constraints XGrammar 0.2.x accepts without fully enforcing."""
    for subschema, pointer in _iter_referenced_subschemas(schema):
        if subschema is not schema and "$id" in subschema:
            _raise_unsupported("xgrammar", pointer, "nested $id resource scope is unsupported")
        unsupported = sorted(_XGRAMMAR_UNSUPPORTED_KEYWORDS.intersection(subschema))
        if subschema.get("uniqueItems") is False:
            unsupported.remove("uniqueItems")
        if unsupported:
            _raise_unsupported(
                "xgrammar",
                pointer,
                f"keyword(s) {', '.join(unsupported)} would be ignored",
            )

        if _can_describe_string(subschema) and "format" in subschema:
            format_name = subschema["format"]
            if (
                not isinstance(format_name, str)
                or format_name not in _XGRAMMAR_STRING_FORMATS
            ):
                _raise_unsupported(
                    "xgrammar",
                    pointer,
                    f"string format {format_name!r} is not implemented",
                )

        groups = (
            _string_constraint_groups(subschema)
            if _can_describe_string(subschema)
            else []
        )
        # The pinned converter composes pattern with string-length metadata.
        # format+pattern still selects one branch and would drop an assertion.
        if len(groups) > 1 and set(groups) != {"pattern", "minLength/maxLength"}:
            _raise_unsupported(
                "xgrammar",
                pointer,
                f"constraints {', '.join(groups)} cannot be enforced together",
            )


def validate_outlines_json_schema(schema: Any) -> None:
    """Reject constraints Outlines Core 0.1.x accepts without preserving."""
    for subschema, pointer in _iter_subschemas(schema):
        unsupported = sorted(_OUTLINES_UNSUPPORTED_KEYWORDS.intersection(subschema))
        if unsupported:
            _raise_unsupported(
                "outlines",
                pointer,
                f"keyword(s) {', '.join(unsupported)} would be ignored or weakened",
            )

        groups = (
            _string_constraint_groups(subschema)
            if _can_describe_string(subschema)
            else []
        )
        if len(groups) > 1:
            _raise_unsupported(
                "outlines",
                pointer,
                f"constraints {', '.join(groups)} cannot be enforced together",
            )

        if "properties" in subschema:
            ignored_object_constraints = {
                "additionalProperties",
                "maxProperties",
                "minProperties",
            }.intersection(subschema)
            if subschema.get("additionalProperties") is False:
                # The properties regex already excludes undeclared keys.
                ignored_object_constraints.discard("additionalProperties")
            if ignored_object_constraints:
                _raise_unsupported(
                    "outlines",
                    pointer,
                    "properties cannot be combined with "
                    + ", ".join(sorted(ignored_object_constraints)),
                )

        if "prefixItems" in subschema:
            ignored_array_constraints = sorted(
                {"maxItems", "minItems"}.intersection(subschema)
            )
            if ignored_array_constraints:
                _raise_unsupported(
                    "outlines",
                    pointer,
                    "prefixItems cannot be combined with "
                    + ", ".join(ignored_array_constraints),
                )

        if "required" in subschema and "properties" not in subschema:
            _raise_unsupported(
                "outlines",
                pointer,
                "required is only enforced when properties is present",
            )
