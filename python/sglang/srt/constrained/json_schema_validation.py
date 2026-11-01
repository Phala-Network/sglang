"""Detect JSON Schema constraints that grammar backends silently ignore."""

import copy
import re
from collections.abc import Iterator
from typing import Any
from urllib.parse import unquote, urldefrag, urljoin

from sglang.srt.constrained.json_schema_dialect import (
    DEFAULT_JSON_SCHEMA_DIALECT,
    JSONSchemaDialectError,
    NON_ASSERTION_DIALECT_KEYWORDS,
    resolve_json_schema_dialect,
)
from sglang.srt.constrained.json_schema_intersections import normalize_ref_min_length


class UnsupportedJSONSchemaFeature(ValueError):
    """A schema uses constraints that the selected backend cannot preserve."""

    def __init__(self, message: str, reason_code: str = "unsupported_schema_constraint"):
        super().__init__(message)
        self._reason_code = reason_code

    @property
    def reason_code(self) -> str:
        return self._reason_code


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
        "dependencies",
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


def _resource_scope(node, resource, base_uri):
    """Advance only when an ID changes the current resource's base URI."""
    if not isinstance(node, dict) or "$id" not in node or node is resource:
        return resource, base_uri
    identifier = node["$id"]
    if not isinstance(identifier, str):
        _raise_unsupported("xgrammar", "#", "invalid_resource_identifier")
    uri, fragment = urldefrag(urljoin(base_uri, identifier))
    if fragment:
        _raise_unsupported("xgrammar", "#", "unsupported_resource_identifier_fragment")
    return (resource, base_uri) if uri == base_uri else (node, uri)


def _schema_scope(node, resource, base_uri, dialect, resource_dialect, dialects=None):
    """Bind the dialect before dropping IDs or interpreting schema keywords."""
    if dialect is None or (isinstance(node, dict) and "$schema" in node):
        try:
            declaration = (
                node["$schema"]
                if isinstance(node, dict) and "$schema" in node
                else DEFAULT_JSON_SCHEMA_DIALECT
            )
            # This memo lives for one normalization only. It avoids rechecking
            # the same immutable declaration at every root-relative reference.
            cached = (
                dialects.get(declaration)
                if dialects is not None and isinstance(declaration, str)
                else None
            )
            dialect = cached or resolve_json_schema_dialect(declaration)
            if dialects is not None:
                dialects[declaration] = dialect
        except JSONSchemaDialectError as exc:
            _raise_unsupported("xgrammar", "#", exc.reason_code)
    if not isinstance(node, dict):
        return resource, base_uri, dialect, resource_dialect or dialect
    # Draft 6/7 reference objects ignore all siblings, including identifiers.
    if not (dialect.ignores_ref_siblings and "$ref" in node):
        resource, base_uri = _resource_scope(node, resource, base_uri)
    if node is resource or resource_dialect is None:
        resource_dialect = dialect
    return resource, base_uri, dialect, resource_dialect


def resolve_local_json_pointer(
    ref: str,
    root_schema: Any,
    *,
    _scope_info=False,
    _resource_uri=None,
    _resource_dialect=None,
    _dialect_info=False,
    _dialects=None,
) -> Any:
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
    resource = root_schema
    resource_uri = _resource_uri or "https://sglang.invalid/__schema__/root"
    dialect = resource_dialect = _resource_dialect
    if _dialect_info:
        resource, resource_uri, dialect, resource_dialect = _schema_scope(
            root_schema, resource, resource_uri, dialect, resource_dialect, _dialects
        )
    if not pointer:
        if _dialect_info:
            return root_schema, resource, resource_uri, dialect, resource_dialect
        return (root_schema, resource, resource_uri) if _scope_info else root_schema
    if not pointer.startswith("/"):
        return None
    value = root_schema
    location = "schema"
    for part in pointer[1:].split("/"):
        if location == "schema" and _dialect_info:
            resource, resource_uri, dialect, resource_dialect = _schema_scope(
                value, resource, resource_uri, dialect, resource_dialect, _dialects
            )
        if (
            not _dialect_info
            and location == "schema"
            and isinstance(value, dict)
            and value is not root_schema
            and "$id" in value
        ):
            if not _scope_info:
                _raise_unsupported("xgrammar", ref, "nested $id resource scope is unsupported")
            resource, resource_uri = _resource_scope(value, resource, resource_uri)
        if re.search(r"~(?![01])", part):
            return None
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and key in value:
            child = value[key]
            if location == "schema_map":
                location = "schema"
            elif location == "schema":
                if _dialect_info and (key not in dialect.keywords or key in NON_ASSERTION_DIALECT_KEYWORDS):
                    location = "data"
                elif key in _SUBSCHEMA_MAP_KEYWORDS or key == "dependencies":
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
    if _dialect_info:
        resource, resource_uri, dialect, resource_dialect = _schema_scope(
            value, resource, resource_uri, dialect, resource_dialect, _dialects
        )
    elif isinstance(value, dict) and value is not root_schema and "$id" in value:
        if not _scope_info:
            _raise_unsupported(
                "xgrammar", ref, "nested $id resource scope is unsupported"
            )
        resource, resource_uri = _resource_scope(value, resource, resource_uri)
    if not isinstance(value, (bool, dict)):
        return None
    if _dialect_info:
        return value, resource, resource_uri, dialect, resource_dialect
    return (value, resource, resource_uri) if _scope_info else value


def normalize_json_schema_resources(schema: Any) -> Any:
    """Remove harmless resource IDs only after binding the active reference graph.

    Local fragments bind to their nearest resource root before IDs are removed.
    URI/anchor/dynamic refs remain unsupported; there is no remote retrieval.
    Unused definitions and unknown annotations are pruned. Literal data is never
    treated as schema unless an explicit reference selects it. Dialects bind
    before resource flattening, and the original input is never changed.
    """
    if not isinstance(schema, (dict, bool)):
        _raise_unsupported("xgrammar", "#", "invalid_schema")
    root_uri = "https://sglang.invalid/__schema__/root"
    dialects = {}
    _, _, root_dialect, _ = _schema_scope(schema, schema, root_uri, None, None, dialects)
    if isinstance(schema, bool):
        return schema
    output = {}
    if "$id" in schema and not (root_dialect.ignores_ref_siblings and "$ref" in schema):
        _, root_uri = _resource_scope(schema, None, root_uri)
    targets = {(id(schema), id(schema), root_dialect.uri): "#"}
    definitions = {}
    ref_edges = {}
    pending = [(schema, output, schema, root_uri, root_dialect, root_dialect, 0)]
    resource_nodes = {root_uri: {id(schema)}}
    referenced_resources = set()
    count = 0
    scheduled = 1
    needs_intersection = False

    def destination(
        source,
        resource,
        resource_uri,
        dialect,
        resource_dialect,
        depth,
        *,
        lifted=False,
    ):
        nonlocal scheduled
        scheduled += 1
        if scheduled > 10000 or depth > 128:
            _raise_unsupported("xgrammar", "#", "schema_graph_limit")
        if isinstance(source, bool):
            return source
        if not isinstance(source, dict):
            _raise_unsupported("xgrammar", "#", "invalid_subschema")
        result = {}
        # A reference may select a descendant whose resource inherited another
        # dialect. Its new position in root $defs must not change that dialect.
        if lifted and dialect != root_dialect and "$schema" not in source:
            result["$schema"] = dialect.uri
        pending.append(
            (source, result, resource, resource_uri, dialect, resource_dialect, depth)
        )
        return result

    while pending:
        node, result, resource, resource_uri, dialect, resource_dialect, depth = (
            pending.pop()
        )
        count += 1
        if count > 10000 or depth > 128:
            _raise_unsupported("xgrammar", "#", "schema_graph_limit")
        resource, resource_uri, dialect, resource_dialect = _schema_scope(
            node, resource, resource_uri, dialect, resource_dialect, dialects
        )
        resource_nodes.setdefault(resource_uri, set()).add(id(resource))
        legacy_ref = dialect.ignores_ref_siblings and "$ref" in node
        if not legacy_ref and any(
            key in node and key in dialect.keywords
            for key in ("$dynamicRef", "$recursiveRef")
        ):
            _raise_unsupported("xgrammar", "#", "unsupported_reference_kind")
        if not legacy_ref and isinstance(node.get("items"), list):
            _raise_unsupported("xgrammar", "#", "unsupported_tuple_items" if dialect.tuple_items else "invalid_schema")
        for key, value in node.items():
            if legacy_ref and key not in ("$ref", "$schema"):
                continue
            if key not in dialect.keywords or key in NON_ASSERTION_DIALECT_KEYWORDS:
                continue
            if key in ("$defs", "definitions") or (key == "$id" and node is not schema):
                continue
            if key == "additionalItems" or (key == "dependencies" and value == {}):
                # additionalItems has no effect without tuple items, which the
                # pinned converter does not support. Empty dependencies is true.
                continue
            if key == "$ref":
                needs_intersection |= not legacy_ref and "minLength" in node
                referenced_resources.add(resource_uri)
                resolved = resolve_local_json_pointer(
                    value,
                    resource,
                    _scope_info=True,
                    _resource_uri=resource_uri,
                    _resource_dialect=resource_dialect,
                    _dialect_info=True,
                    _dialects=dialects,
                )
                if resolved is None:
                    _raise_unsupported(
                        "xgrammar", "#", "unresolved_or_nonlocal_reference"
                    )
                (
                    target,
                    target_resource,
                    target_uri,
                    target_dialect,
                    target_resource_dialect,
                ) = resolved
                identity = (id(target), id(target_resource), target_dialect.uri)
                ref_edges[(id(node), id(resource), dialect.uri)] = identity
                if identity not in targets:
                    name = "__sglang_ref_" + str(len(targets))
                    targets[identity] = "#/$defs/" + name
                    definitions[name] = destination(
                        target,
                        target_resource,
                        target_uri,
                        target_dialect,
                        target_resource_dialect,
                        0,
                        lifted=True,
                    )
                result[key] = targets[identity]
            elif key in _SINGLE_SUBSCHEMA_KEYWORDS and isinstance(value, (dict, bool)):
                result[key] = destination(
                    value, resource, resource_uri, dialect, resource_dialect, depth + 1
                )
            elif (key in _SUBSCHEMA_ARRAY_KEYWORDS or key == "items") and isinstance(
                value, list
            ):
                result[key] = [
                    destination(
                        child,
                        resource,
                        resource_uri,
                        dialect,
                        resource_dialect,
                        depth + 1,
                    )
                    for child in value
                ]
            elif key in _SUBSCHEMA_MAP_KEYWORDS and isinstance(value, dict):
                result[key] = {
                    name: destination(
                        child,
                        resource,
                        resource_uri,
                        dialect,
                        resource_dialect,
                        depth + 1,
                    )
                    for name, child in value.items()
                }
            elif key == "dependencies" and isinstance(value, dict):
                result[key] = {
                    name: (
                        destination(
                            child,
                            resource,
                            resource_uri,
                            dialect,
                            resource_dialect,
                            depth + 1,
                        )
                        if isinstance(child, (dict, bool))
                        else copy.deepcopy(child)
                    )
                    for name, child in value.items()
                }
            else:
                result[key] = copy.deepcopy(value)
    if any(len(resource_nodes[uri]) > 1 for uri in referenced_resources):
        _raise_unsupported("xgrammar", "#", "ambiguous_resource_identifier")
    # Reject only non-consuming reference cycles, not recursive object/array
    # grammars. This also makes pre-scheduler validation match native refusal.
    complete = set()
    for origin in ref_edges:
        chain = set()
        current = origin
        while current in ref_edges and current not in complete:
            if current in chain:
                _raise_unsupported("xgrammar", "#", "nonconsuming_reference_cycle")
            chain.add(current)
            current = ref_edges[current]
        complete.update(chain)
    if definitions:
        output["$defs"] = definitions
    if needs_intersection:
        try:
            # Legacy reference siblings have already been removed. Derive only
            # the exact supported modern intersection, then prune inactive defs.
            reduced = normalize_ref_min_length(output, _iter_subschemas)
        except ValueError:
            _raise_unsupported("xgrammar", "#", "unsupported_reference_intersection")
        return normalize_json_schema_resources(reduced)
    return output


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
    safe_reasons = {
        "invalid_schema", "invalid_subschema", "schema_graph_limit",
        "nested_resource_reference", "unresolved_or_nonlocal_reference",
        "unsupported_reference_kind", "nonconsuming_reference_cycle",
        "invalid_resource_identifier", "unsupported_resource_identifier_fragment",
        "ambiguous_resource_identifier",
        "invalid_schema_dialect", "unsupported_schema_dialect",
        "missing_vocabulary_declaration", "invalid_vocabulary_declaration",
        "unknown_required_vocabulary", "unsupported_tuple_items",
        "unsupported_reference_intersection",
    }
    detail = ("nested_resource_reference: nested $id resource scope is unsupported"
              if reason == "nested_resource_reference" else reason)
    raise UnsupportedJSONSchemaFeature(
        f"JSON Schema at {pointer} is not supported by {backend}: {detail}",
        reason if reason in safe_reasons else "unsupported_schema_constraint",
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
    """Validate the same supported resource graph used by native normalization."""
    normalize_json_schema_resources(schema)


def validate_xgrammar_json_schema(schema: Any) -> None:
    """Reject constraints XGrammar 0.2.x accepts without fully enforcing."""
    schema = normalize_json_schema_resources(schema)
    for subschema, pointer in _iter_referenced_subschemas(schema):
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
