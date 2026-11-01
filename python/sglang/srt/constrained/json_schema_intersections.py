"""Bounded exact ref/minLength intersections, after dialect/resource binding.

Pure Python: no native compiler or validation-module dependency. Unsupported
intersections raise rather than approximate an assertion or an empty language.
"""
from typing import Any, Dict, Union

_JSON_TYPES = (
    "string",
    "number",
    "integer",
    "boolean",
    "array",
    "object",
    "null",
)
_SCHEMA_METADATA_KEYWORDS = {
    "$comment",
    "$defs",
    "$schema",
    "default",
    "definitions",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
}
def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "integer" if value.is_integer() else "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _matches_json_type(value: Any, json_type: str) -> bool:
    value_type = _json_type(value)
    return value_type == json_type or (
        json_type == "number" and value_type == "integer"
    )



def _possible_types(schema):
    if schema is False:
        return set()
    if schema is True:
        return set(_JSON_TYPES)
    explicit = schema.get("type")
    kinds = set([explicit] if isinstance(explicit, str) else explicit or _JSON_TYPES)
    if "number" in kinds:
        kinds.add("integer")
    if "const" in schema:
        kinds.intersection_update({_json_type(schema["const"])})
    elif isinstance(schema.get("enum"), list):
        kinds.intersection_update(_json_type(value) for value in schema["enum"])
    return kinds


def intersect_ref_min_length(schema: Union[bool, Dict[str, Any]], minimum: int, depth: int = 0, _budget=None):
    """Exact length intersection for leaf values and pure non-reference unions."""
    if _budget is None:
        _budget = [10000]
    _budget[0] -= 1
    if _budget[0] < 0:
        raise ValueError("JSON Schema minLength intersection exceeds its node limit")
    if depth > 32:
        raise ValueError("JSON Schema minLength intersection exceeds its depth limit")
    if schema is False:
        return False
    if schema is True:
        schema = {"type": list(_JSON_TYPES)}
    assertions = set(schema).difference(_SCHEMA_METADATA_KEYWORDS)
    for keyword in ("anyOf", "oneOf"):
        if assertions == {keyword} and isinstance(schema[keyword], list):
            if keyword == "oneOf":
                seen_types = set()
                for branch in schema[keyword]:
                    branch_types = _possible_types(branch)
                    if "number" in branch_types:
                        branch_types.add("integer")
                    if seen_types.intersection(branch_types):
                        raise ValueError("JSON Schema cannot prove oneOf branches disjoint for minLength")
                    seen_types.update(branch_types)
            return dict(schema, **{keyword: [
                intersect_ref_min_length(branch, minimum, depth + 1, _budget)
                for branch in schema[keyword]
            ]})
    schema_type = schema.get("type")
    types = [schema_type] if isinstance(schema_type, str) else schema_type
    if isinstance(types, list) and types and all(kind in _JSON_TYPES for kind in types) and "string" not in types:
        if assertions.intersection({"allOf", "anyOf", "oneOf", "not", "$ref"}):
            raise ValueError("JSON Schema cannot prove this $ref minLength intersection")
        return schema
    allowed = {"type", "minLength", "maxLength", "pattern", "enum", "const"}
    if assertions.difference(allowed):
        raise ValueError("JSON Schema cannot prove this $ref minLength intersection")
    result = dict(schema)
    for bound in ("minLength", "maxLength"):
        if bound in result and (type(result[bound]) is not int or result[bound] < 0):
            raise ValueError("JSON Schema string lengths must be nonnegative integers")
    minimum = max(minimum, result.get("minLength", 0))
    maximum = result.get("maxLength")
    if "enum" in result and "const" in result:
        raise ValueError("JSON Schema cannot prove combined const/enum/minLength intersection")
    if maximum is not None and minimum > maximum:
        remaining = [kind for kind in (types if types is not None else _JSON_TYPES) if kind != "string"]
        if not remaining:
            return False
        result["type"] = remaining
        types = remaining
    if "enum" in result or "const" in result:
        if "pattern" in result:
            raise ValueError("JSON Schema cannot prove enum/pattern/minLength intersection")
        def valid(value):
            return (types is None or any(_matches_json_type(value, kind) for kind in types)) and (
                not isinstance(value, str) or (len(value) >= minimum and (maximum is None or len(value) <= maximum))
            )
        if "const" in result and not valid(result["const"]):
            return False
        if "enum" in result:
            if not isinstance(result["enum"], list):
                raise ValueError("JSON Schema enum must be an array")
            result["enum"] = [value for value in result["enum"] if valid(value)]
            if not result["enum"]:
                return False
        # Keep all assertions; the filtered finite domain makes the native
        # converter's const/enum precedence safe for the newly added length.
    if types is None:
        result["type"] = list(_JSON_TYPES)
    result["minLength"] = minimum
    return result


def normalize_ref_min_length(schema, iter_subschemas):
    """Rewrite bound ref/length uses to memoized exact derived definitions.

    `schema` is the fresh output of resource normalization. That caller runs
    resource normalization once more to prune and canonically name definitions.
    Instance-valued data is never traversed. The input is modified in place.
    """
    candidates = [node for node, _ in iter_subschemas(schema)
                  if "$ref" in node and "minLength" in node]
    original_defs = schema.get("$defs", {})
    derived = {}
    cache = {}
    replacements = []
    def resolve(ref):
        if ref == "#":
            return schema
        if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
            raise ValueError("JSON Schema minLength requires a bound local reference")
        name = ref[len("#/$defs/"):]
        if "/" in name or name not in original_defs:
            raise ValueError("JSON Schema minLength reference is not a bound definition")
        return original_defs[name]
    for candidate in candidates:
        current = candidate
        minimum = 0
        seen = set()
        while isinstance(current, dict) and "$ref" in current:
            if id(current) in seen:
                raise ValueError("JSON Schema cyclic minLength reference")
            seen.add(id(current))
            if set(current).difference(_SCHEMA_METADATA_KEYWORDS, {"$ref", "minLength"}):
                raise ValueError("JSON Schema unsupported ref siblings with minLength")
            bound = current.get("minLength", 0)
            if type(bound) is not int or bound < 0:
                raise ValueError("JSON Schema string lengths must be nonnegative integers")
            minimum = max(minimum, bound)
            current = resolve(current["$ref"])
        key = (id(current), minimum)
        if key not in cache:
            if len(derived) >= 10000:
                raise ValueError("JSON Schema minLength intersection resource limit")
            reduced = intersect_ref_min_length(current, minimum)
            if isinstance(reduced, dict):
                reduced = {k: v for k, v in reduced.items() if k not in ("$defs", "definitions")}
            name = "__sglang_min_length_" + str(len(derived))
            while name in original_defs or name in derived:
                name += "_"
            derived[name] = reduced
            cache[key] = "#/$defs/" + name
        replacements.append((candidate, cache[key]))
    for candidate, ref in replacements:
        # Metadata does not constrain the instance. Keep root definitions so
        # the following canonicalization can resolve all still-active refs.
        definitions = {key: candidate[key] for key in ("$defs", "definitions", "$schema") if key in candidate}
        candidate.clear()
        candidate.update(definitions)
        candidate["$ref"] = ref
    if derived:
        schema.setdefault("$defs", {}).update(derived)
    if sum(1 for _ in iter_subschemas(schema)) > 10000:
        raise ValueError("JSON Schema minLength intersection resource limit")
    return schema
