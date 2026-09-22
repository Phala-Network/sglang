"""Fail-closed checks for JSON Schema features XGrammar cannot enforce."""


def validate_xgrammar_whitespace_limit(
    limit, *, any_whitespace: bool = True, backend: str = "xgrammar"
) -> None:
    """An explicit bound must never silently fall back to unbounded decoding."""
    if limit is None:
        return
    if type(limit) is not int or not 0 < limit <= 2_147_483_647:
        raise ValueError("constrained JSON whitespace limit must be a positive int32")
    if backend != "xgrammar":
        raise ValueError("constrained JSON whitespace limit requires xgrammar")
    if not any_whitespace:
        raise ValueError(
            "constrained JSON whitespace limit conflicts with compact JSON"
        )


def has_xgrammar_unsupported_json_features(schema: dict) -> bool:
    """Return whether XGrammar would silently ignore a schema constraint."""

    schema_array_keywords = ("allOf", "anyOf", "oneOf", "prefixItems")
    schema_keywords = (
        "additionalItems",
        "additionalProperties",
        "else",
        "if",
        "items",
        "not",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    )
    schema_map_keywords = ("$defs", "definitions", "properties")
    unsupported_keywords = (
        "contains",
        "dependentSchemas",
        "maxContains",
        "minContains",
        "multipleOf",
        "patternProperties",
        "propertyNames",
        "uniqueItems",
    )

    def check_schema(obj) -> bool:
        if not isinstance(obj, dict):
            return False

        if any(key in obj for key in unsupported_keywords):
            return True
        # Native XGrammar returns the pattern grammar before applying lengths;
        # it does not intersect the constraints, including in patched union1.
        if "pattern" in obj and ("minLength" in obj or "maxLength" in obj):
            return True

        for key in schema_keywords:
            if check_schema(obj.get(key)):
                return True

        for key in schema_array_keywords:
            value = obj.get(key)
            if isinstance(value, list) and any(check_schema(item) for item in value):
                return True

        for key in schema_map_keywords:
            value = obj.get(key)
            if isinstance(value, dict) and any(
                check_schema(item) for item in value.values()
            ):
                return True

        return False

    return check_schema(schema)


def validate_xgrammar_schema(schema) -> None:
    if has_xgrammar_unsupported_json_features(schema):
        raise RuntimeError(
            "JSON schema uses features unsupported by xgrammar; "
            "the constraint would otherwise be silently ignored"
        )
