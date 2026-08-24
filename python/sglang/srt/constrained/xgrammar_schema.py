"""Fail-closed checks for JSON Schema features XGrammar cannot enforce."""


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
