"""Bounded keyword counts for rejected constraints, never request/schema text."""

_KEYWORDS = (
    "$ref",
    "type",
    "properties",
    "required",
    "additionalProperties",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "dependentRequired",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "patternProperties",
    "propertyNames",
    "minProperties",
    "maxProperties",
)


def constraint_failure_summary(tools, tool_choice, server_level):
    """Describe original request declarations, before server-forced strictness.

    Keyword counts concern raw roots/direct oneOf branches only, not resolved
    references. No names, descriptions, values, refs, or exception text escape.
    This is failure-path diagnostics, not a validation or admission policy.
    """
    named = not isinstance(tool_choice, str)
    name = tool_choice.function.name if named else None
    root_counts = {key: 0 for key in _KEYWORDS}
    branch_counts = {key: 0 for key in _KEYWORDS}
    selected = strict = branches_seen = 0
    truncated = len(tools) > 64
    for tool in tools[:64]:
        if named and tool.function.name != name:
            continue
        selected += 1
        strict += int(tool.function.strict is True)
        schema = tool.function.parameters
        if not isinstance(schema, dict):
            continue
        for key in _KEYWORDS:
            root_counts[key] += int(key in schema)
        branches = schema.get("oneOf")
        if not isinstance(branches, list):
            continue
        truncated |= len(branches) > 8
        for branch in branches[:8]:
            branches_seen += 1
            if isinstance(branch, dict):
                for key in _KEYWORDS:
                    branch_counts[key] += int(key in branch)
    return {
        "choice": (
            "named"
            if named
            else (
                tool_choice if tool_choice in ("auto", "required", "none") else "other"
            )
        ),
        "server_level": int(server_level),
        "selected_tools_observed": selected,
        "explicit_strict_observed": strict,
        "oneof_branches_observed": branches_seen,
        "root_keywords": {k: v for k, v in root_counts.items() if v},
        "oneof_branch_keywords": {k: v for k, v in branch_counts.items() if v},
        "truncated": bool(truncated),
    }
