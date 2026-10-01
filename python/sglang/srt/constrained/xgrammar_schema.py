"""Fail-closed checks for JSON Schema features XGrammar cannot enforce."""

import copy
import json


class UnsupportedXGrammarSchema(ValueError):
    """A client schema cannot be enforced by the selected decoding grammar."""


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
    """Check active schema positions against the pinned native feature limits.

    This is not a schema validator or a general simplifier. Follow local refs
    instead of treating every unused definition as an applied constraint, and
    ignore only directly inapplicable types and exact keyword no-ops.
    """
    seen = set()

    def check_schema(obj, allow_integer_multiple=True) -> bool:
        if not isinstance(obj, dict):
            return False
        # Bare integer support does not establish support for intersections.
        # The pinned compiler drops constraints in allOf and ref siblings.
        if any(key in obj for key in ("allOf", "anyOf", "oneOf")):
            allow_integer_multiple = False
        if "$ref" in obj and any(
            key
            not in {
                "$ref",
                "$defs",
                "definitions",
                "$schema",
                "$id",
                "$comment",
                "title",
                "description",
                "default",
                "examples",
            }
            for key in obj
        ):
            allow_integer_multiple = False
        visit = (id(obj), allow_integer_multiple)
        if visit in seen:
            return False
        seen.add(visit)

        ref = obj.get("$ref")
        if ref is not None:
            if not isinstance(ref, str) or (ref != "#" and not ref.startswith("#/")):
                # The pinned compiler treats non-local references as AnySpec.
                return True
            target = schema
            try:
                for part in ref[2:].split("/") if ref != "#" else ():
                    part = part.replace("~1", "/").replace("~0", "~")
                    target = (
                        target[int(part)] if isinstance(target, list) else target[part]
                    )
            except (KeyError, IndexError, TypeError, ValueError):
                return True
            if check_schema(target, allow_integer_multiple):
                return True

        declared = obj.get("type")
        types = {declared} if isinstance(declared, str) else set(declared or ())

        def applies(*kinds):
            return not types or bool(types.intersection(kinds))

        if applies("number", "integer") and "multipleOf" in obj:
            value = obj["multipleOf"]
            # Pinned XGrammar supports unbounded integer divisors up to 1024.
            # With one-sided or wide ranges it silently drops the divisor.
            # multipleOf:1 is an exact no-op for any integer range.
            if not (
                allow_integer_multiple
                and not any(
                    key in obj
                    for key in (
                        "enum",
                        "const",
                        "not",
                        "if",
                        "then",
                        "else",
                        "$dynamicRef",
                        "$recursiveRef",
                    )
                )
                and declared == "integer"
                and type(value) in (int, float)
                and 1 <= value <= 1024
                and value == int(value)
                and (
                    value == 1
                    or not any(
                        k in obj
                        for k in (
                            "minimum",
                            "maximum",
                            "exclusiveMinimum",
                            "exclusiveMaximum",
                        )
                    )
                )
            ):
                return True

        if applies("array"):
            if "uniqueItems" in obj and obj["uniqueItems"] is not False:
                return True
            # min/maxContains have no effect without contains; count >= 0
            # with no upper bound also imposes no restriction.
            if "contains" in obj and (
                obj.get("minContains", 1) != 0 or "maxContains" in obj
            ):
                return True
            for key in ("items", "additionalItems", "unevaluatedItems"):
                if check_schema(obj.get(key), allow_integer_multiple):
                    return True
            if any(
                check_schema(item, allow_integer_multiple)
                for item in obj.get("prefixItems", [])
            ):
                return True

        if applies("object"):
            if obj.get("patternProperties") or obj.get("dependentSchemas"):
                return True
            if "propertyNames" in obj and obj["propertyNames"] not in ({}, True):
                return True
            for item in obj.get("properties", {}).values():
                if check_schema(item, allow_integer_multiple):
                    return True
            for key in ("additionalProperties", "unevaluatedProperties"):
                if check_schema(obj.get(key), allow_integer_multiple):
                    return True

        if (
            applies("string")
            and "pattern" in obj
            and (obj.get("minLength", 0) != 0 or "maxLength" in obj)
        ):
            return True

        for key in ("not", "if", "then", "else"):
            if check_schema(obj.get(key), allow_integer_multiple):
                return True
        for key in ("allOf", "anyOf", "oneOf"):
            if any(
                check_schema(item, allow_integer_multiple) for item in obj.get(key, [])
            ):
                return True
        return False

    try:
        return check_schema(schema)
    except (TypeError, ValueError, AttributeError, RecursionError):
        # Direct /generate callers need not pass through the OpenAI validator.
        return True


def normalize_xgrammar_schema_noops(schema):
    """Copy only to erase exact JSON Schema no-ops mishandled by native code.

    Preserve the original request/prompt schema. Never traverse enum/const data
    or remove an active assertion. Native strict_mode remains unchanged.
    """
    result = copy.deepcopy(schema)
    seen = set()

    def visit(obj):
        if not isinstance(obj, dict) or id(obj) in seen:
            return
        seen.add(id(obj))
        if obj.get("propertyNames") is True:
            del obj["propertyNames"]
        if "contains" not in obj:
            obj.pop("minContains", None)
            obj.pop("maxContains", None)
        for key in (
            "items",
            "additionalItems",
            "additionalProperties",
            "unevaluatedItems",
            "unevaluatedProperties",
            "not",
            "if",
            "then",
            "else",
            "contains",
            "propertyNames",
        ):
            visit(obj.get(key))
        for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
            for child in obj.get(key, []):
                visit(child)
        for key in (
            "properties",
            "patternProperties",
            "dependentSchemas",
            "$defs",
            "definitions",
        ):
            for child in obj.get(key, {}).values():
                visit(child)

    visit(result)
    return result


def validate_xgrammar_schema(schema) -> None:
    if has_xgrammar_unsupported_json_features(schema):
        raise RuntimeError(
            "JSON schema uses features unsupported by xgrammar; "
            "the constraint would otherwise be silently ignored"
        )


def sanitize_xgrammar_structural_format(structural_format):
    """Normalize/validate only schema-bearing positions, never schema data."""
    if not isinstance(structural_format, dict):
        return

    fmt_type = structural_format.get("type")
    if fmt_type in {"json_schema", "qwen_xml_parameter"}:
        if structural_format.get("json_schema") is None:
            structural_format["json_schema"] = {}
        validate_xgrammar_schema(structural_format["json_schema"])
        structural_format["json_schema"] = normalize_xgrammar_schema_noops(
            structural_format["json_schema"]
        )
        if fmt_type == "json_schema":
            validate_xgrammar_whitespace_limit(
                structural_format.get("max_whitespace_cnt")
            )

    if fmt_type in {"tag", "optional", "plus", "star", "repeat"}:
        sanitize_xgrammar_structural_format(structural_format.get("content"))
    elif fmt_type in {"sequence", "or"}:
        for element in structural_format.get("elements", []):
            sanitize_xgrammar_structural_format(element)
    elif fmt_type in {"triggered_tags", "tags_with_separator"}:
        for tag in structural_format.get("tags", []):
            sanitize_xgrammar_structural_format(tag)
    elif fmt_type in {"dispatch", "token_dispatch"}:
        for _, content in structural_format.get("rules", []):
            sanitize_xgrammar_structural_format(content)


def sanitize_xgrammar_structural_tag_structures(structural_tag: dict) -> None:
    for structure in structural_tag.get("structures", []):
        if structure.get("schema") is None:
            structure["schema"] = {}
        validate_xgrammar_schema(structure["schema"])
        structure["schema"] = normalize_xgrammar_schema_noops(structure["schema"])


def validate_xgrammar_sampling_constraints(sampling_params: dict) -> None:
    """Check only schemas in the final selected decoding constraint.

    Tool definitions can remain in the prompt without constraining parameters.
    Inspect the selected grammar after tool choice and output-format precedence,
    rather than predicting each model detector's behavior from its tool list.
    """
    try:
        if sampling_params.get("json_schema") is not None:
            schema = sampling_params["json_schema"]
            validate_xgrammar_schema(
                json.loads(schema) if isinstance(schema, str) else schema
            )
        if sampling_params.get("structural_tag") is not None:
            tag = sampling_params["structural_tag"]
            tag = json.loads(tag) if isinstance(tag, str) else tag
            if tag.get("structures") is not None:
                sanitize_xgrammar_structural_tag_structures(tag)
            else:
                sanitize_xgrammar_structural_format(tag.get("format"))
    except (RuntimeError, ValueError, TypeError, RecursionError) as exc:
        # Do not expose schema bodies, native compiler output, or model data.
        raise UnsupportedXGrammarSchema(
            "Selected decoding constraint contains a JSON schema unsupported by xgrammar."
        ) from exc
