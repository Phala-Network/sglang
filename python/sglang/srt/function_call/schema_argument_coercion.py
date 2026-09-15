"""Interpret XML tool argument text using complete, local JSON Schemas."""
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.utils import safe_literal_eval

logger = logging.getLogger(__name__)


def _find_argument_schema(parameters: Dict[str, Any], arg_key: str) -> Optional[dict]:
    """Find an argument schema, including top-level schema unions."""
    properties = parameters.get("properties")
    if isinstance(properties, dict):
        arg_schema = properties.get(arg_key)
        if isinstance(arg_schema, dict):
            return arg_schema

    for keyword in ("anyOf", "oneOf"):
        choices = parameters.get(keyword)
        if not isinstance(choices, list):
            continue
        matches = [
            match
            for choice in choices
            if isinstance(choice, dict)
            for match in [_find_argument_schema(choice, arg_key)]
            if match is not None
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return {"anyOf": matches}

    all_of = parameters.get("allOf")
    if isinstance(all_of, list):
        matches = [
            match
            for choice in all_of
            if isinstance(choice, dict)
            for match in [_find_argument_schema(choice, arg_key)]
            if match is not None
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return {"allOf": matches}

    return None


def get_argument_schema(
    func_name: str, arg_key: str, defined_tools: List[Tool]
) -> Optional[dict]:
    """Return a complete argument schema with its local definitions attached."""
    name2tool = {tool.function.name: tool for tool in defined_tools}
    tool = name2tool.get(func_name)
    if tool is None:
        return None

    parameters = getattr(tool.function, "parameters", None)
    if not isinstance(parameters, dict):
        return None

    arg_schema = _find_argument_schema(parameters, arg_key)
    if arg_schema is None:
        return None

    schema_with_definitions = dict(arg_schema)
    for definitions_key in ("$defs", "definitions"):
        definitions = parameters.get(definitions_key)
        if definitions_key not in schema_with_definitions and isinstance(
            definitions, dict
        ):
            schema_with_definitions[definitions_key] = definitions
    return schema_with_definitions


def _json_type_name(value: Any) -> Optional[str]:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return None


def _schema_candidate_types(schema: Any) -> List[str]:
    """Collect candidate types without collapsing heterogeneous schemas."""
    candidates: List[str] = []

    def add(candidate: Optional[str]) -> None:
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return

        type_value = node.get("type")
        if isinstance(type_value, str):
            add(type_value)
        elif isinstance(type_value, list):
            for item in type_value:
                if isinstance(item, str):
                    add(item)

        if "const" in node:
            add(_json_type_name(node["const"]))

        enum_values = node.get("enum")
        if isinstance(enum_values, list):
            for item in enum_values:
                add(_json_type_name(item))

        for keyword in ("anyOf", "oneOf", "allOf"):
            choices = node.get(keyword)
            if isinstance(choices, list):
                for choice in choices:
                    visit(choice)

        if "properties" in node or "additionalProperties" in node:
            add("object")
        if "items" in node or "prefixItems" in node:
            add("array")

    visit(schema)
    return candidates


def _is_finite_json_value(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_finite_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_finite_json_value(item)
            for key, item in value.items()
        )
    return value is None or isinstance(value, (str, bool, int))


def _has_external_schema_ref(schema: Any) -> bool:
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str) and not ref.startswith("#"):
            return True
        return any(_has_external_schema_ref(value) for value in schema.values())
    if isinstance(schema, list):
        return any(_has_external_schema_ref(value) for value in schema)
    return False


def _schema_value_candidates(raw_value: str, schema: dict) -> List[Any]:
    """Build conservative JSON-compatible candidates from raw argument text."""
    stripped = raw_value.strip()
    candidates: List[Any] = []
    candidate_keys = set()

    def add(value: Any) -> None:
        if not _is_finite_json_value(value):
            return
        try:
            serialized = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            return
        key = (type(value).__name__, serialized)
        if key not in candidate_keys:
            candidate_keys.add(key)
            candidates.append(value)

    parsed_json: Any = None
    parsed_json_ok = False
    try:
        parsed_json = json.loads(stripped)
        parsed_json_ok = True
    except (json.JSONDecodeError, ValueError):
        pass

    parsed_literal: Any = None
    parsed_literal_ok = False
    try:
        parsed_literal = safe_literal_eval(stripped)
        parsed_literal_ok = True
    except (ValueError, SyntaxError):
        pass

    if parsed_json_ok and isinstance(parsed_json, str):
        add(parsed_json)
    elif (
        parsed_literal_ok
        and isinstance(parsed_literal, str)
        and stripped[:1] in {'"', "'"}
    ):
        add(parsed_literal)
    else:
        add(raw_value)

    if parsed_json_ok:
        add(parsed_json)
    if parsed_literal_ok:
        add(parsed_literal)

    scalar_text = stripped
    if parsed_json_ok and isinstance(parsed_json, str):
        scalar_text = parsed_json.strip()
    elif (
        parsed_literal_ok
        and isinstance(parsed_literal, str)
        and stripped[:1] in {'"', "'"}
    ):
        scalar_text = parsed_literal.strip()

    candidate_types = _schema_candidate_types(schema)
    if "integer" in candidate_types and re.fullmatch(r"[+-]?\d+", scalar_text):
        try:
            add(int(scalar_text))
        except ValueError:
            pass

    if "number" in candidate_types:
        try:
            number = float(scalar_text)
            if math.isfinite(number):
                if number.is_integer() and not any(
                    char in scalar_text for char in ".eE"
                ):
                    add(int(number))
                else:
                    add(number)
        except ValueError:
            pass

    if "boolean" in candidate_types:
        lowered = scalar_text.lower()
        if lowered == "true":
            add(True)
        elif lowered == "false":
            add(False)

    if "null" in candidate_types and scalar_text.lower() == "null":
        add(None)

    if (
        "array" in candidate_types
        and parsed_literal_ok
        and isinstance(parsed_literal, tuple)
    ):
        add(list(parsed_literal))

    return candidates


def coerce_argument_to_schema(raw_value: str, schema: dict) -> Tuple[Any, bool]:
    """Return a schema-valid candidate without inventing values from the schema."""
    if _has_external_schema_ref(schema):
        return raw_value, False

    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        valid_candidates = [
            candidate
            for candidate in _schema_value_candidates(raw_value, schema)
            if validator.is_valid(candidate)
        ]
    except Exception:
        logger.debug("Unable to validate tool argument schema", exc_info=True)
        return raw_value, False

    if not valid_candidates:
        return raw_value, False
    if len(valid_candidates) == 1:
        return valid_candidates[0], True

    for candidate in valid_candidates:
        if isinstance(candidate, str):
            return candidate, True
    return valid_candidates[0], True
