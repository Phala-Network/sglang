import ast
import json
import logging
import math
import re
import threading
import warnings
from json import JSONDecodeError, JSONDecoder
from json.decoder import WHITESPACE
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import orjson
import partial_json_parser
from jsonschema import Draft202012Validator
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice

logger = logging.getLogger(__name__)

_STANDARD_JSON_SCHEMA_TYPES = {
    "null",
    "boolean",
    "object",
    "array",
    "number",
    "string",
    "integer",
}

# Non-standard ``type`` values commonly emitted by DB/ORM-driven tool-schema
# generators. Mapped to the closest JSON Schema 2020-12 primitive so that
# ``Draft202012Validator.check_schema`` does not reject an otherwise-usable
# tool definition.
_JSON_SCHEMA_TYPE_ALIASES: Dict[str, str] = {
    "str": "string",
    "text": "string",
    "varchar": "string",
    "char": "string",
    "enum": "string",
    "uuid": "string",
    "date": "string",
    "datetime": "string",
    "time": "string",
    "timestamp": "string",
    "binary": "string",
    "blob": "string",
    "bytea": "string",
    "bytes": "string",
    "varbinary": "string",
    "bool": "boolean",
    "bigint": "integer",
    "smallint": "integer",
    "tinyint": "integer",
    "double": "number",
    "decimal": "number",
    "real": "number",
    "numeric": "number",
    "arr": "array",
    "tuple": "array",
    "set": "array",
    "map": "object",
}

# Prefix-based matching so that parameterised names like ``int32`` /
# ``float64`` / ``list[str]`` / ``dict[str, int]`` resolve. A prefix only
# matches when it spans the entire token or is followed by a non-identifier
# char, so "int" does not swallow "internal" and "list" does not swallow
# "list_price".
_PREFIX_BOUNDARY_CHARS = frozenset("0123456789[<( \t")
_PREFIX_RULES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("int", "uint", "long", "short", "unsigned"), "integer"),
    (("num", "float"), "number"),
    (("list",), "array"),
    (("dict",), "object"),
)


def _matches_type_prefix(base: str, prefixes: Tuple[str, ...]) -> bool:
    for p in prefixes:
        if base == p:
            return True
        if (
            len(base) > len(p)
            and base.startswith(p)
            and base[len(p)] in _PREFIX_BOUNDARY_CHARS
        ):
            return True
    return False


def _normalize_single_type(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    if raw in _STANDARD_JSON_SCHEMA_TYPES:
        return raw
    # ``split("(", 1)[0]`` strips parenthesized params like ``varchar(255)``
    # or ``decimal(10,2)`` without the overhead of a regex per call.
    base = raw.split("(", 1)[0].strip().lower()
    if base in _STANDARD_JSON_SCHEMA_TYPES:
        return base
    mapped = _JSON_SCHEMA_TYPE_ALIASES.get(base)
    if mapped is not None:
        return mapped
    for prefixes, target in _PREFIX_RULES:
        if _matches_type_prefix(base, prefixes):
            return target
    return raw


def _normalize_type_list(raw_items: List[Any]) -> List[Any]:
    normalized_items: List[Any] = []
    for item in raw_items:
        normalized_item = _normalize_single_type(item)
        if normalized_item not in normalized_items:
            normalized_items.append(normalized_item)
    return normalized_items


def normalize_json_schema_types(schema: Any) -> None:
    """
    Walk a JSON Schema in place and rewrite non-standard ``"type"`` values
    (e.g. ``"varchar"``, ``"enum"``, ``"int"``) to their standard JSON Schema
    equivalents.

    Acts as a compatibility layer for tool ``parameters`` schemas exported
    from database / ORM tooling, which often uses DB type names rather than
    JSON Schema types. Unknown types are left untouched so that downstream
    validation can still surface genuine errors.

    Mutates the input dict in place; the rewritten schema is also what gets
    rendered into the model prompt, so e.g. a user-supplied ``"varchar"``
    reaches the model as ``"string"``. ``$ref`` values are not resolved;
    callers pass tree-shaped schemas (HTTP JSON input is always a tree).
    """
    if isinstance(schema, list):
        for item in schema:
            normalize_json_schema_types(item)
        return
    if not isinstance(schema, dict):
        return

    if "type" in schema:
        t = schema["type"]
        if isinstance(t, str):
            schema["type"] = _normalize_single_type(t)
        elif isinstance(t, list):
            schema["type"] = _normalize_type_list(t)

    for key in (
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    ):
        nested = schema.get(key)
        if isinstance(nested, dict):
            for v in nested.values():
                normalize_json_schema_types(v)

    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        nested = schema.get(key)
        if isinstance(nested, list):
            for v in nested:
                normalize_json_schema_types(v)

    for key in (
        "items",
        "additionalProperties",
        "not",
        "if",
        "then",
        "else",
        "contains",
        "propertyNames",
        "unevaluatedItems",
        "unevaluatedProperties",
    ):
        if key in schema:
            normalize_json_schema_types(schema[key])


def _find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def _partial_json_loads(input_str: str, flags: Allow) -> Tuple[Any, int]:
    """
    Parse incomplete or partial JSON strings commonly encountered during streaming.

    Args:
        input_str (str): The potentially incomplete JSON string to parse.
        flags (Allow): Bitwise flags controlling what types of partial data are allowed.
            Common flags include:
            - Allow.STR: Allow partial strings (e.g., '"hello wo' -> 'hello wo')
            - Allow.OBJ: Allow partial objects (e.g., '{"key":' -> {'key': None})
            - Allow.ARR: Allow partial arrays (e.g., '[1, 2,' -> [1, 2])
            - Allow.ALL: Allow all types of partial data

    Returns:
        Tuple[Any, int]: A tuple containing:
            - parsed_object: The Python object parsed from the JSON
            - consumed_length: Number of characters consumed from input_str
    """
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except (JSONDecodeError, IndexError) as e:
        msg = getattr(e, "msg", str(e))
        if "Extra data" in msg or "pop from empty list" in msg:
            start = WHITESPACE.match(input_str, 0).end()
            obj, end = JSONDecoder().raw_decode(input_str, start)
            return obj, end
        raise
    except AssertionError as e:
        # partial_json_parser.fix_fast() asserts on some partial/ambiguous inputs
        # (e.g. trailing non-whitespace after an otherwise-fixable prefix) instead
        # of signaling "incomplete". Convert to JSONDecodeError so streaming
        # callers treat it as not-yet-complete (wait for more tokens) rather than
        # raising and failing the request.
        raise JSONDecodeError(
            "partial_json_parser assertion (treat as incomplete)", input_str, 0
        ) from e


def _is_complete_json(input_str: str) -> bool:
    try:
        orjson.loads(input_str)
        return True
    except JSONDecodeError:
        return False


# ``warnings.catch_warnings`` mutates the *process-global* warning filters and
# is therefore not thread-safe (CPython docs). Tool-call parsing runs on the
# request path and may execute concurrently, so the enter/eval/restore window
# is serialized. These helpers are microsecond-cheap; the lock has no perf impact.
_safe_ast_lock = threading.Lock()


def _run_ast_quiet(fn, *args):
    """Run an ``ast`` function with invalid-escape warnings suppressed.

    CPython parses invalid escapes (e.g. ``"\\d+"``) with the backslash kept
    and only emits a warning, so the parsed value is already correct —
    promoting the warning to an error would drop otherwise-valid tool calls.

    Holds ``_safe_ast_lock`` because ``catch_warnings`` touches global state."""
    with _safe_ast_lock, warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return fn(*args)


def safe_literal_eval(value: str) -> Any:
    return _run_ast_quiet(ast.literal_eval, value)


def safe_ast_parse(source: str) -> ast.Module:
    return _run_ast_quiet(ast.parse, source)


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


def _get_tool_schema_defs(tools: List[Tool]) -> dict:
    """
    Get consolidated $defs from all tools, validating for conflicts.

    Args:
        tools: List of tools to process

    Returns:
        Dictionary of consolidated $defs from all tools

    Raises:
        ValueError: If conflicting $defs are found
    """
    all_defs = {}
    for tool in tools:
        if tool.function.parameters is None:
            continue
        defs = tool.function.parameters.get("$defs", {})
        for def_name, def_schema in defs.items():
            if def_name in all_defs and all_defs[def_name] != def_schema:
                raise ValueError(
                    f"Tool definition '{def_name}' has "
                    "multiple schemas, which is not "
                    "supported."
                )
            else:
                all_defs[def_name] = def_schema
    return all_defs


def _get_tool_schema(tool: Tool) -> dict:
    return {
        "properties": {
            "name": {"type": "string", "enum": [tool.function.name]},
            "parameters": (
                tool.function.parameters
                if tool.function.parameters
                else {"type": "object", "properties": {}}
            ),
        },
        "required": ["name", "parameters"],
    }


def infer_type_from_json_schema(schema: Dict[str, Any]) -> Optional[str]:
    """
    Infer the primary type of a parameter from JSON Schema.

    Supports complex JSON Schema structures including:
    - Direct type field (including type arrays)
    - anyOf/oneOf: parameter can be any of multiple types
    - enum: parameter must be one of enum values
    - allOf: parameter must satisfy all type definitions
    - properties: inferred as object type
    - items: inferred as array type

    Args:
        schema: JSON Schema definition

    Returns:
        Inferred type ('string', 'number', 'object', 'array', etc.) or None
    """
    if not isinstance(schema, dict):
        return None

    # Priority 1: Direct type field (including type arrays)
    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            return type_value
        elif isinstance(type_value, list) and type_value:
            # Handle type arrays: return first non-null type
            non_null_types = [t for t in type_value if t != "null"]
            if non_null_types:
                return non_null_types[0]
            return "string"  # If only null, default to string

    # Priority 2: Handle anyOf/oneOf
    if "anyOf" in schema or "oneOf" in schema:
        schemas = schema.get("anyOf") or schema.get("oneOf")
        types = []

        if isinstance(schemas, list):
            for sub_schema in schemas:
                inferred_type = infer_type_from_json_schema(sub_schema)
                if inferred_type:
                    types.append(inferred_type)

            if types:
                # If all types are the same, return unified type
                if len(set(types)) == 1:
                    return types[0]
                # If it's an optional type, return original type.
                if len(set(types)) == 2 and "null" in types:
                    return [t for t in types if t != "null"][0]
                # When types differ, prioritize string (safest)
                if "string" in types:
                    return "string"
                # Otherwise return first type
                return types[0]

    # Priority 3: Handle enum (infer type from enum values)
    if "enum" in schema and isinstance(schema["enum"], list):
        if not schema["enum"]:
            return "string"

        # Infer type from enum values
        enum_types = set()
        for value in schema["enum"]:
            if value is None:
                enum_types.add("null")
            elif isinstance(value, bool):
                enum_types.add("boolean")
            elif isinstance(value, int):
                enum_types.add("integer")
            elif isinstance(value, float):
                enum_types.add("number")
            elif isinstance(value, str):
                enum_types.add("string")
            elif isinstance(value, list):
                enum_types.add("array")
            elif isinstance(value, dict):
                enum_types.add("object")

        # If type is uniform, return that type
        if len(enum_types) == 1:
            return enum_types.pop()
        # Mixed types, prioritize string
        return "string"

    # Priority 4: Handle allOf (must satisfy all types)
    if "allOf" in schema and isinstance(schema["allOf"], list):
        schemas = schema["allOf"]
        for sub_schema in schemas:
            inferred_type = infer_type_from_json_schema(sub_schema)
            if inferred_type and inferred_type != "string":
                return inferred_type
        return "string"

    # Priority 5: Infer object type
    if "properties" in schema:
        return "object"

    # Priority 6: Infer array type
    if "items" in schema:
        return "array"

    return None


def get_json_schema_constraint(
    tools: List[Tool],
    tool_choice: Union[ToolChoice, Literal["required"]],
    parallel_tool_calls: bool = True,
) -> Optional[dict]:
    """
    Get the JSON schema constraint for the specified tool choice.

    Args:
        tool_choice: The tool choice specification
        parallel_tool_calls: If False, constrain to exactly one tool call (maxItems=1)

    Returns:
        JSON schema dict, or None if no valid tools found
    """

    if isinstance(tool_choice, ToolChoice):
        # For specific function choice, return the user's parameters schema directly
        fn_name = tool_choice.function.name
        for tool in tools:
            if tool.function.name == fn_name:
                schema = {
                    "type": "array",
                    "minItems": 1,
                    "items": _get_tool_schema(tool),
                }
                if not parallel_tool_calls:
                    schema["maxItems"] = 1
                return schema
        return None
    elif tool_choice == "required":
        json_schema = {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "anyOf": [_get_tool_schema(tool) for tool in tools],
            },
        }
        if not parallel_tool_calls:
            json_schema["maxItems"] = 1
        json_schema_defs = _get_tool_schema_defs(tools)
        if json_schema_defs:
            json_schema["$defs"] = json_schema_defs
        return json_schema

    return None
