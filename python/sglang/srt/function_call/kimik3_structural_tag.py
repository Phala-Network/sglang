import copy
import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    AnyTokensFormat,
    ConstStringFormat,
    ExcludeTokenFormat,
    Format,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    RegexFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TokenFormat,
    TriggeredTagsFormat,
)

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.constrained.json_schema_intersections import (
    intersect_ref_min_length as _intersect_ref_min_length,
)
from sglang.srt.constrained.json_schema_validation import (
    _iter_subschemas,
    resolve_local_json_pointer as _resolve_local_ref,
    normalize_json_schema_resources,
)
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    CALL_OPEN,
    MESSAGE_CLOSE,
    THINK_CLOSE,
    THINK_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)

_JSON_TYPES = (
    "string",
    "number",
    "integer",
    "boolean",
    "array",
    "object",
    "null",
)
_CLOSE_TOKEN = "<|close|>"
_ARGUMENT_CLOSE_SUFFIX = ARGUMENT_CLOSE.removeprefix(_CLOSE_TOKEN)
_JSON_TO_XTML_TYPE = {
    "string": "string",
    "number": "number",
    "integer": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
    "null": "null",
}
_STRING_KEYWORDS = {"format", "maxLength", "minLength", "pattern"}
_NUMBER_KEYWORDS = {
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maximum",
    "minimum",
    "multipleOf",
}
_ARRAY_KEYWORDS = {
    "contains",
    "items",
    "maxContains",
    "maxItems",
    "minContains",
    "minItems",
    "prefixItems",
    "uniqueItems",
}
_OBJECT_KEYWORDS = {
    "additionalProperties",
    "dependentRequired",
    "dependentSchemas",
    "maxProperties",
    "minProperties",
    "patternProperties",
    "properties",
    "propertyNames",
    "required",
}
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
_ROOT_SCHEMA_KEYWORDS = _SCHEMA_METADATA_KEYWORDS | {
    "minProperties",
    "maxProperties",
    "dependentRequired",
    "additionalProperties",
    "properties",
    "required",
    "type",
}


def _escape_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;")


def _json_type(value: Any) -> str:
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
    return "object"


def _matches_json_type(value: Any, json_type: str) -> bool:
    value_type = _json_type(value)
    return value_type == json_type or (
        json_type == "number" and value_type == "integer"
    )



def _dereference_schema(
    schema: Union[bool, Dict[str, Any]], root_schema: Dict[str, Any]
) -> Union[bool, Dict[str, Any]]:
    """Resolve refs and the proven minLength intersection without losing assertions."""
    seen_refs: Set[str] = set()
    minimum = None
    while isinstance(schema, dict) and "$ref" in schema:
        unsupported = set(schema).difference(_SCHEMA_METADATA_KEYWORDS, {"$ref", "minLength"})
        if "minLength" in schema:
            bound = schema["minLength"]
            if type(bound) is not int or bound < 0:
                raise ValueError("Kimi K3 string lengths must be nonnegative integers")
            minimum = max(minimum or 0, bound)
        if unsupported:
            raise ValueError(
                f"Kimi K3 schema has unsupported $ref siblings: {sorted(unsupported)!r}"
            )
        ref = schema["$ref"]
        if not isinstance(ref, str):
            raise ValueError("Kimi K3 schema $ref must be a local JSON pointer")
        if ref in seen_refs:
            raise ValueError(f"Kimi K3 schema has a cyclic $ref: {ref!r}")
        target = _resolve_local_ref(ref, root_schema)
        if target is None:
            raise ValueError(f"Kimi K3 schema $ref is unresolved or nonlocal: {ref!r}")
        seen_refs.add(ref)
        schema = target
    return schema if minimum is None else _intersect_ref_min_length(schema, minimum)


def _schema_types(
    schema: Union[bool, Dict[str, Any]],
    root_schema: Dict[str, Any],
    seen_refs: Optional[Set[str]] = None,
) -> List[str]:
    if schema is False:
        return []
    if schema is True:
        return list(_JSON_TYPES)

    ref = schema.get("$ref")
    if isinstance(ref, str):
        seen_refs = set() if seen_refs is None else set(seen_refs)
        if ref not in seen_refs:
            target = _resolve_local_ref(ref, root_schema)
            if target is not None:
                seen_refs.add(ref)
                return _schema_types(target, root_schema, seen_refs)

    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return [schema_type] if schema_type in _JSON_TYPES else list(_JSON_TYPES)
    if isinstance(schema_type, list):
        return [item for item in _JSON_TYPES if item in schema_type]

    for keyword in ("anyOf", "oneOf"):
        options = schema.get(keyword)
        if isinstance(options, list):
            option_types = {
                item
                for option in options
                if isinstance(option, (bool, dict))
                for item in _schema_types(option, root_schema, seen_refs)
            }
            return [item for item in _JSON_TYPES if item in option_types]

    options = schema.get("allOf")
    if isinstance(options, list):
        type_sets = [
            set(_schema_types(option, root_schema, seen_refs))
            for option in options
            if isinstance(option, (bool, dict))
        ]
        constrained = [types for types in type_sets if types != set(_JSON_TYPES)]
        if constrained:
            result = constrained[0] | (
                {"integer"} if "number" in constrained[0] else set()
            )
            for types in constrained[1:]:
                result &= types | ({"integer"} if "number" in types else set())
            # number survives the intersection only if every branch allows it,
            # making the widened integer redundant rather than the other way.
            if "number" in result and "integer" in result:
                result.remove("integer")
            return [item for item in _JSON_TYPES if item in result]

    if "const" in schema:
        return [_json_type(schema["const"])]
    enum = schema.get("enum")
    if isinstance(enum, list):
        enum_types = {_json_type(value) for value in enum}
        return [item for item in _JSON_TYPES if item in enum_types]
    if _OBJECT_KEYWORDS.intersection(schema):
        return ["object"]
    if _ARRAY_KEYWORDS.intersection(schema):
        return ["array"]
    if _STRING_KEYWORDS.intersection(schema):
        return ["string"]
    if _NUMBER_KEYWORDS.intersection(schema):
        return ["number"]
    return list(_JSON_TYPES)


def _with_root_definitions(
    schema: Union[bool, Dict[str, Any]], root_schema: Dict[str, Any]
) -> Union[bool, Dict[str, Any]]:
    if not isinstance(schema, dict):
        return schema
    # XTML extracts an argument into a new schema document. Preserve the
    # original pointer scope by copying only targets reachable from this value.
    # Instance-valued const/enum/default data must never be rewritten.
    if not any("$ref" in node for node, _ in _iter_subschemas(schema, include_definitions=False)):
        return schema
    result = copy.deepcopy(schema)
    definitions = result.setdefault("$defs", {})
    targets: Dict[int, str] = {}
    rebound_nodes: Set[int] = set()
    pending = [result]
    next_id = 0
    while pending:
        current = pending.pop()
        # Snapshot: adding target definitions while visiting must not mutate
        # the schema-map iterator, or revisit newly rewritten pointers.
        nodes = list(_iter_subschemas(current, include_definitions=False))
        for node, _ in nodes:
            if "$ref" not in node:
                continue
            # Union type restriction may share a target's nested dictionaries.
            # deepcopy preserves those aliases; a rewritten pointer belongs to
            # this extracted document and must not be resolved in the old root.
            if id(node) in rebound_nodes:
                continue
            rebound_nodes.add(id(node))
            ref = node["$ref"]
            target = _resolve_local_ref(ref, root_schema) if isinstance(ref, str) else None
            if target is None:
                raise ValueError("Kimi K3 schema $ref is unresolved or nonlocal")
            identity = id(target)
            if identity not in targets:
                while f"__k3_ref_{next_id}" in definitions:
                    next_id += 1
                name = f"__k3_ref_{next_id}"
                next_id += 1
                targets[identity] = name
                cloned = copy.deepcopy(target)
                definitions[name] = cloned
                pending.append(cloned)
            node["$ref"] = "#/$defs/" + targets[identity]
    return result


def _restrict_schema_type(
    schema: Union[bool, Dict[str, Any]],
    json_type: str,
    root_schema: Dict[str, Any],
) -> Union[bool, Dict[str, Any]]:
    schema = _dereference_schema(schema, root_schema)
    if not isinstance(schema, dict):
        return {"type": json_type} if schema else False

    result = dict(schema)
    schema_type = result.get("type")
    if isinstance(schema_type, list):
        if json_type not in schema_type:
            return False
        result["type"] = json_type
    elif isinstance(schema_type, str):
        if schema_type != json_type:
            return False
    else:
        result["type"] = json_type

    for keyword in ("anyOf", "oneOf"):
        options = result.get(keyword)
        if not isinstance(options, list):
            continue
        restricted = [
            _restrict_schema_type(option, json_type, root_schema)
            for option in options
            if isinstance(option, (bool, dict))
            and json_type in _schema_types(option, root_schema)
        ]
        if not restricted:
            return False
        if len(restricted) == 1 and isinstance(restricted[0], dict):
            result.pop(keyword)
            result.update(restricted[0])
        else:
            result[keyword] = restricted

    enum = result.get("enum")
    if isinstance(enum, list):
        result["enum"] = [
            value for value in enum if _matches_json_type(value, json_type)
        ]
        if not result["enum"]:
            return False
    if "const" in result and not _matches_json_type(result["const"], json_type):
        return False
    return result


_STRING_SHAPE_KEYWORDS = _STRING_KEYWORDS | {
    "$ref",
    "allOf",
    "anyOf",
    "const",
    "enum",
    "not",
    "oneOf",
}


def _is_free_text(schema: Union[bool, Dict[str, Any]]) -> bool:
    """A string schema that constrains nothing about the text itself."""
    return schema is True or not _STRING_SHAPE_KEYWORDS.intersection(schema)


def _value_format(
    schema: Union[bool, Dict[str, Any]],
    json_type: str,
) -> Format:
    # XGrammar 0.2.1 miscompiles a one-sided negative integer lower bound:
    # {"type": "integer", "minimum": -N} accepts the incomplete value "-"
    # and rejects every valid negative integer. Splitting the range at zero
    # avoids that converter bug without weakening the schema.
    if (
        json_type == "integer"
        and isinstance(schema, dict)
        and "maximum" not in schema
        and "exclusiveMaximum" not in schema
        and "multipleOf" not in schema
    ):
        lower_bounds = []
        minimum = schema.get("minimum")
        if isinstance(minimum, int) and not isinstance(minimum, bool):
            lower_bounds.append(minimum)
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(exclusive_minimum, int) and not isinstance(
            exclusive_minimum, bool
        ):
            lower_bounds.append(exclusive_minimum + 1)
        if lower_bounds and max(lower_bounds) < 0:
            negative = dict(schema)
            negative["maximum"] = -1
            nonnegative = dict(schema)
            nonnegative.pop("exclusiveMinimum", None)
            nonnegative["minimum"] = 0
            definitions = schema.get("$defs")
            schema = {"anyOf": [negative, nonnegative]}
            if definitions is not None:
                schema["$defs"] = definitions
    return JSONSchemaFormat(json_schema=schema, style="json", property_order="unordered")


def _argument_value_variants(
    schema: Union[bool, Dict[str, Any]],
    root_schema: Dict[str, Any],
    loose_strings: bool = False,
) -> List[Tuple[str, str, Format]]:
    schema = _dereference_schema(schema, root_schema)
    variants = []
    for json_type in _schema_types(schema, root_schema):
        restricted = _restrict_schema_type(schema, json_type, root_schema)
        if restricted is False:
            continue
        restricted = _with_root_definitions(restricted, root_schema)
        if json_type == "string":
            if loose_strings or _is_free_text(restricted):
                # Raw strings remain delimiter-framed. JSON strings provide an
                # unambiguous alternative for every value, including the end tag.
                variants.append((json_type, "string", AnyTextFormat()))
            variants.append(
                (json_type, "json-string", _value_format(restricted, json_type))
            )
        else:
            variants.append(
                (
                    json_type,
                    _JSON_TO_XTML_TYPE[json_type],
                    _value_format(restricted, json_type),
                )
            )
    return variants


def _known_argument_format(
    key: str,
    schema: Union[bool, Dict[str, Any]],
    root_schema: Dict[str, Any],
) -> Optional[Format]:
    escaped_key = _escape_attr(key)
    variants = [
        TagFormat(
            begin=(f'<|open|>argument key="{escaped_key}" type="{wire_type}"<|sep|>'),
            content=value_format,
            end=ARGUMENT_CLOSE,
        )
        for _json_type, wire_type, value_format in _argument_value_variants(
            schema, root_schema
        )
    ]
    if not variants:
        return None
    if len(variants) == 1:
        return variants[0]
    return OrFormat(elements=variants)


# Regex text of the characters an argument key may not contain, and the same
# characters as a Python set.
_KEY_FORBIDDEN_CHARS = r'"& \t\r\n\f\v=<>'
_KEY_FORBIDDEN_SET = set('"& \t\r\n\f\v=<>')


def _regex_literal(text: str) -> str:
    """Escape ``text`` for xgrammar's regex dialect, inside or outside a class."""
    return "".join(
        char
        if char.isascii() and (char.isalnum() or char == "_")
        else f"\\{char}"
        if char.isascii()
        else f"\\u{ord(char):04x}"
        if ord(char) <= 0xFFFF
        else f"\\U{ord(char):08x}"
        for char in text
    )


def _dynamic_key_pattern(excluded_keys: Iterable[str]) -> str:
    """Regex for an argument key that is none of ``excluded_keys``.

    A strict tool pins each declared property to its own schema, so the
    additionalProperties branch must not accept a declared name under the
    looser schema (JSON Schema applies additionalProperties only to keys that
    ``properties`` does not name). xgrammar's regex dialect has no lookahead,
    so the exclusion walks a trie of the names: a key is acceptable when it
    leaves the trie with a character no excluded name continues with, or when
    it stops at a proper prefix of an excluded name.
    """
    any_key_char = f"[^{_KEY_FORBIDDEN_CHARS}]"
    names = {key for key in excluded_keys if key and not set(key) & _KEY_FORBIDDEN_SET}
    if not names:
        return any_key_char + "+"

    trie: Dict[str, Any] = {}
    for name in names:
        node = trie
        for char in name:
            node = node.setdefault(char, {})
        node[""] = {}

    alternatives: List[str] = []

    def walk(node: Dict[str, Any], prefix: str) -> None:
        children = [char for char in node if char]
        diverge = f"[^{_KEY_FORBIDDEN_CHARS}{''.join(map(_regex_literal, children))}]"
        alternatives.append(f"{_regex_literal(prefix)}{diverge}{any_key_char}*")
        if prefix and "" not in node:
            alternatives.append(_regex_literal(prefix))
        for char in children:
            walk(node[char], prefix + char)

    walk(trie, "")
    return "(?:" + "|".join(alternatives) + ")"


def _dynamic_argument_format(
    schema: Union[bool, Dict[str, Any]],
    root_schema: Dict[str, Any],
    loose_strings: bool = False,
    excluded_keys: Iterable[str] = (),
) -> Format:
    variants = [
        SequenceFormat(
            elements=[
                RegexFormat(pattern=_dynamic_key_pattern(excluded_keys)),
                ConstStringFormat(value=(f'" type="{wire_type}"<|sep|>')),
                value_format,
            ]
        )
        for _json_type, wire_type, value_format in _argument_value_variants(
            schema, root_schema, loose_strings=loose_strings
        )
    ]
    if not variants:
        raise ValueError("Kimi K3 additional parameter schema accepts no values")
    content = variants[0] if len(variants) == 1 else OrFormat(elements=variants)
    return TagFormat(
        begin='<|open|>argument key="',
        content=content,
        end=ARGUMENT_CLOSE,
    )



class _EmptyRootLanguage(ValueError):
    pass


def _finite_constrained_arguments_format(
    parameters: Dict[str, Any], root_schema: Dict[str, Any],
    presence_mode: str = "none", presence_names=(), prove_values: bool = False,
    verified_values=None,
) -> Format:
    # Native capability, not package-version text or a Python model field alone:
    # older native parsers ignore unknown JSON fields and would widen the grammar.
    from xgrammar.base import _core

    version = "V2" if presence_mode != "none" else "V1"
    capability = getattr(_core, "SupportsUnorderedConstraints" + version, None)
    fields = getattr(SequenceFormat, "model_fields", {})
    if presence_mode != "none" and not {"presence_mode", "presence_indices"}.issubset(fields):
        raise ValueError("Kimi K3 root presence constraints require native unordered constraints v2")
    if not callable(capability) or not capability() or not {
        "min_elements", "max_elements", "dependencies"
    }.issubset(fields):
        raise ValueError("Kimi K3 finite object constraints require native unordered constraints " + version.lower())
    if parameters.get("additionalProperties", True) is not False:
        raise ValueError("Kimi K3 finite object constraints require additionalProperties=false")
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    if not isinstance(properties, dict):
        raise ValueError("Kimi K3 tool parameters 'properties' must be an object")
    if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
        raise ValueError("Kimi K3 tool parameters 'required' must be a string list")
    dependencies = parameters.get("dependentRequired", {})
    if not isinstance(dependencies, dict) or not all(
        isinstance(key, str) and isinstance(value, list)
        and all(isinstance(target, str) for target in value)
        for key, value in dependencies.items()
    ):
        raise ValueError("Kimi K3 dependentRequired must map names to string lists")
    minimum = parameters.get("minProperties", 0)
    maximum = parameters.get("maxProperties", len(properties))
    if any(type(bound) is not int or bound < 0 for bound in (minimum, maximum)):
        raise ValueError("Kimi K3 property counts must be nonnegative integers")
    if prove_values and (len(properties) > 64 or sum(len(value) for value in dependencies.values()) > 4096):
        raise ValueError("presence_resource_limit: at most 64 properties and 4096 dependency edges")
    witnesses = {}
    proven_values = {}
    if prove_values:
        from sglang.srt.function_call.kimik3_root_constraints import EMPTY, prove_value, verify_native_witnesses
    arguments = {}
    for key, schema in properties.items():
        if not isinstance(key, str) or not isinstance(schema, (dict, bool)):
            raise ValueError("Kimi K3 tool property schemas must be JSON schemas")
        if prove_values:
            schema, witness = prove_value(schema, root_schema)
            if witness is not EMPTY:
                witnesses[key] = witness
                proven_values[key] = schema
        argument = _known_argument_format(key, schema, root_schema)
        if argument is not None:
            arguments[key] = argument
    # A trigger requiring an absent/false property can never appear. Propagate
    # that fact through chains before constructing the finite native entries.
    changed = True
    while changed:
        changed = False
        for key in list(arguments):
            if any(target not in arguments for target in dependencies.get(key, [])):
                del arguments[key]
                changed = True
    required_set = set(required)
    pending = list(required_set)
    while pending:
        key = pending.pop()
        if key not in arguments:
            raise _EmptyRootLanguage("Kimi K3 required/dependent parameter accepts no values")
        for target in dependencies.get(key, []):
            if target not in required_set:
                required_set.add(target)
                pending.append(target)
    if presence_mode == "not_all" and any(key not in arguments for key in presence_names):
        presence_mode, presence_names = "none", ()
    elif presence_mode == "exactly_one":
        presence_names = [key for key in presence_names if key in arguments]
        if not presence_names:
            raise _EmptyRootLanguage("Kimi K3 presence constraint accepts no values")
    maximum = min(maximum, len(arguments))
    if presence_mode != "none" and maximum < len(arguments):
        raise ValueError("Kimi K3 presence constraints with a constraining maxProperties are unsupported")
    if minimum > maximum or len(required_set) > maximum:
        raise _EmptyRootLanguage("Kimi K3 finite object constraints accept no values")
    if maximum == 0:
        return ConstStringFormat(value="")
    if prove_values:
        verify_native_witnesses(
            arguments, {key: witnesses[key] for key in arguments},
            proven_values, verified_values,
        )
    indices = {key: index for index, key in enumerate(arguments)}
    edges = sorted({
        (indices[key], indices[target])
        for key in arguments for target in dependencies.get(key, []) if target != key
    })
    if edges and maximum < len(arguments):
        raise ValueError("Kimi K3 dependencies with a constraining maxProperties are unsupported")
    return SequenceFormat(
        elements=list(arguments.values()), unordered=True,
        required=[key in required_set for key in arguments],
        min_elements=minimum, max_elements=maximum, dependencies=edges,
        **({"presence_mode": presence_mode,
            "presence_indices": [indices[key] for key in presence_names]} if presence_mode != "none" else {}),
    )


def _strict_arguments_format(
    parameters: Dict[str, Any], root_schema: Dict[str, Any]
) -> Format:
    if {"minProperties", "maxProperties", "dependentRequired"}.intersection(parameters):
        return _finite_constrained_arguments_format(parameters, root_schema)
    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("Kimi K3 tool parameters 'properties' must be an object")
    required = parameters.get("required", [])
    if not isinstance(required, list) or not all(
        isinstance(item, str) for item in required
    ):
        raise ValueError("Kimi K3 tool parameters 'required' must be a string list")

    required_set = set(required)
    missing = required_set.difference(properties)
    if missing:
        raise ValueError(
            f"Kimi K3 required parameters are missing schemas: {sorted(missing)!r}"
        )

    elements: List[Format] = []
    unordered = parameters.get("additionalProperties", True) is False
    required_flags: List[bool] = []
    for key, schema in properties.items():
        if not isinstance(key, str) or not isinstance(schema, (bool, dict)):
            raise ValueError("Kimi K3 tool property schemas must be JSON schemas")
        argument = _known_argument_format(key, schema, root_schema)
        if argument is None:
            if key in required_set:
                raise ValueError(
                    f"Kimi K3 required parameter {key!r} accepts no values"
                )
            continue
        elements.append(
            argument if unordered or key in required_set else OptionalFormat(content=argument)
        )
        required_flags.append(key in required_set)

    additional = parameters.get("additionalProperties", True)
    declared = set(properties)
    if additional is True:
        elements.append(
            StarFormat(
                content=_dynamic_argument_format(
                    True, root_schema, excluded_keys=declared
                )
            )
        )
    elif isinstance(additional, dict):
        elements.append(
            StarFormat(
                content=_dynamic_argument_format(
                    additional, root_schema, excluded_keys=declared
                )
            )
        )
    elif additional is not False:
        raise ValueError(
            "Kimi K3 tool parameters 'additionalProperties' must be a schema"
        )
    if not elements:
        return ConstStringFormat(value="")
    return SequenceFormat(elements=elements, unordered=unordered, required=required_flags if unordered else [])



def _checked_root_arguments_format(parameters, root_schema):
    if not isinstance(parameters, dict):
        if parameters is False:
            raise _EmptyRootLanguage("Kimi K3 root schema accepts no values")
        raise ValueError("Kimi K3 tool parameters must be an object schema")
    if "not" in parameters or "oneOf" in parameters:
        return _root_logic_arguments_format(parameters, root_schema)
    unsupported = set(parameters).difference(_ROOT_SCHEMA_KEYWORDS)
    if unsupported:
        raise ValueError("Kimi K3 tool parameters have unsupported root schema keywords: " + repr(sorted(unsupported)))
    if "object" not in _schema_types(parameters, root_schema):
        raise ValueError("Kimi K3 tool parameters must be an object schema")
    return _strict_arguments_format(parameters, root_schema)


def _presence_required(schema):
    if not isinstance(schema, dict):
        return None
    active = set(schema).difference(_SCHEMA_METADATA_KEYWORDS)
    if active.difference({"type", "required"}) or ("type" in schema and schema["type"] != "object"):
        return None
    names = schema.get("required", [])
    if not isinstance(names, list) or not all(isinstance(key, str) for key in names):
        raise ValueError("Kimi K3 presence required must be a string list")
    if len(names) != len(set(names)):
        raise ValueError("Kimi K3 presence required names must be unique")
    if len(names) > 64:
        raise ValueError("presence_resource_limit: at most 64 group keys")
    return names


def _root_logic_arguments_format(parameters, root_schema):
    if "not" in parameters and "oneOf" in parameters:
        raise ValueError("Kimi K3 multiple root logic groups are unsupported")
    if "object" not in _schema_types(parameters, root_schema):
        raise ValueError("Kimi K3 tool parameters must describe objects")
    if "not" in parameters:
        negated = parameters["not"]
        base = {key: value for key, value in parameters.items() if key != "not"}
        if negated is False:
            return _checked_root_arguments_format(base, root_schema)
        if negated is True:
            raise _EmptyRootLanguage("Kimi K3 not:true accepts no object values")
        if isinstance(negated, dict) and not set(negated).difference(_SCHEMA_METADATA_KEYWORDS, {"type"}):
            types = negated.get("type", list(_JSON_TYPES))
            types = [types] if isinstance(types, str) else types
            if not isinstance(types, list) or not types or any(kind not in _JSON_TYPES for kind in types):
                raise ValueError("Kimi K3 not type is invalid")
            if "object" in types:
                raise _EmptyRootLanguage("Kimi K3 not object type accepts no object values")
            return _checked_root_arguments_format(base, root_schema)
        names = _presence_required(negated)
        if names is None:
            raise ValueError("Kimi K3 root not shape is unsupported")
        if not names:
            raise _EmptyRootLanguage("Kimi K3 not required-empty accepts no values")
        if set(base).difference(_ROOT_SCHEMA_KEYWORDS):
            raise ValueError("Kimi K3 root presence has unsupported sibling assertions")
        return _finite_constrained_arguments_format(base, root_schema, "not_all", names, True)
    branches = parameters["oneOf"]
    if not isinstance(branches, list) or not branches:
        raise ValueError("Kimi K3 oneOf must have branches")
    if len(branches) > 64:
        raise ValueError("presence_resource_limit: at most 64 singleton branches")
    names = [_presence_required(branch) for branch in branches]
    if all(group is not None and len(group) == 1 for group in names):
        group = [item[0] for item in names]
        if len(group) != len(set(group)):
            raise ValueError("Kimi K3 singleton oneOf keys must be unique")
        base = {key: value for key, value in parameters.items() if key != "oneOf"}
        if set(base).difference(_ROOT_SCHEMA_KEYWORDS):
            raise ValueError("Kimi K3 root presence has unsupported sibling assertions")
        return _finite_constrained_arguments_format(base, root_schema, "exactly_one", group, True)
    return _disjoint_root_branches_format(parameters, root_schema)


def _disjoint_root_branches_format(parameters, root_schema):
    from sglang.srt.function_call.kimik3_root_constraints import EMPTY, prove_value
    active = set(parameters).difference(_SCHEMA_METADATA_KEYWORDS, {"oneOf"})
    if active and not (active == {"type"} and parameters["type"] == "object"):
        raise ValueError("Kimi K3 oneOf parent assertions require an unsupported intersection")
    branches = [_dereference_schema(branch, root_schema) for branch in parameters["oneOf"]]
    if len(branches) > 8:
        raise ValueError("root_branch_resource_limit: at most 8 closed object branches")
    total = 0
    formats, viable = [], []
    verified_values = set()
    for branch in branches:
        if not isinstance(branch, dict) or branch.get("type") != "object" or branch.get("additionalProperties") is not False:
            raise ValueError("Kimi K3 oneOf branches must be complete closed objects")
        if set(branch).difference(_ROOT_SCHEMA_KEYWORDS):
            raise ValueError("Kimi K3 oneOf branch has unsupported root assertions")
        if not isinstance(branch.get("properties", {}), dict):
            raise ValueError("Kimi K3 oneOf branch properties must be an object")
        total += len(branch.get("properties", {}))
        if total > 512:
            raise ValueError("root_branch_resource_limit: at most 512 total branch properties")
        try:
            fmt = _finite_constrained_arguments_format(
                branch, root_schema, prove_values=True, verified_values=verified_values
            )
        except _EmptyRootLanguage:
            continue
        viable.append(branch)
        formats.append(fmt)
    if not viable:
        raise _EmptyRootLanguage("Kimi K3 oneOf branches accept no values")
    if len(viable) == 1:
        return formats[0]
    common = set.intersection(*(set(branch.get("required", [])) & set(branch.get("properties", {})) for branch in viable))
    for key in sorted(common):
        seen, disjoint = set(), True
        for branch in viable:
            proven, witness = prove_value(branch["properties"][key], root_schema)
            domain = proven.get("enum") if isinstance(proven, dict) else None
            if witness is EMPTY or not isinstance(domain, list) or not domain or not all(isinstance(value, str) for value in domain):
                disjoint = False
                break
            values = set(domain)
            if seen.intersection(values):
                disjoint = False
                break
            seen.update(values)
        if disjoint:
            return OrFormat(elements=formats)
    raise ValueError("Kimi K3 oneOf requires a common required disjoint string discriminator")


def _tool_arguments_format(tool: Tool) -> Format:
    parameters = tool.function.parameters
    if not tool.function.strict:
        root_schema = parameters if isinstance(parameters, dict) else {}
        return StarFormat(
            content=_dynamic_argument_format(True, root_schema, loose_strings=True)
        )
    if parameters is None:
        # Server-side strict levels mark tools without parameters strict too;
        # they take no arguments rather than failing the whole constraint.
        return ConstStringFormat(value="")
    if not isinstance(parameters, dict):
        raise ValueError(
            f"Kimi K3 strict tool {tool.function.name!r} must define parameters"
        )
    root_schema = normalize_json_schema_resources(parameters)
    # Binding is complete; extracted documents must not reintroduce the root
    # ID as a nested resource when cloning recursive targets.
    root_schema.pop("$id", None)
    parameters = _dereference_schema(root_schema, root_schema)
    # A single allOf branch with annotation-only siblings is an exact wrapper.
    # Do not merge arbitrary intersections or overwrite sibling assertions.
    wrappers = set()
    while isinstance(parameters, dict) and "allOf" in parameters:
        options = parameters["allOf"]
        if id(options) in wrappers:
            raise ValueError("Kimi K3 tool parameters have a cyclic allOf wrapper")
        wrappers.add(id(options))
        assertions = set(parameters).difference(_SCHEMA_METADATA_KEYWORDS, {"$id", "allOf"})
        object_only = assertions == {"type"} and parameters["type"] == "object"
        if not isinstance(options, list) or len(options) != 1 or (assertions and not object_only):
            break
        branch = _dereference_schema(options[0], root_schema)
        if object_only:
            if "object" not in _schema_types(branch, root_schema):
                raise ValueError("Kimi K3 tool allOf cannot describe an object")
            if isinstance(branch, dict):
                branch = dict(branch, type="object")
        parameters = branch
    if isinstance(parameters, dict) and "$id" in parameters:
        # Local refs have already been bound and rewritten against their resource.
        parameters = {key: value for key, value in parameters.items() if key != "$id"}
    if parameters is True:
        parameters = {}
    return _checked_root_arguments_format(parameters, root_schema)



def _tool_call_tag(tool: Tool, arguments_format: Optional[Format] = None) -> TagFormat:
    name = _escape_attr(tool.function.name)
    if arguments_format is None:
        arguments_format = _tool_arguments_format(tool)
    return TagFormat(
        begin=f'{CALL_OPEN} tool="{name}" index="',
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[1-9][0-9]*"),
                ConstStringFormat(value='"<|sep|>'),
                arguments_format,
            ]
        ),
        end=CALL_CLOSE,
    )


def _tool_calls_tag(call_tags: List[TagFormat], parallel_tool_calls: bool) -> TagFormat:
    if parallel_tool_calls:
        content: Format = TagsWithSeparatorFormat(
            tags=call_tags, separator="", at_least_one=True
        )
    elif len(call_tags) == 1:
        content = call_tags[0]
    else:
        content = OrFormat(elements=call_tags)
    return TagFormat(begin=TOOLS_OPEN, content=content, end=TOOLS_CLOSE)


def _auto_suffix(
    tools_tag: TagFormat, parallel_tool_calls: bool
) -> TriggeredTagsFormat:
    # A retriggered second tools section would evade the single-call limit.
    return TriggeredTagsFormat(
        triggers=[TOOLS_OPEN],
        tags=[tools_tag],
        excludes=[THINK_OPEN, THINK_CLOSE, CALL_OPEN],
        stop_after_first=not parallel_tool_calls,
    )


def _with_reasoning(suffix: Format, thinking_mode: bool) -> Format:
    if not thinking_mode:
        return suffix
    return SequenceFormat(
        elements=[
            TagFormat(begin="", content=AnyTextFormat(), end=THINK_CLOSE),
            suffix,
        ]
    )


def _single_xtml_type(
    schema: Union[bool, Dict[str, Any]], root_schema: Dict[str, Any]
) -> Optional[str]:
    schema_types = _schema_types(schema, root_schema)
    if len(schema_types) != 1:
        return None
    return _JSON_TO_XTML_TYPE[schema_types[0]]


def _nonempty_argument_format(key: str, xtml_type: str) -> Format:
    # A token-based end keeps the first close token out of both content formats.
    argument = TagFormat(
        begin=(f'<|open|>argument key="{_escape_attr(key)}" type="{xtml_type}"<|sep|>'),
        content=SequenceFormat(elements=[ExcludeTokenFormat(), AnyTokensFormat()]),
        end=TokenFormat(token=_CLOSE_TOKEN),
    )
    return SequenceFormat(
        elements=[
            argument,
            ConstStringFormat(value=_ARGUMENT_CLOSE_SUFFIX),
        ]
    )


def _auto_tool_arguments_format(tool: Tool) -> Format:
    parameters = tool.function.parameters
    if not isinstance(parameters, dict):
        return AnyTextFormat()
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return AnyTextFormat()

    required_tags: List[Format] = []
    for key in required:
        schema = properties.get(key)
        if not isinstance(key, str) or not isinstance(schema, (bool, dict)):
            return AnyTextFormat()
        xtml_type = _single_xtml_type(schema, parameters)
        if xtml_type is None:
            return AnyTextFormat()
        required_tags.append(_nonempty_argument_format(key, xtml_type))

    if required_tags:
        elements = required_tags
    else:
        alternatives = [
            _nonempty_argument_format(key, xtml_type)
            for key, schema in properties.items()
            if isinstance(key, str)
            and isinstance(schema, (bool, dict))
            and (xtml_type := _single_xtml_type(schema, parameters)) is not None
        ]
        if not alternatives:
            return AnyTextFormat()
        elements = [
            (
                alternatives[0]
                if len(alternatives) == 1
                else OrFormat(elements=alternatives)
            )
        ]

    return SequenceFormat(
        elements=[*elements, AnyTextFormat(excludes=[CALL_OPEN, CALL_CLOSE])]
    )


def get_kimik3_auto_tool_call_structural_tag(
    tools: List[Tool],
    thinking_mode: bool = False,
    parallel_tool_calls: bool = True,
) -> Optional[StructuralTag]:
    if not tools:
        return None

    call_tags = [
        _tool_call_tag(tool, _auto_tool_arguments_format(tool)) for tool in tools
    ]
    suffix = _auto_suffix(
        _tool_calls_tag(call_tags, parallel_tool_calls), parallel_tool_calls
    )
    return StructuralTag(format=_with_reasoning(suffix, thinking_mode))


def _select_tools(
    tools: List[Tool],
    tool_choice: Union[ToolChoice, Literal["auto", "required"]],
) -> Tuple[List[Tool], bool]:
    if not isinstance(tool_choice, ToolChoice):
        return tools, tool_choice == "required"
    name = tool_choice.function.name
    selected = [tool for tool in tools if tool.function.name == name]
    if not selected:
        raise ValueError(f"Kimi K3 tool choice {name!r} is not in the tools list")
    return selected, True


def get_kimik3_structural_tag(
    tools: List[Tool],
    tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
    thinking_mode: bool = False,
    parallel_tool_calls: bool = True,
) -> StructuralTag:
    selected_tools, at_least_one = _select_tools(tools, tool_choice)
    if not selected_tools:
        raise ValueError("Kimi K3 structural tags require at least one tool")

    call_tags = [_tool_call_tag(tool) for tool in selected_tools]
    tools_tag = _tool_calls_tag(call_tags, parallel_tool_calls)
    if at_least_one:
        suffix: Format = SequenceFormat(
            elements=[
                AnyTextFormat(
                    excludes=[
                        TOOLS_OPEN,
                        TOOLS_CLOSE,
                        MESSAGE_CLOSE,
                        THINK_OPEN,
                        THINK_CLOSE,
                        CALL_OPEN,
                    ]
                ),
                tools_tag,
            ]
        )
    else:
        suffix = _auto_suffix(tools_tag, parallel_tool_calls)
    return StructuralTag(format=_with_reasoning(suffix, thinking_mode))
