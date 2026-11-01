"""Bounded, assertion-preserving value proof for K3 root logic fragments.

This gate is used only by new root logic. It does not change the V1 value path.
Unsupported forms are never treated as an empty language.
"""
import json
import math

from jsonschema import Draft202012Validator

from sglang.srt.constrained.json_schema_validation import validate_xgrammar_json_schema

EMPTY = object()
MAX_VALUE_NODES = 512
MAX_VALUE_DEPTH = 16
MAX_WITNESS_BYTES = 65536


def _finite_number(value):
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _error(reason):
    raise ValueError("Kimi K3 root value proof: " + reason)


def _count(schema, key, default):
    value = schema.get(key, default)
    if type(value) is not int or value < 0:
        _error(key + " must be a nonnegative integer")
    return value


def _disjoint_types(branches, root, k3):
    from sglang.srt.constrained.json_schema_intersections import _possible_types
    seen = set()
    for branch in branches:
        kinds = _possible_types(k3._dereference_schema(branch, root))
        if "number" in kinds:
            kinds.add("integer")
        if seen.intersection(kinds):
            _error("oneOf branch disjointness is unproven")
        seen.update(kinds)


def prove_value(schema, root):
    from sglang.srt.function_call import kimik3_structural_tag as k3
    validate_xgrammar_json_schema(k3._with_root_definitions(schema, root))
    budget = [MAX_VALUE_NODES]

    def visit(node, depth=0):
        budget[0] -= 1
        if budget[0] < 0 or depth > MAX_VALUE_DEPTH:
            _error("value_resource_limit: 512 nodes and depth 16")
        node = k3._dereference_schema(node, root)
        if node is False:
            return False, EMPTY
        if node is True:
            return True, None
        if not isinstance(node, dict):
            _error("value must be a schema")
        active = set(node).difference(k3._SCHEMA_METADATA_KEYWORDS)
        if not active:
            return True, None
        for logical in ("anyOf", "oneOf"):
            if logical not in node:
                continue
            branches = node[logical]
            if active != {logical} or not isinstance(branches, list) or not branches:
                _error("union siblings require an unsupported intersection")
            if logical == "oneOf":
                _disjoint_types(branches, root, k3)
            proven = [visit(branch, depth + 1) for branch in branches]
            proven = [(branch, value) for branch, value in proven if value is not EMPTY]
            if not proven:
                return False, EMPTY
            return {logical: [branch for branch, _ in proven]}, proven[0][1]
        allowed = {"type", "const", "enum", "minLength", "maxLength", "minimum", "maximum",
                   "exclusiveMinimum", "exclusiveMaximum", "items", "minItems", "maxItems",
                   "properties", "required", "additionalProperties"}
        if active.difference(allowed):
            _error("unsupported active value assertion")
        # Native const/enum dispatch ignores siblings. Evaluate the finite domain
        # first, then compile only that exact domain, preserving every assertion.
        if "const" in node or "enum" in node:
            domain = [node["const"]] if "const" in node else node["enum"]
            if not isinstance(domain, list) or len(domain) > 128:
                _error("value_resource_limit: finite domain at most 128 values")
            validator = Draft202012Validator(k3._with_root_definitions(node, root))
            domain = [value for value in domain if validator.is_valid(value)]
            if not domain:
                return False, EMPTY
            if len(json.dumps(domain, ensure_ascii=True)) > MAX_WITNESS_BYTES:
                _error("value_resource_limit: finite domain exceeds 65536 bytes")
            return {"enum": domain}, domain[0]
        kind = node.get("type")
        if kind is None or isinstance(kind, list):
            kinds = list(k3._JSON_TYPES) if kind is None else kind
            if not kinds or any(item not in k3._JSON_TYPES for item in kinds):
                _error("invalid type union")
            kinds = list(dict.fromkeys(kinds))
            if "number" in kinds and "integer" in kinds:
                kinds.remove("integer")
            proven = [visit(dict(node, type=item), depth + 1) for item in kinds]
            proven = [(branch, value) for branch, value in proven if value is not EMPTY]
            if not proven:
                return False, EMPTY
            return {"anyOf": [branch for branch, _ in proven]}, proven[0][1]
        if kind not in k3._JSON_TYPES:
            _error("invalid type")
        # Known type-specific keywords are vacuous on other JSON types.
        if kind == "string":
            lo = _count(node, "minLength", 0)
            hi = _count(node, "maxLength", 0) if "maxLength" in node else math.inf
            if lo > hi:
                return False, EMPTY
            if lo > 4096:
                _error("value_resource_limit: string witness exceeds 4096 characters")
            result = {key: node[key] for key in ("type", "minLength", "maxLength") if key in node}
            return result, "a" * lo
        if kind in ("integer", "number"):
            lower, upper = -math.inf, math.inf
            lower_open = upper_open = False
            for key, is_lower, exclusive in [("minimum", True, False), ("exclusiveMinimum", True, True),
                                               ("maximum", False, False), ("exclusiveMaximum", False, True)]:
                if key not in node:
                    continue
                bound = node[key]
                if type(bound) not in (int, float) or not _finite_number(bound):
                    _error("numeric bounds must be finite numbers")
                if is_lower and (bound > lower or (bound == lower and exclusive)):
                    lower, lower_open = bound, exclusive
                if not is_lower and (bound < upper or (bound == upper and exclusive)):
                    upper, upper_open = bound, exclusive
            if kind == "integer":
                if lower != -math.inf:
                    lower = math.floor(lower) + 1 if lower_open else math.ceil(lower)
                if upper != math.inf:
                    upper = math.ceil(upper) - 1 if upper_open else math.floor(upper)
                lower_open = upper_open = False
            if lower > upper or (lower == upper and (lower_open or upper_open)):
                return False, EMPTY
            # Construct a finite witness without midpoint overflow or assuming
            # that +1 changes a large floating point endpoint. Failure to find
            # a representable witness is unsupported, never proof of emptiness.
            candidates = [0]
            if lower != -math.inf:
                if not lower_open:
                    candidates.append(lower)
                else:
                    candidates.extend([lower + 1, math.nextafter(lower, math.inf)])
            if upper != math.inf:
                if not upper_open:
                    candidates.append(upper)
                else:
                    candidates.extend([upper - 1, math.nextafter(upper, -math.inf)])
            if kind == "number" and lower != -math.inf and upper != math.inf:
                candidates.append(lower / 2 + upper / 2)
            witness = next((value for value in candidates
                if _finite_number(value)
                and (value > lower if lower_open else value >= lower)
                and (value < upper if upper_open else value <= upper)
                and (kind == "number" or type(value) is int)), EMPTY)
            if witness is EMPTY:
                _error("numeric witness exceeds supported representable range")
            keys = {"type", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
            return {key: value for key, value in node.items() if key in keys}, witness
        if kind in ("boolean", "null"):
            return {"type": kind}, False if kind == "boolean" else None
        if kind == "array":
            lo = _count(node, "minItems", 0)
            hi = _count(node, "maxItems", 0) if "maxItems" in node else math.inf
            if lo > hi:
                return False, EMPTY
            if lo > 64 or ("maxItems" in node and hi > 256):
                _error("value_resource_limit: array minimum 64 / finite maximum 256")
            item, value = visit(node.get("items", True), depth + 1)
            if value is EMPTY and lo:
                return False, EMPTY
            result = {key: node[key] for key in ("type", "minItems", "maxItems") if key in node}
            result["items"] = item
            return result, [value] * lo
        properties = node.get("properties", {})
        required = node.get("required", [])
        if not isinstance(properties, dict) or len(properties) > 64:
            _error("value_resource_limit: nested object at most 64 properties")
        if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
            _error("required must be a string array")
        additional, extra = visit(node.get("additionalProperties", True), depth + 1)
        fields, witness = {}, {}
        if len(set(properties).union(required)) > 64:
            _error("value_resource_limit: nested object at most 64 effective properties")
        for key in dict.fromkeys(list(properties) + required):
            value_schema, value = visit(properties[key], depth + 1) if key in properties else (additional, extra)
            fields[key] = value_schema
            if key in required:
                if value is EMPTY:
                    return False, EMPTY
                witness[key] = value
        return {"type": "object", "properties": fields, "required": required,
                "additionalProperties": additional}, witness

    proven, witness = visit(schema)
    if witness is not EMPTY and len(json.dumps(witness, ensure_ascii=True)) > MAX_WITNESS_BYTES:
        _error("value_resource_limit: witness exceeds 65536 bytes")
    return proven, witness


def verify_native_witnesses(arguments, witnesses, proven_values, verified=None):
    """Check each distinct proven value language once within this tool request.

    A property key contributes only a fixed literal tag. Equal fully reduced
    schemas therefore differ only by that literal, which cannot make an
    otherwise productive value grammar empty. Keep distinct schema languages
    separate even when their selected witnesses happen to be equal.
    """
    if verified is None:
        verified = set()
    pending = {}
    for key in arguments:
        identity = json.dumps(proven_values[key], sort_keys=True, ensure_ascii=True,
                              separators=(",", ":"))
        if identity not in verified and identity not in pending:
            pending[identity] = key
    arguments = {key: arguments[key] for key in pending.values()}
    witnesses = {key: witnesses[key] for key in arguments}
    if not arguments:
        return
    import xgrammar as xgr
    from xgrammar.structural_tag import OrFormat, StructuralTag
    from sglang.srt.function_call import kimik3_structural_tag as k3
    info = xgr.TokenizerInfo(["[EOS]"], stop_token_ids=0)
    compiler = xgr.GrammarCompiler(info, max_threads=1, cache_enabled=False)
    compiled = compiler.compile_structural_tag(StructuralTag(format=OrFormat(elements=list(arguments.values()))))
    for key, value in witnesses.items():
        kind = "json-string" if isinstance(value, str) else k3._JSON_TO_XTML_TYPE[k3._json_type(value)]
        wire = ('<|open|>argument key="' + k3._escape_attr(key) + '" type="' + kind + '"<|sep|>'
                + json.dumps(value, ensure_ascii=True, separators=(",", ":")) + k3.ARGUMENT_CLOSE)
        matcher = xgr.GrammarMatcher(compiled)
        if not matcher.accept_string(wire) or not matcher.is_completed():
            _error("native value grammar did not accept its proven witness")
    verified.update(pending)
