"""Bounded, opt-in Xing4 native parameter assistance; never edit caller schemas.

The generated language covers root declared keys, required keys and direct value
types. Other original assertions remain the model/scorer's responsibility.
"""
import copy
import json
import logging
import os
from collections import Counter
from collections.abc import Mapping

VERSION = 'xing4-native-schema-v2-root-types-5-shallow'
MARKERS = ('<tool_call>', '</tool_call>', '<param_key>', '</param_key>',
           '<param_value>', '</param_value>')
ANNOTATIONS = frozenset({'title', 'description', 'default', 'examples', '$comment',
                         'deprecated', 'readOnly', 'writeOnly'})
MAX_TOOLS = 64
MAX_PROPERTIES = 32
MAX_DEPTH = 12
MAX_TOOL_NODES = 512
MAX_REQUEST_NODES = 2048
logger = logging.getLogger(__name__)


class NativeToolSchemaTag(str):
    """Server-only provenance retained by the normal Python IPC path.

    JSON callers can supply only an ordinary string. The current request's
    concrete type, never a cache key or a JSON marker, authorizes fallback.
    """


def maybe_apply_native_schema(sampling_params, request, tools, choice, parser_name):
    if os.environ.get('SGLANG_XING4_NATIVE_TOOL_SCHEMA_V2') != '1' or parser_name != 'xing4_0':
        return False
    if any(sampling_params.get(key) is not None for key in ('json_schema','regex','ebnf','structural_tag')):
        return False
    from sglang.srt.environ import ToolStrictLevel, envs
    if envs.SGLANG_TOOL_STRICT_LEVEL.get() != ToolStrictLevel.OFF:
        return False
    try:
        if isinstance(tools, list) and (not tools or len(tools) > MAX_TOOLS):
            return False
        wire = request.model_dump(mode='python', exclude_none=True, include={
            'parallel_tool_calls', 'n', 'response_format', 'json_schema',
            'regex', 'ebnf', 'structural_tag',
        })
        wire['tools'] = [tool.model_dump(mode='python',exclude_none=True) for tool in tools]
        wire['tool_choice'] = choice
        tag = build_native_tag(wire)
        if tag is None:
            return False
        sampling_params['structural_tag'] = NativeToolSchemaTag(json.dumps(tag))
        logger.info('Xing4 native parameter assistance bound')
        return True
    except Exception:
        logger.warning('Xing4 native parameter assistance unavailable; using original request path')
        return False


def _safe_text(value):
    return (isinstance(value, str) and value == value.strip() and '\x00' not in value
            and not any(marker in value for marker in MARKERS))


def inspect_parameters(schema):
    reasons, omitted = Counter(), Counter()
    nodes = 0

    def visit(value, depth=0, native_root=False):
        nonlocal nodes
        nodes += 1
        if depth > MAX_DEPTH or nodes > MAX_TOOL_NODES:
            reasons['schema_resource_bound'] += 1
            return
        if not isinstance(value, Mapping):
            reasons['non_object_schema'] += 1
            return
        kind = value.get('type')
        if not isinstance(kind, str):
            reasons['type_union_or_unspecified'] += 1
            return
        if kind not in {'string', 'number', 'integer', 'boolean', 'null', 'array', 'object'}:
            reasons['unsupported_type'] += 1
            return
        if native_root and kind != 'object':
            reasons['root_not_object'] += 1
            return
        allowed = {'type', 'enum', 'const'} | ANNOTATIONS
        if kind == 'object':
            allowed |= {'properties', 'required', 'additionalProperties'}
            props, required = value.get('properties', {}), value.get('required', [])
            if not isinstance(props, Mapping) or len(props) > MAX_PROPERTIES:
                reasons['properties_shape_or_bound'] += 1
                return
            if (not isinstance(required, list) or
                    any(not isinstance(key, str) for key in required) or
                    len(set(required)) != len(required) or not set(required).issubset(props)):
                reasons['required_shape_or_undeclared'] += 1
                return
            if any(not _safe_text(key) for key in props):
                reasons['native_key_not_lossless'] += 1
            ap = value.get('additionalProperties', True)
            if native_root and ap is not False:
                reasons['root_ap_not_explicit_false'] += 1
            elif not native_root and isinstance(ap, Mapping):
                visit(ap, depth+1)
            elif not native_root and not isinstance(ap, bool):
                reasons['additional_properties_shape'] += 1
            for child in props.values():
                visit(child, depth+1)
        elif kind == 'array':
            allowed |= {'items', 'minItems', 'maxItems'}
            if 'items' in value:
                visit(value['items'], depth+1)
        elif kind in {'integer', 'number'}:
            # An internal type-only projection safely omits these intersections.
            allowed |= {'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf'}
        elif kind == 'string':
            allowed |= {'minLength','maxLength','pattern','format','contentEncoding',
                        'contentMediaType','contentSchema'}
        unknown = set(value) - allowed
        if unknown:
            reasons['unsupported_schema_keyword'] += 1
        for keyword in set(value) - {'type'} - ANNOTATIONS:
            if not native_root:
                omitted[keyword if keyword in allowed else 'unknown'] += 1

    visit(schema, native_root=True)
    return {'eligible': not reasons, 'schema_nodes':nodes,
            'fallback_reason_counts':dict(sorted(reasons.items())),
            'unmodeled_child_keyword_counts':dict(sorted(omitted.items()))}


def _inspect_passthrough_safety(schema):
    """Bound JSON traversal without interpreting unsupported schema semantics."""
    reasons, nodes = Counter(), 0

    def visit(value, depth=0):
        nonlocal nodes
        if reasons:
            return
        nodes += 1
        if depth > MAX_DEPTH or nodes > MAX_TOOL_NODES:
            reasons['schema_resource_bound'] += 1
            return
        if isinstance(value, Mapping):
            kind = value.get('type')
            if kind == 'object' or 'properties' in value:
                props, required = value.get('properties', {}), value.get('required', [])
                if not isinstance(props, Mapping) or len(props) > MAX_PROPERTIES:
                    reasons['properties_shape_or_bound'] += 1
                    return
                if (not isinstance(required, list) or len(required) > len(props) or
                        any(not isinstance(key, str) for key in required) or
                        len(set(required)) != len(required) or not set(required).issubset(props)):
                    reasons['required_shape_or_undeclared'] += 1
                    return
                if any(not _safe_text(key) for key in props):
                    reasons['native_key_not_lossless'] += 1
                    return
                ap = value.get('additionalProperties', True)
                if not isinstance(ap, (bool, Mapping)):
                    reasons['additional_properties_shape'] += 1
                    return
            for key, child in value.items():
                if not isinstance(key, str):
                    reasons['schema_json_shape'] += 1
                    return
                visit(child, depth+1)
                if reasons:
                    return
        elif isinstance(value, list):
            for child in value:
                visit(child, depth+1)
                if reasons:
                    return
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            reasons['schema_json_shape'] += 1

    if not isinstance(schema, Mapping):
        reasons['non_object_schema'] += 1
    else:
        visit(schema)
    return {'eligible':not reasons, 'schema_nodes':nodes,
            'fallback_reason_counts':dict(sorted(reasons.items()))}


def _has_supported_root_types(schema):
    """Qualify only the root and direct types already used by the generator."""
    if not isinstance(schema, Mapping):
        return False
    root_keys = {'type', 'properties', 'required', 'additionalProperties'} | ANNOTATIONS
    if set(schema) - root_keys:
        return False
    if schema.get('type') != 'object' or schema.get('additionalProperties') is not False:
        return False
    props = schema.get('properties')
    if not isinstance(props, Mapping) or not isinstance(schema.get('required'), list):
        return False
    if any(not isinstance(child, Mapping) for child in props.values()):
        return False
    # Full JSON safety was checked by the caller; do not interpret descendants.
    projected = dict(schema)
    projected['properties'] = {key: {'type': child.get('type')} for key, child in props.items()}
    return inspect_parameters(projected)['eligible']


def inspect_request(request, target_name=None):
    reasons, omitted = Counter(), Counter()
    if not isinstance(request, Mapping):
        return {'version':VERSION, 'whole_request_eligible':False,
                'target_tool_found':False, 'target_tool_eligible':False,
                'fallback_reason_counts':{'request_shape':1}}
    if request.get('tool_choice', 'auto') != 'auto':
        reasons['choice_not_auto'] += 1
    if request.get('parallel_tool_calls', True) is not True:
        reasons['parallel_not_true_or_default'] += 1
    if request.get('n',1) != 1:
        reasons['multiple_choices'] += 1
    response_format = request.get('response_format')
    if response_format and (not isinstance(response_format,Mapping) or response_format.get('type') != 'text'):
        reasons['existing_output_constraint'] += 1
    if any(request.get(key) is not None for key in ('json_schema','regex','ebnf','structural_tag')):
        reasons['existing_output_constraint'] += 1
    tools = request.get('tools',[])
    tools_valid = isinstance(tools,list) and bool(tools) and len(tools) <= MAX_TOOLS
    if not tools_valid:
        reasons['tools_shape_or_bound'] += 1
        tools = tools if isinstance(tools,list) and len(tools) <= MAX_TOOLS else []
    names, nodes, eligible_count, target_found, target_eligible = [], 0, 0, False, False
    modes, passthrough_count, passthrough_reasons = [], 0, Counter()
    root_types_count = 0
    soft_reasons = {'type_union_or_unspecified', 'unsupported_schema_keyword',
                    'root_ap_not_explicit_false'}
    for tool in tools:
        if not isinstance(tool,Mapping) or tool.get('type') != 'function' or not isinstance(tool.get('function'),Mapping):
            reasons['function_tool_shape'] += 1
            modes.append('fallback')
            continue
        function = tool['function']
        name, strict = function.get('name'), function.get('strict',False)
        names.append(name)
        good_name = bool(name) and _safe_text(name)
        if not good_name:
            reasons['native_tool_name_not_lossless'] += 1
        if strict is not False and strict is not None:
            reasons['explicit_strict_or_nonboolean'] += 1
        result = inspect_parameters(function.get('parameters'))
        eligible = result['eligible'] and bool(good_name) and (strict is False or strict is None)
        eligible_count += eligible
        mode, counted_nodes = 'schema' if eligible else 'fallback', result['schema_nodes']
        local_reasons = result['fallback_reason_counts']
        if (not eligible and good_name and (strict is False or strict is None)
                and local_reasons and set(local_reasons).issubset(soft_reasons)):
            safety = _inspect_passthrough_safety(function.get('parameters'))
            counted_nodes = safety['schema_nodes']
            if safety['eligible']:
                if _has_supported_root_types(function.get('parameters')):
                    mode, eligible = 'schema', True
                    eligible_count += 1
                    root_types_count += 1
                    local_reasons = {}
                else:
                    mode = 'passthrough'
                    passthrough_count += 1
                    passthrough_reasons.update(local_reasons)
            else:
                reasons.update(safety['fallback_reason_counts'])
        modes.append(mode)
        nodes += counted_nodes
        if mode != 'passthrough':
            reasons.update(local_reasons)
        omitted.update(result['unmodeled_child_keyword_counts'])
        if nodes > MAX_REQUEST_NODES:
            break
        if target_name is not None and name == target_name:
            target_found, target_eligible = True, eligible
    if len(set(name for name in names if isinstance(name,str))) != len(names):
        reasons['duplicate_or_invalid_tool_name'] += 1
    if nodes > MAX_REQUEST_NODES:
        reasons['request_schema_resource_bound'] += 1
    if passthrough_count or root_types_count:
        if not eligible_count:
            reasons['no_schema_eligible_tools'] += 1
        begins = sorted('<tool_call>'+prefix+name+'\n'
                        for name in names if isinstance(name, str) for prefix in ('', ' '))
        if any(right.startswith(left) for left, right in zip(begins, begins[1:])):
            reasons['native_tool_prefix_ambiguity'] += 1
    return {'version':VERSION, 'whole_request_eligible':not reasons,
            'target_tool_found':target_found, 'target_tool_eligible':target_eligible,
            'tools_count':len(tools), 'schema_nodes_count':nodes,
            'schema_eligible_tools_count':eligible_count,
            'schema_passthrough_tools_count':passthrough_count,
            'schema_root_types_promoted_tools_count':root_types_count,
            'tool_assistance_modes':modes,
            'passthrough_reason_counts':dict(sorted(passthrough_reasons.items())),
            'fallback_reason_counts':dict(sorted(reasons.items())),
            'unmodeled_child_keyword_counts':dict(sorted(omitted.items()))}


def build_native_tag(request):
    """Return a format dict or None. Only XGrammar's cheap schema-to-EBNF conversion.

    Actual tokenizer compilation remains in the existing async backend.
    """
    inspection = inspect_request(request)
    if not inspection['whole_request_eligible']:
        return None
    import xgrammar as xgr
    tags = []
    ws = {'type':'grammar', 'grammar':'root ::= [ \\t\\r\\n]{0,16}\n'}
    for tool, mode in zip(request['tools'], inspection['tool_assistance_modes']):
        function = tool['function']
        if mode == 'passthrough':
            content = {'type':'any_text', 'excludes':['</tool_call>']}
        else:
            schema = function['parameters']
            required = set(schema.get('required',[]))
            elements = []
            for key, child in schema.get('properties',{}).items():
                kind = child['type']
                if kind == 'string':
                    content = {'type':'any_text', 'excludes':['</param_value>','</tool_call>']}
                else:
                    projection = {'type':kind}
                    if kind == 'object':
                        projection['additionalProperties'] = True
                    elif kind == 'array' and isinstance(child.get('items'), Mapping) and child['items'].get('type') == 'string':
                        projection['items'] = {'type':'string'}
                    grammar = xgr.Grammar.from_json_schema(projection, strict_mode=False)
                    content = {'type':'grammar', 'grammar':str(grammar)}
                field = {'type':'sequence','elements':[copy.deepcopy(ws), {
                    'type':'tag', 'begin':'<param_key>'+key+'</param_key><param_value>',
                    'content':content, 'end':'</param_value>'}]}
                elements.append(field if key in required else {'type':'optional','content':field})
            elements.append(copy.deepcopy(ws))
            content = {'type':'sequence','elements':elements}
        # SentencePiece may restart after the native special tag with one dummy space.
        for name_prefix in ('', ' '):
            tags.append({'type':'tag', 'begin':'<tool_call>'+name_prefix+function['name']+'\n',
                         'content':content, 'end':'</tool_call>'})
    return {'type':'structural_tag', 'format':{'type':'triggered_tags',
            'triggers':['<tool_call>'], 'tags':tags, 'at_least_one':False,
            'stop_after_first':False}}
