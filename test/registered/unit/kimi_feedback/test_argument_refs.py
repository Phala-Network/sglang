"""Synthetic root-scope and preflight checks; no request captures."""
import copy
import json
import pytest
import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string
from sglang.srt.function_call.kimik3_structural_tag import (
    _argument_value_variants, _resolve_local_ref, _with_root_definitions,
)
from sglang.srt.constrained.json_schema_validation import (
    validate_xgrammar_json_schema, UnsupportedJSONSchemaFeature,
)

def accepts(schema, value):
    grammar = xgr.Grammar.from_json_schema(json.dumps(schema), strict_mode=False, separators=(',', ':'))
    return _is_grammar_accept_string(grammar, json.dumps(value, ensure_ascii=False, separators=(',', ':')))

def extract(root, name, kind):
    variants = _argument_value_variants(root['properties'][name], root)
    return next(fmt.json_schema for typ, _, fmt in variants if typ == kind)

def test_cross_argument_root_scope():
    root={'type':'object','properties':{
        'operations':{'type':'array','items':{'type':'object','properties':{
            'node':{'type':'object','properties':{'position':{'type':'integer','minimum':3,'maximum':9}}}}}},
        'value':{'type':'object','properties':{'p':{'$ref':'#/properties/operations/items/properties/node/properties/position'}},
                 'required':['p'],'additionalProperties':False}}}
    original=copy.deepcopy(root)
    schema=extract(root,'value','object')
    assert accepts(schema,{'p':3})
    assert not accepts(schema,{'p':2})
    assert not accepts(schema,{'p':'3'})
    assert root==original

@pytest.mark.parametrize('ref',['#/$defs/a~1b~0c%20d','#%2F$defs%2Fa~1b~0c%20d'])
def test_escaped_argument_ref(ref):
    root={'$defs':{'a/b~c d':{'type':'integer','minimum':2,'maximum':5}},
          'properties':{'v':{'type':'array','items':{'$ref':ref}}}}
    schema=extract(root,'v','array')
    assert accepts(schema,[2,5])
    assert not accepts(schema,[1])

def test_ref_rehome_is_bounded_and_does_not_rewrite_literals():
    root={'$defs':{'used':{'type':'integer','maximum':4},'unused':{'enum':['x'*100000]}},
          'properties':{'v':{'type':'object','properties':{
              'p':{'$ref':'#/$defs/used'},'q':{'$ref':'#/$defs/used'},
              'data':{'const':{'$ref':'this is instance data'}}},
              'required':['p','q','data'],'additionalProperties':False}}}
    schema=extract(root,'v','object')
    assert len(json.dumps(schema))<3000
    assert len(schema['$defs'])==1
    assert schema['properties']['data']['const']['$ref']=='this is instance data'
    assert accepts(schema,{'p':1,'q':3,'data':{'$ref':'this is instance data'}})
    assert not accepts(schema,{'p':5,'q':3,'data':{'$ref':'this is instance data'}})

def test_consuming_recursive_reference_preserved():
    root={'$defs':{'node':{'anyOf':[{'type':'null'}, {'type':'object','properties':{
        'value':{'type':'integer'},'next':{'$ref':'#/$defs/node'}},'required':['value','next'],'additionalProperties':False}]}},
          'properties':{'v':{'type':'array','items':{'$ref':'#/$defs/node'}}}}
    schema=extract(root,'v','array')
    assert accepts(schema,[{'value':3,'next':{'value':4,'next':None}}])
    assert not accepts(schema,[{'value':'bad','next':None}])

def test_namespaced_target_does_not_collide():
    root={'$defs':{'t':{'type':'integer'}},'properties':{'v':{
        'type':'array','$defs':{'__k3_ref_0':{'type':'string'}},'items':{'$ref':'#/$defs/t'}}}}
    schema=extract(root,'v','array')
    assert schema['$defs']['__k3_ref_0']=={'type':'string'}
    assert accepts(schema,[1]) and not accepts(schema,['1'])

@pytest.mark.parametrize('ref',['https://example.invalid/schema','#/missing','#/$defs/unused'])
def test_unused_definition_does_not_create_dependency(ref):
    root={'properties':{'v':{'type':'array','items':{'type':'integer'},
          '$defs':{'unused':{'$ref':ref}}}}}
    schema=extract(root,'v','array')
    assert accepts(schema,[1])

def test_unused_definition_inside_reachable_target():
    root={'$defs':{'used':{'type':'integer','$defs':{'unused':{'$ref':'#/missing'}}}},
          'properties':{'v':{'type':'array','items':{'$ref':'#/$defs/used'}}}}
    assert accepts(extract(root,'v','array'),[1])

@pytest.mark.parametrize('ref',['https://example.invalid/schema','#/missing'])
def test_used_bad_definition_still_refused(ref):
    root={'$defs':{'used':{'$ref':ref}},'properties':{'v':{'type':'array','items':{'$ref':'#/$defs/used'}}}}
    with pytest.raises(ValueError):
        extract(root,'v','array')

@pytest.mark.parametrize('ref',['#/', '#%2F', '#/definitions/list/0', '#/definitions/01'])
def test_pointer_edge_positive(ref):
    root={'':{'type':'integer'},'definitions':{'list':[{'type':'integer'}],'01':{'type':'integer'}}}
    assert _resolve_local_ref(ref,root)=={'type':'integer'}

@pytest.mark.parametrize('ref',['#/definitions/list/01','#/definitions/list/-1','#/definitions/list/+0','#/definitions/list/999999999999999999999999999999','#/bad~2','#/bad%xy','https://example.invalid/schema','#anchor'])
def test_pointer_edge_negative(ref):
    assert _resolve_local_ref(ref,{'definitions':{'list':[{}]}}) is None

@pytest.mark.parametrize('value,expected',[('ab',True),('abc',True),('a',False),('abcd',False),('a1',False),('éé',True)])
def test_pattern_length_preflight_and_native(value,expected):
    schema={'type':'string','pattern':'^[a-zé]+$','minLength':2,'maxLength':3}
    validate_xgrammar_json_schema(schema)
    assert accepts(schema,value)==expected

def test_unique_false_is_noop_but_true_stays_rejected():
    schema={'type':'array','items':{'type':'integer'},'uniqueItems':False}
    validate_xgrammar_json_schema(schema)
    assert accepts(schema,[1,1])
    with pytest.raises(UnsupportedJSONSchemaFeature):
        validate_xgrammar_json_schema(dict(schema,uniqueItems=True))

def test_format_pattern_is_not_silently_weakened():
    with pytest.raises(UnsupportedJSONSchemaFeature):
        validate_xgrammar_json_schema({'type':'string','format':'email','pattern':'^x$'})

@pytest.mark.parametrize('jump',[False,True])
def test_nested_resource_scope_is_refused_before_k3_extraction(jump):
    from sglang.srt.entrypoints.openai.protocol import Tool, Function
    from sglang.srt.function_call.kimik3_structural_tag import _tool_arguments_format
    inner={'$id':'https://example.invalid/inner','type':'array',
           '$defs':{'X':{'type':'string'}},'items':{'$ref':'#/$defs/X'}}
    root={'type':'object','$defs':{'X':{'type':'integer'},'inner':inner},
          'properties':{'v':{'$ref':'#/$defs/inner/items'} if jump else inner}}
    tool=Tool(type='function',function=Function(name='synthetic',strict=True,parameters=root))
    with pytest.raises(UnsupportedJSONSchemaFeature,match=r'nested \$id'):
        _tool_arguments_format(tool)
    with pytest.raises(UnsupportedJSONSchemaFeature,match=r'nested \$id'):
        validate_xgrammar_json_schema(root)

def test_reference_scope_guard_ignores_instance_data():
    from sglang.srt.constrained.json_schema_validation import validate_local_json_reference_scope
    value={'$id':'this is data','$ref':'this is also data'}
    for key in ('const','default','examples','enum'):
        schema={key:[value] if key in ('enum','examples') else value}
        validate_local_json_reference_scope(schema)
        validate_xgrammar_json_schema(schema)
