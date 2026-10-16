"""A shared target expanded into union branches must be rebound only once."""
import copy
import json
import pytest
import xgrammar as xgr
from jsonschema import Draft202012Validator
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.kimik3_structural_tag import _tool_arguments_format
from xgrammar.structural_tag import StructuralTag, TagFormat


@pytest.mark.parametrize('branches',[1,2,3])
def test_repeated_union_target_with_nested_reference(branches):
    schema={'type':'object','properties':{'payload':{'anyOf':[{'$ref':'#/$defs/A'} for _ in range(branches)]}},
            'required':['payload'],'additionalProperties':False,'$defs':{
                'A':{'type':'object','properties':{'value':{'$ref':'#/$defs/B'}},'required':['value'],'additionalProperties':False},
                'B':{'type':'integer','minimum':2,'maximum':5}}}
    original=copy.deepcopy(schema)
    tool=Tool(type='function',function={'name':'synthetic','parameters':schema,'strict':True})
    tag=StructuralTag(format=TagFormat(begin='[',end=']',content=_tool_arguments_format(tool)))
    assert schema==original
    compiled=xgr.GrammarCompiler(xgr.TokenizerInfo(['x','EOS'],stop_token_ids=1)).compile_structural_tag(tag)
    oracle=Draft202012Validator(schema)
    for value in [{'value':2},{'value':5},{'value':1},{'value':6},{'value':'2'},{}, {'value':3,'extra':1}]:
        wire='[<|open|>argument key="payload" type="object"<|sep|>'+json.dumps(value)+'<|close|>argument<|sep|>]'
        matcher=xgr.GrammarMatcher(compiled)
        assert (matcher.accept_string(wire) and matcher.is_completed())==oracle.is_valid({'payload':value})


def test_repeated_array_target_keeps_nested_constraints():
    schema={'type':'object','properties':{'payload':{'anyOf':[{'$ref':'#/$defs/A'},{'$ref':'#/$defs/A'}]}},
            'required':['payload'],'additionalProperties':False,'$defs':{
                'A':{'type':'array','items':{'$ref':'#/$defs/B'},'minItems':1,'maxItems':2},
                'B':{'type':'integer','minimum':2,'maximum':5}}}
    original=copy.deepcopy(schema)
    tool=Tool(type='function',function={'name':'synthetic','parameters':schema,'strict':True})
    tag=StructuralTag(format=TagFormat(begin='[',end=']',content=_tool_arguments_format(tool)))
    assert schema==original
    compiled=xgr.GrammarCompiler(xgr.TokenizerInfo(['x','EOS'],stop_token_ids=1)).compile_structural_tag(tag)
    oracle=Draft202012Validator(schema)
    for value in [[],[2],[2,5],[1],[6],['2'],[2,3,4]]:
        wire='[<|open|>argument key="payload" type="array"<|sep|>'+json.dumps(value)+'<|close|>argument<|sep|>]'
        matcher=xgr.GrammarMatcher(compiled)
        assert (matcher.accept_string(wire) and matcher.is_completed())==oracle.is_valid({'payload':value})
