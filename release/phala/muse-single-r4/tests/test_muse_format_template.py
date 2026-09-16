"""Exercise schema visibility and native-format preservation in the installed image."""
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers.utils.chat_template_utils import _compile_jinja_template

SOURCE = Path(os.environ.get('MUSE_SERVING_SOURCE', '/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py'))
TEMPLATE = Path(os.environ.get('MUSE_TEMPLATE', '/opt/muse-glimmer-chat-template.jinja'))
SCHEMA = {'type':'array','minItems':1,'items':{'type':'object','properties':{'name':{'const':'lookup'},'parameters':{'type':'object','properties':{'text':{'type':'string'}},'required':['text'],'additionalProperties':False}},'required':['name','parameters'],'additionalProperties':False}}
TOOLS = [{'type':'function','function':{'name':'lookup','description':'Look up the record.','parameters':{'type':'object','properties':{'text':{'type':'string'}},'required':['text']}}}]


def helper():
    tree=ast.parse(SOURCE.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='muse_format_template_kwargs']
    assert len(nodes)==1, 'Muse format metadata must reach its template'
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),nodes[0]],type_ignores=[])
    ast.fix_missing_locations(module)
    namespace={}
    exec(compile(module,str(SOURCE),'exec'),namespace)
    return namespace['muse_format_template_kwargs']


class Format:
    def __init__(self,value): self.value=value; self.type=value['type']
    def model_dump(self,**kwargs): return copy.deepcopy(self.value)


@pytest.mark.parametrize('parser',['gpt-oss','nemotron_3_5','qwen3',None])
def test_other_models_have_no_template_override(parser):
    request=SimpleNamespace(response_format=Format({'type':'json_object'}))
    assert helper()(request,parser,('json_schema',SCHEMA))=={}


@pytest.mark.parametrize('constraint',[None,('structural_tag',{'type':'structural_tag'}),('json_schema',SCHEMA)])
def test_only_the_actual_json_tool_constraint_is_exposed(constraint):
    request=SimpleNamespace(response_format=None)
    original=copy.deepcopy(constraint)
    result=helper()(request,'muse',constraint)
    assert constraint==original
    if constraint and constraint[0]=='json_schema':
        assert result['_phala_muse_tool_schema']==SCHEMA
    else:
        assert result=={}


@pytest.mark.parametrize('kind',['json_object','json_schema','text'])
def test_response_schema_is_not_replaced_by_a_tool_schema(kind):
    value={'type':kind}
    if kind=='json_schema': value['json_schema']={'name':'record','strict':True,'schema':{'type':'object','properties':{'record':{'const':'原始\\value'}}}}
    request=SimpleNamespace(response_format=Format(value))
    result=helper()(request,'muse',None)
    assert result==({'_phala_muse_response_format':value} if kind!='text' else {})


@pytest.mark.parametrize('system',[True,False])
@pytest.mark.parametrize('reasoning',['none','high'])
def test_json_mode_keeps_the_original_user_content_and_reasoning_contract(system,reasoning):
    template=_compile_jinja_template(TEMPLATE.read_text())
    messages=([{'role':'system','content':'Keep this system rule.'}] if system else [])+[{'role':'user','content':'Literal 中文 and "quotes" and \\ backslash.'}]
    kwargs={'messages':messages,'bos_token':'<s>','add_generation_prompt':True,'reasoning_strength':reasoning,'current_date':'2026-09-16'}
    baseline=template.render(**kwargs)
    rendered=template.render(**kwargs,_phala_muse_response_format={'type':'json_object'})
    marker='Your final answer must be one valid JSON object containing the data requested by the user, without explanation or unrelated content.'
    assert rendered.count(marker)==1
    assert rendered.replace('\n\n'+marker,'',1)==baseline
    assert ('<|start|>assistant to=user<|message|>' if reasoning=='none' else '<|start|>assistant') in rendered


@pytest.mark.parametrize('system',[True,False])
def test_required_tools_describe_the_enforced_array_not_atem(system):
    template=_compile_jinja_template(TEMPLATE.read_text())
    messages=([{'role':'system','content':'Keep this system rule.'}] if system else [])+[{'role':'user','content':'Use the requested tools.'}]
    rendered=template.render(messages=messages,tools=TOOLS,bos_token='<s>',add_generation_prompt=True,reasoning_strength='none',_phala_muse_tool_schema=SCHEMA)
    assert 'JSON array' in rendered
    assert '<atem:invoke name="$FUNCTION_NAME">' not in rendered
    assert 'Look up the record.' in rendered
    assert json.dumps(SCHEMA,ensure_ascii=False,separators=(',',':')) in rendered.replace(' ','')
    assert 'Use the requested tools.' in rendered


@pytest.mark.parametrize('media',[False,True])
@pytest.mark.parametrize('tools',[False,True])
def test_unconstrained_chat_and_media_preserve_the_original_template(media,tools):
    template=_compile_jinja_template(TEMPLATE.read_text())
    original=_compile_jinja_template((Path(__file__).parent/'muse-r3-original.jinja').read_text())
    content=[{'type':'text','text':'Inspect the image.'},{'type':'image'}] if media else 'Plain answer.'
    kwargs={'messages':[{'role':'user','content':content}],'bos_token':'<s>','add_generation_prompt':True,'reasoning_strength':'high','current_date':'2026-09-16'}
    if tools: kwargs['tools']=TOOLS
    assert template.render(**kwargs)==original.render(**kwargs)
