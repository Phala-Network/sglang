"""Exercise the actual message-processing to Jinja call chain without a GPU."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, '/nearby')
from test_serving_chat import _MockTemplateManager, _MockTokenizerManager
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat


def server():
    manager=_MockTokenizerManager()
    template=_MockTemplateManager()
    template.chat_template_name=None
    template.jinja_template_content_format='string'
    manager._config_overrides.update({'tool_call_parser':'muse','reasoning_parser':'muse'})
    manager.tokenizer.apply_chat_template.return_value='rendered prompt'
    return OpenAIServingChat(manager,template),manager


@pytest.mark.parametrize('kind',[None,'json_object','json_schema'])
def test_plain_and_structured_requests_reach_the_template(kind):
    chat,manager=server()
    kwargs={}
    if kind=='json_object': kwargs['response_format']={'type':kind}
    if kind=='json_schema': kwargs['response_format']={'type':kind,'json_schema':{'name':'sample','strict':True,'schema':{'type':'array','items':{'type':'string'}}}}
    request=ChatCompletionRequest(model='muse',messages=[{'role':'user','content':'Read this record.'}],**kwargs)
    result=chat._process_messages(request,False)
    actual=manager.tokenizer.apply_chat_template.call_args.kwargs
    assert result.prompt_ids==[1,2,3,4,5]
    if kind:
        assert actual['_phala_muse_response_format']['type']==kind
        if kind=='json_schema': assert actual['_phala_muse_response_format']['json_schema']['schema']['type']=='array'
    else:
        assert '_phala_muse_response_format' not in actual
    assert '_phala_muse_tool_schema' not in actual


@pytest.mark.parametrize('choice',['required',{'type':'function','function':{'name':'lookup'}}])
@pytest.mark.parametrize('parallel',[True,False])
def test_selected_tool_array_is_passed_without_changing_cardinality(choice,parallel):
    chat,manager=server()
    request=ChatCompletionRequest(model='muse',messages=[{'role':'user','content':'Use the tool.'}],tools=[{'type':'function','function':{'name':'lookup','description':'Look up a record.','strict':True,'parameters':{'type':'object','properties':{'text':{'type':'string'}},'required':['text'],'additionalProperties':False}}}],tool_choice=choice,parallel_tool_calls=parallel)
    result=chat._process_messages(request,False)
    actual=manager.tokenizer.apply_chat_template.call_args.kwargs
    assert actual['_phala_muse_tool_schema']==result.tool_call_constraint[1]
    assert actual['_phala_muse_tool_schema']['minItems']==1
    assert actual['_phala_muse_tool_schema'].get('maxItems')==(None if parallel else 1)
