"""Actual Chat/Responses parser entrypoints preserve constrained JSON strings."""
import json
from unittest.mock import Mock
import pytest
from responses_cpu_helpers import make_serving, StreamFixture, engine_chunk, event_payloads
from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.function_call.kimik3_format import THINK_CLOSE, RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE

@pytest.mark.parametrize('thinking',[True,False])
@pytest.mark.parametrize('kind,strict',[('json_object',None),('json_schema',True),('json_schema',False)])
def test_responses_json_entrypoints(thinking,kind,strict):
    serving=make_serving()
    serving.reasoning_parser='kimi_k3'
    serving.tool_call_parser=None
    serving.template_manager.force_reasoning=True
    value={'text':RESPONSE_CLOSE+' '+RESPONSE_OPEN+' \\ "'}
    fmt={'type':kind}
    if kind=='json_schema':fmt.update(name='fixture',schema={'type':'object'},strict=strict)
    request=ResponsesRequest(model='x',input='synthetic',text={'format':fmt},store=False)
    assert request.has_json_schema_constraint()
    assert request.to_sampling_params(default_max_tokens=128).get('json_schema') is not None
    body=json.dumps(value,separators=(',',':'))
    native=('reason'+THINK_CLOSE if thinking else '')+RESPONSE_OPEN+body+RESPONSE_CLOSE+MESSAGE_CLOSE
    items=serving._make_response_output_items(request,native,tokenizer=Mock(),require_reasoning=thinking)
    messages=[i for i in items if i.type=='message']
    assert len(messages)==1 and json.loads(messages[0].content[0].text)==value
    # Engine text is cumulative. Split native markers and JSON escape sequences.
    chunks=[engine_chunk(native[:i],completion_tokens=i,finish=i==len(native)) for i in range(1,len(native)+1)]
    events=event_payloads(StreamFixture(serving,request,require_reasoning=thinking).run(chunks))
    streamed=''.join(e.get('delta','') for e in events if e.get('type')=='response.output_text.delta')
    assert json.loads(streamed)==value

@pytest.mark.parametrize('thinking',[True,False])
@pytest.mark.parametrize('kind,strict',[('json_object',None),('json_schema',True),('json_schema',False)])
def test_chat_stream_parser_entrypoint_and_choice_isolation(thinking,kind,strict):
    from types import SimpleNamespace, MethodType
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    fmt={'type':kind}
    if kind=='json_schema':fmt['json_schema']={'name':'fixture','schema':{'type':'object'},'strict':strict}
    request=ChatCompletionRequest(model='x',messages=[{'role':'user','content':'synthetic'}],response_format=fmt)
    owner=SimpleNamespace(reasoning_parser='kimi_k3',template_manager=SimpleNamespace(force_reasoning=True),tokenizer_manager=SimpleNamespace(tokenizer=Mock()))
    owner._get_reasoning_from_request=lambda req: thinking
    owner._tool_call_parsing_active=lambda req: False
    owner._should_force_reasoning=lambda req: owner.template_manager.force_reasoning or thinking
    owner._kimi_k3_structured_json=MethodType(OpenAIServingChat._kimi_k3_structured_json,owner)
    states={}
    protected=kind=='json_object' or strict is not False
    values=[{'text':RESPONSE_CLOSE},{'text':RESPONSE_OPEN}]
    outputs=['','']
    natives=[('reason'+THINK_CLOSE if thinking else '')+RESPONSE_OPEN+json.dumps(v)+RESPONSE_CLOSE+MESSAGE_CLOSE for v in values]
    for offset in range(max(map(len,natives))):
        for choice,native in enumerate(natives):
            if offset>=len(native):continue
            _,normal=OpenAIServingChat._process_reasoning_stream(owner,choice,native[offset],states,{},request,finish_reason_type='stop' if offset==len(native)-1 else None)
            outputs[choice]+=normal or ''
    assert states[0].detector is not states[1].detector
    for choice in range(2):
        assert (states[choice].detector._json_content_parser is not None)==protected
        if protected:assert json.loads(outputs[choice])==values[choice]
