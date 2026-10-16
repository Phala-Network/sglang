"""Synthetic clock and real chat stream: no wall-clock sleeps or GPU traffic."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sglang.srt.entrypoints.openai import serving_chat as M
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest


def run(rows, *, kimi=True, tools=True, visible=False, cancel_comment=False):
    clock = [0.0]
    class Generation:
        closed = 0
        def __aiter__(self): return self.generate()
        async def generate(self):
            for when, index, tokens, finish in rows:
                clock[0] = when
                yield {'text':'synthetic', 'index':index, 'meta_info':{
                    'id':'synthetic','prompt_tokens':1,'completion_tokens':tokens,
                    'cached_tokens':0,'finish_reason':finish}}
        async def aclose(self): self.closed += 1
    generation=Generation()
    serving=object.__new__(M.OpenAIServingChat)
    serving.chat_encoding_spec='kimi_k3' if kimi else None
    # Attributes __init__ sets on the fork's main line.
    serving.reasoning_parser=serving.tool_call_parser=None
    serving._encoding_adapter=None
    serving.tokenizer_manager=SimpleNamespace(generate_request=lambda *a:generation)
    serving._validate_kimi_k3_tool_call_completion=lambda *a:None
    async def content(**kwargs):
        if visible:
            yield M.build_sse_content(chunk_id='synthetic',created=1,model='synthetic',index=kwargs['index'],content='synthetic')
    serving._generate_stream_content=content
    request=ChatCompletionRequest(model='synthetic',messages=[{'role':'user','content':'synthetic'}],
        stream=True,n=max(r[1] for r in rows)+1,
        tools=[{'type':'function','function':{'name':'synthetic','parameters':{'type':'object'}}}] if tools else None)
    config=SimpleNamespace(stream_response_default_include_usage=False,return_input_ids=False,
        return_output_ids=False,enable_cache_report=False,incremental_streaming_output=False)
    async def collect():
        chunks=[];error=None
        stream=serving._generate_chat_stream(SimpleNamespace(),request,None)
        try:
            async for chunk in stream:
                chunks.append(chunk)
                if cancel_comment and chunk.startswith(':'):
                    await stream.aclose();break
        except BaseException as exc:error=exc
        return chunks,error
    with patch.object(M,'get_serving',return_value=config),patch.object(M,'_stream_progress_clock',create=True,side_effect=lambda:clock[0]):
        chunks,error=asyncio.run(collect())
    return chunks,error,generation.closed


def comments(chunks):return [c for c in chunks if c.startswith(':')]
def events(chunks):return [json.loads(c[6:]) for c in chunks if c.startswith('data: ') and c.strip()!='data: [DONE]']


def test_buffered_generation_emits_bounded_progress_without_data_changes():
    rows=[(0,0,1,None),(14,0,2,None),(15,0,3,None),(29,0,4,None),(30,0,5,None),(90,0,6,{'type':'stop'})]
    chunks,error,closed=run(rows)
    assert error is None and closed==1
    assert comments(chunks)==[': keep-alive\n\n']*2
    assert sum(c.strip()=='data: [DONE]' for c in chunks)==1
    assert not any('error' in e for e in events(chunks))
    assert [c['finish_reason'] for e in events(chunks) for c in e.get('choices',[]) if c.get('finish_reason')]==['stop']
    assert all(c==': keep-alive\n\n' for c in comments(chunks))


@pytest.mark.parametrize('kwargs',[{'kimi':False},{'tools':False},{'visible':True}])
def test_unaffected_paths_do_not_inject_comments(kwargs):
    chunks,error,closed=run([(0,0,1,None),(30,0,2,None),(60,0,3,{'type':'stop'})],**kwargs)
    assert error is None and closed==1 and not comments(chunks)


def test_no_new_tokens_and_terminal_tokens_do_not_send_progress():
    chunks,error,closed=run([(0,0,2,None),(30,0,2,None),(60,0,1,None),(90,0,4,{'type':'stop'})])
    assert error is None and closed==1 and not comments(chunks)


@pytest.mark.parametrize('status',[400,500,503,504])
def test_first_abort_stays_pre_header(status):
    chunks,error,closed=run([(500,0,0,{'type':'abort','status_code':status,'message':'synthetic'})])
    assert chunks==[] and isinstance(error,HTTPException) and error.status_code==status and closed==1


def test_cancel_after_comment_closes_generation_without_done():
    chunks,error,closed=run([(0,0,1,None),(30,0,2,None),(60,0,3,{'type':'stop'})],cancel_comment=True)
    assert error is None and closed==1 and len(comments(chunks))==1
    assert not any('[DONE]' in c for c in chunks)


def test_multichoice_progress_is_per_choice_and_wire_timer_shared():
    chunks,error,closed=run([(0,0,100,None),(1,1,1,None),(16,1,2,None),
        (20,0,101,None),(31,0,102,None),(40,0,103,{'type':'stop'}),(41,1,3,{'type':'stop'})])
    assert error is None and closed==1 and len(comments(chunks))==2
    assert sorted(c['index'] for e in events(chunks) for c in e.get('choices',[]) if c.get('finish_reason'))==[0,1]
