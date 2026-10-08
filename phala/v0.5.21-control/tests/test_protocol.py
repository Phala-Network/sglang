import copy
import json
import os
from types import SimpleNamespace as NS
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai import encoding_dsv41 as enc
from sglang.srt.entrypoints.openai.dsv41_compat import _validate_media_parts, drop_tool_definitions

checks = []
def check(name, value):
    assert value, name
    checks.append(name)

def request(**kwargs):
    data = dict(model='test', messages=[dict(role='user', content='hi')], **kwargs)
    before = copy.deepcopy(data)
    result = ChatCompletionRequest(**data)
    check('input-preserved', data == before)
    return result

for effort in (1, 50, 100):
    for nested in (False, True):
        r = request(**({'reasoning': {'effort': effort}} if nested else {'reasoning_effort': effort}))
        check('integer-budget', r.reasoning_effort is None and r.chat_template_kwargs == dict(reasoning_effort=effort, thinking=True, enable_thinking=True))
for effort in (0, .5, .99):
    check('float-budget', request(reasoning_effort=effort).reasoning_effort == effort)
for effort in (True, False, -1, 101, 1.0):
    for nested in (False, True):
        try:
            request(**({'reasoning': {'effort': effort}} if nested else {'reasoning_effort': effort}))
        except ValueError:
            checks.append('invalid-budget')
        else:
            raise AssertionError(('accepted invalid', effort, nested))
for enabled in (False, 'false', '0', 'off'):
    r = request(reasoning={'enabled': enabled, 'effort': 50})
    check('disabled', r.reasoning_effort == 'none' and not r.chat_template_kwargs['thinking'])
r = request(reasoning={'enabled': False, 'effort': 50}, reasoning_effort='high')
check('flat-precedence', r.reasoning_effort == 'high' and r.chat_template_kwargs['thinking'])
r = request(reasoning={'effort': 50}, reasoning_effort='low')
check('flat-precedence-nested-int', r.reasoning_effort == 'low' and 'reasoning_effort' not in r.chat_template_kwargs)
r = request(reasoning_effort=50, chat_template_kwargs={'reasoning_effort': 7})
check('explicit-template', r.chat_template_kwargs['reasoning_effort'] == 7)
r = request(reasoning_effort='default', reasoning={'effort': 'default'})
check('default-absent', r.reasoning_effort is None and not r.chat_template_kwargs)
r = request(stream=True, stream_options={'include_usage':False, 'continuous_usage_stats':False}, reasoning={'exclude':'true'})
check('stream-flags-exclude', r.stream_options.include_usage and r.stream_options.continuous_usage_stats and r.reasoning_exclude)
check('nonstream-options', request().stream_options is None)
u = UsageProcessor.calculate_token_usage(10, 7, 3)
check('usage-details', u.completion_tokens_details == {'reasoning_tokens': 3} and u.total_tokens == 17 and u.reasoning_tokens == 3)
check('effort-mapping', enc.REASONING_EFFORT_MAPPINGS == dict(minimal=25, low=50, medium=62, high=75, xhigh=90, max=100))
messages = [dict(role='developer', content='rules'), dict(role='user', content='hi')]
before = copy.deepcopy(messages)
check('developer-system', enc.encode_messages(messages, 'thinking') == enc.encode_messages([dict(role='system', content='rules'), messages[1]], 'thinking'))
check('role-input-preserved', before == messages)
check('lone-developer-generation', '<｜Assistant｜>' in enc.encode_messages([dict(role='developer', content='rules')], 'thinking'))
os.environ['DSV41_SYSTEM_ONLY_USER_TURN'] = 'true'
check('system-user-enabled', '<｜Assistant｜>' in enc.encode_messages([dict(role='system', content='rules')], 'thinking'))
os.environ.pop('DSV41_SYSTEM_ONLY_USER_TURN')
r = NS(tool_choice='none', tools=[1], messages=[NS(tools=[2])])
drop_tool_definitions(r)
check('tool-none', r.tools is None and r.messages[0].tools is None)
s = NS(tokenizer_manager=NS(model_config=NS(is_image_understandable_model=True, is_audio_understandable_model=False), mm_processor=NS(mm_tokens=NS(video_token=None, video_token_id=None)), server_args=NS(media_url_max_file_size_mb=1)))
def media(kind, url):
    return NS(messages=[NS(content=[{'type':kind, kind:{'url':url}}])])
check('media-http', _validate_media_parts(s, media('image_url','https://example.com/image')) is None)
check('media-local', 'Invalid' in _validate_media_parts(s, media('image_url','/tmp/image')))
check('media-audio', 'not supported' in _validate_media_parts(s, media('audio_url','https://example.com/audio')))
check('media-size', 'limit' in _validate_media_parts(s, media('image_url','data:image/png;base64,' + 'A' * 1500000)))

# Invoke real serving methods with tiny parser/test doubles; no GPU model is loaded.
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.utils.common import CLIENT_MEDIA_EXCEPTIONS
from PIL import Image
check('media-exceptions', OSError in CLIENT_MEDIA_EXCEPTIONS and Image.DecompressionBombError in CLIENT_MEDIA_EXCEPTIONS)
parser = NS(parse_stream_chunk=lambda delta: ('secret', 'visible'), parse_stream_end=lambda: ('tail', 'end'))
r = request(reasoning={'exclude':True})
result = OpenAIServingChat._process_reasoning_stream(None, 0, '', {0:parser}, {}, r, 'stop')
check('stream-exclude-preserves-normal', result == (None, 'visibleend'))
r.reasoning_exclude = False
result = OpenAIServingChat._process_reasoning_stream(None, 0, '', {0:parser}, {}, r, 'stop')
check('stream-normal-preserved', result == ('secrettail', 'visibleend'))
from unittest.mock import patch
import sglang.srt.entrypoints.openai.serving_chat as serving_module
serving = NS(chat_encoding_spec='deepseek_v41', reasoning_parser=None,
             _should_return_input_ids=lambda r:False, _should_return_output_ids=lambda r:False,
             _decode_response=lambda item:'visible', _effective_tools=lambda r:None,
             _tool_call_parsing_active=lambda r:False,
             _get_parsed_response_fields=lambda reasoning, tools:('secret', None))
ret = [{'text':'visible', 'meta_info':{'id':'test', 'weight_version':'test', 'finish_reason':{'type':'stop'}, 'prompt_tokens':10, 'completion_tokens':7, 'reasoning_tokens':3}}]
with patch.object(serving_module, 'get_serving', return_value=NS(enable_cache_report=False)):
    r.reasoning_exclude = True
    hidden = OpenAIServingChat._build_chat_response(serving, r, ret, 1)
    r.reasoning_exclude = False
    shown = OpenAIServingChat._build_chat_response(serving, r, ret, 1)
check('nonstream-exclude', hidden.choices[0].message.reasoning_content is None and shown.choices[0].message.reasoning_content == 'secret')
check('nonstream-accounting-preserved', hidden.usage == shown.usage and hidden.choices[0].message.content == 'visible')
print(json.dumps({'passed':len(checks), 'checks':sorted(set(checks))}))
