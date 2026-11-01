"""CPU-only regression of K3 JSON grammar and parser boundary agreement."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import xgrammar as xgr
from transformers import AutoTokenizer

from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarBackend
from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.srt.parser.reasoning_parser import KimiK3Detector
from sglang.srt.function_call.kimik3_format import (
    THINK_CLOSE, RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE, TOOLS_OPEN,
)

class Tokenizer:
    def __init__(self):
        self.vocab = [bytes([i]) for i in range(256)] + [
            x.encode() for x in ('<|open|>', '<|close|>', '<|sep|>', '[EOS]', '<|end_of_msg|>', '[EOT]', 'think', 'tools', 'message', 'response')
        ]
        self.crossing_valid = len(self.vocab)
        self.vocab.append((RESPONSE_OPEN+'{"ok":true}').encode())
        self.crossing_invalid = len(self.vocab)
        self.vocab.append((RESPONSE_OPEN+'{"ok":false}').encode())
        self.vocab_size = len(self.vocab)
        self.eos_token_id = 259
        self.model_eos = 260
        self.additional_stop_token_ids = None
        self.model = SimpleNamespace(decode_single_token_bytes=lambda i: self.vocab[i])
    def encode(self, text, add_special_tokens=False):
        data = text.encode(); result = []
        while data:
            token = next((i for i in range(256,len(self.vocab)) if data.startswith(self.vocab[i])), data[0])
            result.append(token); data = data[len(self.vocab[token]):]
        return result
    def init_xgrammar(self):
        return xgr.TokenizerInfo(self.vocab, stop_token_ids=[self.model_eos]), None
    def decode(self, ids, **kwargs):
        return b''.join(self.vocab[i] for i in ids).decode()

@pytest.fixture(scope='module', params=['synthetic', 'production'])
def engine(request):
    if request.param == 'production':
        path = os.environ.get('KIMIK3_TOKENIZER_DIR')
        if not path:
            pytest.skip('Pinned tokenizer path required for production-tokenizer CPU gate')
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        tok.model_eos = json.loads((Path(path)/'config.json').read_text())['eos_token_id']
        tok.additional_stop_token_ids = None
    else:
        tok = Tokenizer()
    return tok, XGrammarGrammarBackend(tok, tok.vocab_size, model_eos_token_ids=[tok.model_eos])

SCHEMA = {'type':'object','properties':{'ok':{'const':True}},'required':['ok'],'additionalProperties':False}

def reasoner(engine, detector=None):
    return ReasonerGrammarBackend(engine[1], SimpleNamespace(detector=detector or KimiK3Detector()), engine[0])

def make(engine, thinking=True, schema=None):
    return reasoner(engine)._init_value_dispatch(('json',json.dumps(schema or SCHEMA)),thinking)

def mask(obj, engine):
    bits=obj.allocate_vocab_mask(engine[0].vocab_size,1,'cpu');bits.fill_(-1);obj.fill_vocab_mask(bits,0);return bits

def allowed(bits, token):
    return bool(int(bits[0,token//32]) & (1 << (token%32)))

def feed(obj, engine, text):
    ids=engine[0].encode(text,add_special_tokens=False) if isinstance(text,str) else text
    for token in ids:
        assert allowed(mask(obj,engine),token), ('masked',token)
        obj.accept_token(token)
    return ids

def rejects(obj, engine, text):
    ids=engine[0].encode(text,add_special_tokens=False) if isinstance(text,str) else text
    for token in ids:
        if not allowed(mask(obj,engine),token):return True
        obj.accept_token(token)
    return False

@pytest.mark.parametrize('payload',[
    'analysis'+RESPONSE_OPEN+'plain answer'+RESPONSE_CLOSE+MESSAGE_CLOSE,
    'plain answer'+RESPONSE_CLOSE+MESSAGE_CLOSE,
    'analysis'+TOOLS_OPEN+'plain answer',
])
def test_no_implicit_boundary_bypass(engine,payload):
    obj=make(engine)
    assert rejects(obj,engine,payload), 'unconstrained final content bypassed JSON grammar'

@pytest.mark.parametrize('schema',[{'type':'object'},SCHEMA])
def test_eos_not_allowed_during_reasoning(engine,schema):
    obj=make(engine,schema=schema);feed(obj,engine,'analysis')
    assert not allowed(mask(obj,engine),engine[0].model_eos)
    assert not allowed(mask(obj,engine),engine[0].eos_token_id)

@pytest.mark.parametrize('thinking,prefix,suffix',[
    (True,'analysis'+THINK_CLOSE,''),
    (True,'analysis'+THINK_CLOSE+RESPONSE_OPEN,RESPONSE_CLOSE+MESSAGE_CLOSE),
    (True,'analysis'+RESPONSE_OPEN,RESPONSE_CLOSE+MESSAGE_CLOSE),
    (False,'',''),
    (False,RESPONSE_OPEN,RESPONSE_CLOSE+MESSAGE_CLOSE),
])
def test_valid_native_json(engine,thinking,prefix,suffix):
    native=prefix+'{"ok":true}'+suffix
    obj=make(engine,thinking);feed(obj,engine,native)
    feed(obj,engine,[engine[0].model_eos]);assert obj.is_terminated()
    parsed=KimiK3Detector(force_reasoning=thinking,structured_json=True).detect_and_parse(native)
    assert json.loads(parsed.normal_text)=={'ok':True}
    for width in (1,7,1000):
        detector=KimiK3Detector(force_reasoning=thinking,structured_json=True)
        content=''.join((detector.parse_streaming_increment(native[i:i+width]).normal_text or '') for i in range(0,len(native),width))
        assert json.loads(content)=={'ok':True}

@pytest.mark.parametrize('prefix',[THINK_CLOSE,RESPONSE_OPEN])
def test_schema_stays_active(engine,prefix):
    obj=make(engine);feed(obj,engine,'analysis'+prefix)
    assert rejects(obj,engine,'{"ok":false}'+RESPONSE_CLOSE)


def test_rollback_and_copy(engine):
    obj=make(engine);feed(obj,engine,'analysis'+RESPONSE_OPEN)
    ids=feed(obj,engine,'{"ok":true}')
    obj.rollback(len(ids));clone=obj.copy()
    assert rejects(clone,engine,'{"ok":false}')
    feed(obj,engine,'{"ok":true}'+RESPONSE_CLOSE+MESSAGE_CLOSE)
    feed(obj,engine,[engine[0].model_eos]);assert obj.is_terminated()

@pytest.mark.parametrize('first,second',[(True,False),(False,True)])
def test_cache_thinking_modes_are_isolated(engine,first,second):
    backend=reasoner(engine);key=('json',json.dumps(SCHEMA))
    initial=backend._init_value_dispatch(key,first);backend.set_cache(key,initial)
    obj,hit=backend.get_cached_or_future_value(key,second)
    assert not hit  # Each native thinking mode has its own lazily compiled key.
    future=obj;obj=future.result(timeout=10)
    backend.set_cache(key,obj.copy(),second,future=future)
    backend.release_future(future)
    cached,hit=backend.get_cached_or_future_value(key,second)
    assert hit and cached.grammar.ctx is obj.grammar.ctx
    if second:feed(obj,engine,'analysis'+RESPONSE_OPEN)
    else:assert rejects(obj.copy(),engine,'analysis')
    feed(obj,engine,'{"ok":true}'+RESPONSE_CLOSE+MESSAGE_CLOSE)
    feed(obj,engine,[engine[0].model_eos]);assert obj.is_terminated()


def test_alias_token_boundaries(engine):
    if not isinstance(engine[0],Tokenizer):pytest.skip('Synthetic byte alias control')
    obj=make(engine);feed(obj,engine,list(('analysis'+RESPONSE_OPEN).encode()))
    assert rejects(obj,engine,'{"ok":false}')

def test_one_token_crossing_boundary_is_constrained(engine):
    tok=engine[0]
    if not isinstance(tok,Tokenizer):pytest.skip('Synthetic crossing-token control')
    obj=make(engine)
    assert not allowed(mask(obj,engine),tok.crossing_invalid)
    feed(obj,engine,[tok.crossing_valid])
    feed(obj,engine,RESPONSE_CLOSE+MESSAGE_CLOSE)
    feed(obj,engine,[tok.model_eos]);assert obj.is_terminated()

def test_json_looking_reasoning_needs_explicit_boundary(engine):
    obj=make(engine)
    reasoning='{"note":"reasoning only"}'
    assert rejects(obj.copy(),engine,reasoning+RESPONSE_CLOSE+MESSAGE_CLOSE)
    native=reasoning+THINK_CLOSE+'{"ok":true}'
    feed(obj,engine,native);feed(obj,engine,[engine[0].model_eos])
    parsed=KimiK3Detector(force_reasoning=True,structured_json=True).detect_and_parse(native)
    assert parsed.reasoning_text==reasoning
    assert json.loads(parsed.normal_text)=={'ok':True}


def test_non_k3_keeps_existing_reasoning_boundary(engine):
    detector=SimpleNamespace(think_end_token=THINK_CLOSE,think_excluded_tokens=[],
        get_think_end_token_ids=lambda tok: tok.encode(THINK_CLOSE,add_special_tokens=False),
        get_answer_frame=lambda: None)
    backend=reasoner(engine,detector)
    obj=backend._init_value_dispatch(('json',json.dumps(SCHEMA)),True)
    feed(obj,engine,'ordinary reasoning')
    assert obj._kimik3_token_bytes is None
    feed(obj,engine,THINK_CLOSE+'{"ok":true}')
    feed(obj,engine,[engine[0].model_eos]);assert obj.is_terminated()

@pytest.mark.parametrize('thinking',[True,False])
def test_truncation_cannot_become_success(engine,thinking):
    obj=make(engine,thinking)
    if thinking:feed(obj,engine,THINK_CLOSE)
    feed(obj,engine,'{"ok":')
    assert not allowed(mask(obj,engine),engine[0].model_eos)
    assert not obj.is_terminated()


def test_cache_schema_identity(engine):
    backend=reasoner(engine)
    key=('json',json.dumps(SCHEMA));entry=backend._init_value_dispatch(key,True)
    backend.set_cache(key,entry)
    changed=('json',json.dumps({'type':'object','properties':{'ok':{'const':False}},'required':['ok'],'additionalProperties':False}))
    other,hit=backend.get_cached_or_future_value(changed,False)
    assert not hit
    obj=other.result(timeout=60)
    assert rejects(obj.copy(),engine,'{"ok":true}')
    feed(obj,engine,'{"ok":false}')
    feed(obj,engine,[engine[0].model_eos]);assert obj.is_terminated()


def test_request_thinking_budget_keeps_json_complete(engine):
    tok,base=engine
    backend=ReasonerGrammarBackend(base,SimpleNamespace(detector=KimiK3Detector()),tok,enable_strict_thinking=True)
    obj=backend._init_value_dispatch(('json',json.dumps(SCHEMA)),True)
    obj.max_think_tokens=0
    feed(obj,engine,THINK_CLOSE+'{"ok":true}')
    feed(obj,engine,[tok.model_eos]);assert obj.is_terminated()

def test_fixed_whitespace_is_explicitly_unsupported_without_schema_disclosure():
    tok=Tokenizer()
    base=XGrammarGrammarBackend(tok,tok.vocab_size,model_eos_token_ids=[tok.model_eos],any_whitespace=False)
    backend=ReasonerGrammarBackend(base,SimpleNamespace(detector=KimiK3Detector()),tok)
    schema={'type':'object','description':'PRIVATE_SCHEMA_SENTINEL'}
    obj=backend._init_value_dispatch(('json',json.dumps(schema)),True)
    assert isinstance(obj,InvalidGrammarObject)
    assert 'any_whitespace=True' in obj.error_message and 'unsupported' in obj.error_message
    assert 'PRIVATE_SCHEMA_SENTINEL' not in obj.error_message

@pytest.mark.parametrize('first',[True,False])
def test_cached_json_init_reuses_boundary_metadata_and_stop_overrides(engine,monkeypatch,first):
    tok=engine[0]
    encode=tok.encode
    # K3's public encode may recursively call self.encode. Prepare fixture IDs
    # before instrumenting it so only product initialization is counted.
    native_ids={
        thinking: encode(
            ('analysis'+RESPONSE_OPEN if thinking else '')+'{"ok":true}',
            add_special_tokens=False,
        )
        for thinking in (False,True)
    }
    calls=[]
    def counted(text,**kwargs):
        calls.append(text)
        return encode(text,**kwargs)
    monkeypatch.setattr(tok,'encode',counted)
    backend=reasoner(engine)
    key=('json',json.dumps(SCHEMA))
    cached=backend._init_value_dispatch(key,first)
    assert not isinstance(cached,InvalidGrammarObject)
    backend.set_cache(key,cached)
    count=len(calls)
    assert count>1
    stops=set(cached.grammar.override_stop_tokens)
    assert {tok.eos_token_id,tok.model_eos}.issubset(stops)
    other_mode,hit=backend.get_cached_or_future_value(key,not first)
    assert not hit
    future=other_mode
    other_mode=future.result(timeout=30)
    assert not isinstance(other_mode,InvalidGrammarObject)
    backend.set_cache(key,other_mode.copy(),not first,future=future)
    backend.release_future(future)
    # Boundary metadata belongs to the tokenizer and is shared across modes.
    assert len(calls)==count
    for thinking in (first,not first,first):
        obj,hit=backend.get_cached_or_future_value(key,thinking)
        assert hit and len(calls)==count
        assert obj.grammar.matcher is not cached.grammar.matcher
        assert set(obj.grammar.override_stop_tokens)==stops
        matcher=obj.grammar.matcher
        obj.maybe_init_reasoning(thinking)
        assert obj.grammar.matcher is matcher
        assert len(calls)==count
        feed(obj,engine,native_ids[thinking])
        feed(obj,engine,[tok.model_eos]);assert obj.is_terminated()
        obj.rollback(1)
        assert not obj.is_terminated()
        feed(obj,engine,[tok.model_eos]);assert obj.is_terminated()
        with pytest.raises(ValueError,match='thinking mode'):
            obj.maybe_init_reasoning(not thinking)
        obj.maybe_init_reasoning(thinking)
        assert not obj.is_terminated() and not obj.grammar.accepted_tokens
        assert set(obj.grammar.override_stop_tokens)==stops
        assert len(calls)==count
        assert not cached.grammar.accepted_tokens and cached.current_token is None
    # A different schema on the same backend has identical native boundaries.
    other=backend._init_value_dispatch(('json',json.dumps(dict(SCHEMA,description='second schema'))),first)
    assert not isinstance(other,InvalidGrammarObject)
    assert len(calls)==count
    assert other.grammar.matcher is not cached.grammar.matcher
