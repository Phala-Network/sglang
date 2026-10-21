"""Focused .22 migration checks for existing Xing service contracts (CPU only)."""
import json
import pickle
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
import pytest
import xgrammar as xgr
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.function_call.utils import normalize_json_schema_types
from sglang.srt.function_call.xing4_native_tool_schema import NativeToolSchemaTag
from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.constrained.json_structural_grace import JsonStructuralGrace, reserve_json_structural_grace
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang.srt.sampling.sampling_params import SamplingParams


def test_nullable_schema_keywords_retain_original_meaning():
    schema={"type":"object","properties":{"x":{"type":"object","properties":None,"required":None}},"required":None}
    normalize_json_schema_types(schema)
    assert "required" not in schema and schema["properties"]["x"] == {"type":"object"}


@pytest.mark.parametrize("fields,excluded,thinking", [
    ({"include_reasoning":False}, True, None),
    ({"include_reasoning":True}, False, True),
    ({"reasoning":{"exclude":True,"enabled":False}}, True, False),
    ({"include_reasoning":False,"reasoning":{"exclude":False,"effort":"high"}}, False, True),
])
def test_reasoning_visibility_is_independent_of_model_thinking(fields, excluded, thinking):
    req=ChatCompletionRequest(model="fixture",messages=[{"role":"user","content":"x"}],**fields)
    assert req.reasoning_exclude is excluded
    assert (req.chat_template_kwargs or {}).get("thinking") is thinking
    assert "reasoning_exclude" not in req.model_dump()


@pytest.mark.parametrize("tag,native", [(NativeToolSchemaTag("{}"),True),("{}",False)])
def test_cached_invalid_native_assistance_fallback_requires_server_provenance(tag,native):
    tag=pickle.loads(pickle.dumps(tag))
    manager=object.__new__(GrammarManager)
    manager._enable_strict_thinking=False
    manager.grammar_queue=[]
    manager.grammar_backend=NS(get_cached_or_future_value=lambda *_:(InvalidGrammarObject("fixture"),True))
    req=NS(sampling_params=SamplingParams(structural_tag=tag),grammar=None,grammar_key=None,
        grammar_wait_ct=0,require_reasoning=False,set_finish_with_abort=MagicMock())
    assert manager.process_req_with_grammar(req) is False
    if native:
        assert req.sampling_params.structural_tag is None and req.grammar is None
        req.set_finish_with_abort.assert_not_called()
    else:
        req.set_finish_with_abort.assert_called_once()
        assert req.sampling_params.structural_tag == "{}"


def grammar(schema):
    vocab=[bytes([i]) for i in range(128)]+[b"<eos>"]
    info=xgr.TokenizerInfo(vocab,stop_token_ids=[128])
    ctx=xgr.GrammarCompiler(info).compile_json_schema(json.dumps(schema))
    return XGrammarGrammar(xgr.GrammarMatcher(ctx),129,ctx,None,json.dumps(schema))


def test_structural_grace_closes_only_a_proven_value_and_real_eos():
    g=grammar({"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":False})
    prefix=b'{"value":"ok'
    for token in prefix: g.accept_token(token)
    grace=JsonStructuralGrace(len(prefix))
    plan=grace.plan_for(g)
    assert plan is not None and plan[-1] == 128
    assert b"".join(g.ctx.tokenizer_info.decoded_vocab[t] for t in plan[:-1]) == b'"}'
    for token in plan: g.accept_token(token)
    assert g.matcher.is_completed() and g.matcher.is_terminated()


def test_structural_grace_never_supplies_a_missing_value():
    g=grammar({"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":False})
    prefix=b'{"value":'
    for token in prefix: g.accept_token(token)
    assert JsonStructuralGrace(len(prefix)).plan_for(g) is None


def test_grace_reservation_is_bounded_idempotent_and_falls_back_at_context_edge():
    def request():
        return NS(sampling_params={"json_schema":"{}","max_new_tokens":8},require_reasoning=False,
            session_id=None,session_params=None)
    args=NS(grammar_backend="xgrammar",disaggregation_mode="null")
    req=request()
    reserve_json_structural_grace(req,16,100,args,NS())
    reserve_json_structural_grace(req,16,100,args,NS())
    assert req.sampling_params["max_new_tokens"] == 40
    assert req.sampling_params["json_structural_grace_base_max_new_tokens"] == 8
    edge=request()
    reserve_json_structural_grace(edge,90,100,args,NS())
    assert edge.sampling_params["max_new_tokens"] == 8
    assert "json_structural_grace_base_max_new_tokens" not in edge.sampling_params


@pytest.mark.parametrize("byte", list(range(32)))
def test_original_qualified_xgrammar_rejects_literal_controls(byte):
    g=grammar({"type":"string","minLength":1,"maxLength":32})
    assert not g.matcher.accept_string('"a'+chr(byte)+'b"')
