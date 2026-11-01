"""CPU usage regressions: real Req and scheduler helper, no model execution."""
import os
from array import array
from types import SimpleNamespace

import pytest

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.function_call.kimik3_format import MESSAGE_CLOSE, RESPONSE_CLOSE, RESPONSE_OPEN, THINK_CLOSE, THINK_OPEN, TOOLS_OPEN
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import SchedulerBatchResultProcessor
from sglang.srt.parser.reasoning_parser import KimiK3Detector
from sglang.srt.sampling.sampling_params import (
    SamplingParams,
)


class FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids):
        return bytes(list(ids)).decode(errors="replace")


def make_req(tokenizer=None, cap=10000, stop=None, require=True):
    tokenizer = tokenizer or FakeTokenizer()
    params = SamplingParams(max_new_tokens=cap, stop=stop)
    params.normalize(tokenizer)
    req = Req(rid="usage-cpu", origin_input_text="", origin_input_ids=array("q", [0]),
              sampling_params=params, eos_token_ids=set(), vocab_size=1000000,
              require_reasoning=require)
    req.tokenizer = tokenizer
    return req


def processor(sequences=None):
    obj = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)
    object.__setattr__(obj, "model_config", SimpleNamespace(
        think_end_ids=[90, 91], reasoning_end_id_sequences=sequences))
    return obj


def step(req, obj, batch):
    req.output_ids.extend(batch)
    req.update_finish_state(len(batch))
    obj._maybe_update_retained_reasoning_tokens(req, len(batch))


@pytest.mark.parametrize("cap", [0, 1, 2, 3])
def test_length_drops_tokens_and_matcher_state(cap):
    req = make_req(cap=cap)
    step(req, processor(), [65, 66, 90, 91])
    assert req.reasoning_tokens == len(req.output_ids_through_stop) == cap
    assert not req._is_reasoning_over


@pytest.mark.parametrize("stop", ["AB", ["AB"]])
def test_string_stop_cross_batch(stop):
    req, obj = make_req(stop=stop), processor()
    step(req, obj, [65])
    step(req, obj, [66, 90, 91])
    assert req.finished_len == req.reasoning_tokens == 2
    assert not req._is_reasoning_over


@pytest.mark.parametrize("boundary", [[90, 91], [70, 71], [80, 81]])
@pytest.mark.parametrize("chunk_size", [1, 2, 5])
def test_k3_boundaries_across_batches(boundary, chunk_size):
    req, obj = make_req(), processor([[90, 91], [70, 71], [80, 81]])
    tokens = [65] + boundary + [66, 67]
    for start in range(0, len(tokens), chunk_size):
        step(req, obj, tokens[start:start + chunk_size])
    assert req.reasoning_tokens == 3
    assert req._is_reasoning_over


def test_k3_partial_marker_discarded():
    req, obj = make_req(cap=2), processor([[90, 91], [70, 71], [80, 81]])
    step(req, obj, [65, 70, 71, 66])
    assert req.reasoning_tokens == 2
    assert not req._is_reasoning_over
    assert req._reasoning_end_match_lengths == [0, 1, 0]


def test_k3_thinking_disabled():
    req = make_req(require=False)
    step(req, processor([[90, 91], [70, 71], [80, 81]]), [65, 70, 71, 66])
    assert req.reasoning_tokens == 0
    assert req._reasoning_end_matchers is None


def test_normal_model_remains_single_marker():
    req = make_req()
    step(req, processor(), [65, 70, 71, 66])
    assert req.reasoning_tokens == 4
    assert not req._is_reasoning_over


@pytest.fixture(scope="module")
def k3_tokenizer():
    path = os.environ.get("KIMIK3_TOKENIZER_DIR")
    if not path:
        pytest.skip("real K3 tokenizer directory not supplied")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    tokenizer.additional_stop_token_ids = None
    return tokenizer


@pytest.mark.parametrize("marker", [THINK_CLOSE, RESPONSE_OPEN, TOOLS_OPEN])
@pytest.mark.parametrize("chunk_size", [1, 3, 1000])
def test_real_k3_tokenizer_boundaries(k3_tokenizer, marker, chunk_size):
    encode = lambda text: k3_tokenizer.encode(text, add_special_tokens=False)
    sequences = [encode(m) for m in (THINK_CLOSE, RESPONSE_OPEN, TOOLS_OPEN)]
    prefix, suffix = "Reasoning.", "Answer."
    # Encode the whole native sequence, not independently concatenated words.
    native = prefix + marker + suffix
    tokens = encode(native)
    boundary_end = next(i for i in range(1, len(tokens) + 1)
                        if marker in k3_tokenizer.decode(tokens[:i], skip_special_tokens=False))
    req, obj = make_req(k3_tokenizer), processor(sequences)
    obj.model_config.think_end_ids = sequences[0]
    for start in range(0, len(tokens), chunk_size):
        step(req, obj, tokens[start:start + chunk_size])
    assert req.reasoning_tokens == boundary_end
    assert req._is_reasoning_over
    parsed = KimiK3Detector(force_reasoning=True).detect_and_parse(native)
    assert parsed.reasoning_text == prefix
    assert suffix in parsed.normal_text


def test_literal_marker_matches_parser_boundary(k3_tokenizer):
    native = 'quoted "' + RESPONSE_OPEN + '" text'
    # In the thinking channel this literal is a protocol boundary in the parser,
    # unlike markers in an already-entered JSON answer string.
    parsed = KimiK3Detector(force_reasoning=True).detect_and_parse(native)
    assert parsed.reasoning_text == 'quoted "'
    encode = lambda text: k3_tokenizer.encode(text, add_special_tokens=False)
    req = make_req(k3_tokenizer)
    obj = processor([encode(m) for m in (THINK_CLOSE, RESPONSE_OPEN, TOOLS_OPEN)])
    step(req, obj, encode(native))
    assert req._is_reasoning_over
    assert req.reasoning_tokens < len(req.output_ids)


