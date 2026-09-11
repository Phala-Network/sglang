"""Regression for upstream #37450 and neighbouring stop/think boundaries."""
import pytest

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams


@pytest.mark.parametrize("budget,run,expected_output,expected_reasoning", [
    (4, [10, 11, 12, 13, 14, 15], 4, 4),
    (16, [10, 11, 12, 13, 14, 15], 6, 6),
    (4, [10, 7, 8, 13, 14, 15], 4, 3),
    (4, [10, 11, 12, 13, 7, 8], 4, 4),
    (4, [10, 99, 12, 13, 14, 15], 2, 2),
    (4, [10, 11, 12, 13, 99, 15], 4, 4),
])
def test_reasoning_follows_emitted_prefix(budget, run, expected_output, expected_reasoning):
    sampling = SamplingParams(max_new_tokens=budget, temperature=0)
    sampling.normalize(None)
    req = Req(rid="usage", origin_input_text="", origin_input_ids=[1, 2, 3], sampling_params=sampling)
    req.eos_token_ids = {99}
    req.output_ids.extend(run)
    req.update_reasoning_tokens(run, [7, 8])
    req.update_finish_state(len(run))
    assert len(req.output_ids_through_stop) == expected_output
    assert req.reasoning_tokens == expected_reasoning
    assert req.reasoning_tokens <= len(req.output_ids_through_stop)


def test_reasoning_prefix_from_earlier_step_is_preserved():
    sampling = SamplingParams(max_new_tokens=6, temperature=0)
    sampling.normalize(None)
    req = Req(rid="usage-multistep", origin_input_text="", origin_input_ids=[1], sampling_params=sampling)
    req.eos_token_ids = {99}
    for run in [[10, 11, 7], [8, 12, 13, 14, 15]]:
        req.output_ids.extend(run)
        req.update_reasoning_tokens(run, [7, 8])
        req.update_finish_state(len(run))
    assert len(req.output_ids_through_stop) == 6
    assert req.reasoning_tokens == 4
