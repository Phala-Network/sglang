from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor


def test_reasoning_tokens_use_openai_completion_details_shape():
    usage = UsageProcessor.calculate_token_usage(
        prompt_tokens=10,
        completion_tokens=30,
        reasoning_tokens=17,
    )

    assert usage.reasoning_tokens == 17
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 17
    assert usage.model_dump()["completion_tokens_details"] == {
        "reasoning_tokens": 17
    }


def test_zero_reasoning_tokens_keep_completion_details_absent():
    usage = UsageProcessor.calculate_token_usage(
        prompt_tokens=10,
        completion_tokens=30,
        reasoning_tokens=0,
    )

    assert usage.reasoning_tokens == 0
    assert usage.completion_tokens_details is None


def test_non_streaming_usage_aggregates_reasoning_tokens_in_both_shapes():
    usage = UsageProcessor.calculate_response_usage(
        responses=[
            {
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "reasoning_tokens": 7,
                }
            },
            {
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 30,
                    "reasoning_tokens": 11,
                }
            },
        ],
        n_choices=2,
    )

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 50
    assert usage.reasoning_tokens == 18
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 18


def test_streaming_usage_aggregates_reasoning_tokens_in_both_shapes():
    usage = UsageProcessor.calculate_streaming_usage(
        prompt_tokens={0: 10, 1: 10},
        reasoning_tokens={0: 7, 1: 11},
        completion_tokens={0: 20, 1: 30},
        cached_tokens={0: 0, 1: 0},
        n_choices=2,
    )

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 50
    assert usage.reasoning_tokens == 18
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 18
