"""CPU regression coverage for parallel-sampling request-state ownership."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402


register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _make_tokenizer_manager() -> TokenizerManager:
    """Build only the state needed by the real owner and batch lifecycle."""
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.enable_trace = False
    manager.disaggregation_mode = "none"
    manager.rid_to_state = {}
    manager.encoder_dispatch_ready = {}
    return manager


class TestParallelSamplingOwnerLifecycle(CustomTestCase):
    def test_n2_registers_only_initial_owners_and_releases_all_states(self):
        """Real normalization and batch fan-out leave no parent or child owner."""
        cases = (
            ("none", None),
            ("string", "client-rid"),
            ("list", None),
        )

        for rid_kind, supplied_rid in cases:
            for prompt_count, text in ((1, "one"), (2, ["one", "two"])):
                for stream in (False, True):
                    with self.subTest(
                        rid_kind=rid_kind,
                        prompt_count=prompt_count,
                        stream=stream,
                    ):
                        if rid_kind == "list":
                            supplied_rid = [
                                f"client-rid-{index}"
                                for index in range(prompt_count)
                            ]
                        obj = GenerateReqInput(
                            text=text,
                            rid=supplied_rid,
                            sampling_params={"n": 2},
                            stream=stream,
                        )
                        obj.normalize_batch_and_arguments()

                        self.assertFalse(obj.is_single)
                        self.assertEqual(obj.batch_size, prompt_count)
                        self.assertEqual(obj.parallel_sample_num, 2)
                        if rid_kind == "list":
                            self.assertEqual(obj.rid, supplied_rid)
                        else:
                            self.assertEqual(len(obj.rid), prompt_count * 2)

                        manager = _make_tokenizer_manager()
                        manager._init_req_state(obj)

                        # This is the regression assertion. Before the fix,
                        # None/string RIDs registered prompt_count * n owners.
                        self.assertEqual(
                            len(manager.rid_to_state),
                            prompt_count,
                            "n2 initial-owner cardinality must equal the original "
                            "prompt count",
                        )
                        self.assertEqual(
                            set(manager.rid_to_state),
                            set(obj.rid[:prompt_count]),
                        )

                        async def tokenize(sub_obj):
                            return SimpleNamespace(
                                input_ids=[1, 2],
                                mm_inputs=None,
                                rid=sub_obj.rid,
                                sampling_params=SimpleNamespace(max_new_tokens=8),
                                stream=sub_obj.stream,
                            )

                        dispatched_rids = []

                        async def dispatch(tokenized_obj):
                            state = manager.rid_to_state[tokenized_obj.rid]
                            state.dispatched = True
                            dispatched_rids.append(tokenized_obj.rid)

                        def deliver_finished_response(sub_obj, _request=None):
                            async def response():
                                state = manager.rid_to_state.pop(sub_obj.rid)
                                state.finished = True
                                yield {
                                    "text": "done",
                                    "meta_info": {
                                        "id": sub_obj.rid,
                                        "finish_reason": {"type": "length"},
                                    },
                                }

                            return response()

                        # The lifecycle under test is real. These three seams
                        # stand in only for tokenizer work, scheduler dispatch,
                        # and scheduler output delivery.
                        manager._tokenize_one_request = AsyncMock(side_effect=tokenize)
                        manager._send_one_request = AsyncMock(side_effect=dispatch)
                        manager._wait_one_response = deliver_finished_response

                        async def drive_lifecycle():
                            return [
                                output
                                async for output in manager._handle_batch_request(
                                    obj, request_rids=set(obj.rid)
                                )
                            ]

                        outputs = asyncio.run(drive_lifecycle())

                        self.assertEqual(
                            manager._tokenize_one_request.await_count, prompt_count
                        )
                        # One prefix-cache request and n child requests per prompt.
                        self.assertEqual(len(dispatched_rids), prompt_count * 3)
                        self.assertEqual(
                            manager._send_one_request.await_count, prompt_count * 3
                        )
                        self.assertEqual(len(set(dispatched_rids)), prompt_count * 3)
                        if stream:
                            self.assertEqual(len(outputs), prompt_count * 2)
                            self.assertEqual(
                                {output["index"] for output in outputs},
                                set(range(prompt_count * 2)),
                            )
                        else:
                            self.assertEqual(len(outputs), 1)
                            self.assertEqual(len(outputs[0]), prompt_count * 2)
                        self.assertFalse(manager.rid_to_state)

    def test_real_normalization_still_rejects_duplicate_supplied_rids(self):
        obj = GenerateReqInput(
            text=["one", "two"],
            rid=["duplicate", "duplicate"],
            sampling_params={"n": 2},
        )

        with self.assertRaisesRegex(ValueError, "Duplicate request IDs"):
            obj.normalize_batch_and_arguments()


if __name__ == "__main__":
    unittest.main(verbosity=2)
