"""CPU regression using exact scheduler commit statements and Req methods.

AST loading avoids importing CUDA/torch just to execute these Python methods.
It does not replace finish policy, matcher, or commit ordering with a model.
"""

import ast
import copy
import logging
import re
import unittest
from array import array
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[4]
MANAGERS = ROOT / "python/sglang/srt/managers"


def compile_nodes(nodes, namespace):
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(unit), "<actual-source>", "exec"), namespace)


def load_source():
    ns = dict(re=re, logger=logging.getLogger(__name__), HTTPStatus=HTTPStatus)
    matcher = ROOT / "python/sglang/srt/utils/token_sequence_matcher.py"
    exec(compile(matcher.read_text(encoding="utf-8"), str(matcher), "exec"), ns)
    tree = ast.parse((MANAGERS / "schedule_batch.py").read_text(encoding="utf-8"))
    req = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Req")
    names = {
        "finished",
        "output_ids_through_stop",
        "_stop_match_tail_len",
        "tail_str",
        "_check_token_based_finish",
        "_locate_str_stop_finished_len",
        "_check_str_based_finish",
        "_check_vocab_boundary_finish",
        "_cap_finished_len_at_max_new_tokens",
        "update_finish_state",
        "update_reasoning_tokens",
    }
    methods = [
        n for n in req.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in methods} == names
    classes = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and (
            n.name == "BaseFinishReason" or n.name.startswith(("FINISH_", "FINISHED_"))
        )
    ]
    compile_nodes(
        classes
        + [
            ast.ClassDef(
                name="Req", bases=[], keywords=[], body=methods, decorator_list=[]
            )
        ],
        ns,
    )
    tree = ast.parse(
        (MANAGERS / "scheduler_components/batch_result_processor.py").read_text(
            encoding="utf-8"
        )
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SchedulerBatchResultProcessor"
    )
    helper = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_maybe_update_reasoning_tokens"
    )
    # Only the unrelated request-custom-parameter resolver is stubbed here.
    ns["get_request_reasoning_end_token_ids"] = lambda params, **kw: params.get(
        "test_end_ids"
    )
    compile_nodes(
        [
            ast.ClassDef(
                name="Processor",
                bases=[],
                keywords=[],
                body=[helper],
                decorator_list=[],
            )
        ],
        ns,
    )
    decode = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "process_batch_result_decode"
    )
    loop = next(
        n
        for n in decode.body
        if isinstance(n, ast.For) and ast.unparse(n.target) == "(i, req)"
    )
    # Start at the exact append/abort branch; finish/accounting run until the next abort handler.
    start = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.If)
        and n.body
        and isinstance(n.body[0], ast.Assign)
        and ast.unparse(n.body[0].targets[0]) == "req.to_finish"
    )
    end = next(
        i
        for i in range(start + 1, len(loop.body))
        if isinstance(loop.body[i], ast.If)
        and any(isinstance(n, ast.Continue) for n in loop.body[i].body)
    )
    decode_code = compile(
        ast.fix_missing_locations(
            ast.Module(body=copy.deepcopy(loop.body[start:end]), type_ignores=[])
        ),
        "<decode-commit>",
        "exec",
    )
    prefill = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "process_batch_result_prefill"
    )
    block = next(
        n
        for n in ast.walk(prefill)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "req.beam_group is not None"
        and n.orelse
        and "req.output_ids.append" in ast.unparse(n.orelse[0])
    )
    prefill_code = compile(
        ast.fix_missing_locations(
            ast.Module(body=copy.deepcopy(block.orelse), type_ignores=[])
        ),
        "<prefill-commit>",
        "exec",
    )
    return ns, decode_code, prefill_code


NSOURCE, DECODE, PREFILL = load_source()


def make_req(cap=512, stop=None, regex=None, ends=(90, 91)):
    req = NSOURCE["Req"]()
    req.output_ids = array("q")
    req.finished_len = req.finished_reason = req.to_finish = None
    req.reasoning_tokens = req._think_end_match_len = 0
    req._is_reasoning_over = False
    req._think_end_matcher = None
    req.require_reasoning = True
    req.rid = "test"
    req.decoded_text = ""
    req.vocab_size = 1000
    req.eos_token_ids = {2}
    req.grammar = None
    req.sampling_params = NS(
        max_new_tokens=cap,
        ignore_eos=False,
        stop_token_ids={2},
        stop_strs=stop or [],
        stop_regex_strs=regex or [],
        stop_str_max_len=4,
        stop_regex_max_len=4,
        custom_params={},
    )
    req.tokenizer = NS(
        eos_token_id=2,
        additional_stop_token_ids=None,
        decode=lambda ids: "".join(
            {5: "STOP", 6: "XY", 90: "<", 91: ">"}.get(i, "a") for i in ids
        ),
    )
    proc = NSOURCE["Processor"]()
    proc.model_config = NS(think_end_ids=list(ends))
    return req, proc


def step(req, proc, tokens, prefill=False):
    code = PREFILL if prefill else DECODE
    exec(
        code,
        dict(
            req=req, self=proc, next_token_id=tokens, sampling_mask_finish_reason=None
        ),
    )


class TestReasoningUsageCommit(unittest.TestCase):
    def test_exact_513_over_512(self):
        req, proc = make_req()
        step(req, proc, [10] * 510)
        step(req, proc, [10] * 3)
        self.assertEqual(len(req.output_ids), 513)
        self.assertEqual(len(req.output_ids_through_stop), 512)
        self.assertEqual(req.reasoning_tokens, 512)

    def test_short_eos_120_over_119(self):
        req, proc = make_req()
        step(req, proc, [10] * 117)
        step(req, proc, [10, 2, 10])
        self.assertEqual(len(req.output_ids), 120)
        self.assertEqual(len(req.output_ids_through_stop), 119)
        self.assertEqual(req.reasoning_tokens, 119)
        self.assertEqual(req.finished_reason.to_json(), {"type": "stop", "matched": 2})

    def test_stop_string_spans_steps(self):
        req, proc = make_req(stop=["aa"])
        step(req, proc, [10])
        step(req, proc, [10, 90, 91])
        self.assertEqual(req.reasoning_tokens, 2)
        self.assertEqual(list(req.output_ids_through_stop), [10, 10])
        self.assertFalse(req._is_reasoning_over)

    def test_finish_and_output_unchanged(self):
        for stop, regex, tokens, cap in [
            (None, None, [10, 2, 90, 91], 9),
            (["STOP"], None, [10, 5, 90, 91], 9),
            (None, ["XY"], [10, 6, 90, 91], 9),
            (None, None, [10, 10, 2], 2),
            (None, None, [10, 10, 10], 2),
            (None, None, [10, 1001, 90, 91], 9),
        ]:
            with self.subTest(tokens=tokens, stop=stop, regex=regex, cap=cap):
                req, proc = make_req(cap, stop, regex)
                reference = copy.deepcopy(req)
                reference.output_ids.extend(tokens)
                reference.update_finish_state(len(tokens))
                step(req, proc, tokens)
                self.assertEqual(req.output_ids, reference.output_ids)
                self.assertEqual(
                    req.output_ids_through_stop, reference.output_ids_through_stop
                )
                self.assertEqual(
                    req.finished_reason.to_json(), reference.finished_reason.to_json()
                )
                self.assertEqual(req.reasoning_tokens, len(req.output_ids_through_stop))
                self.assertFalse(req._is_reasoning_over)

    def test_multi_token_end_across_steps(self):
        req, proc = make_req()
        step(req, proc, [10, 90])
        self.assertEqual(req._think_end_match_len, 1)
        step(req, proc, [91, 10, 10])
        self.assertTrue(req._is_reasoning_over)
        self.assertEqual(req.reasoning_tokens, 3)
        step(req, proc, [10])
        self.assertEqual(req.reasoning_tokens, 3)

    def test_end_outside_cap_does_not_advance_matcher(self):
        req, proc = make_req(cap=2)
        step(req, proc, [10, 90, 91])
        self.assertEqual(req.reasoning_tokens, 2)
        self.assertEqual(req._think_end_match_len, 1)
        self.assertFalse(req._is_reasoning_over)

    def test_end_at_cap_is_counted(self):
        req, proc = make_req(cap=3)
        step(req, proc, [10, 90, 91, 10])
        self.assertEqual(req.reasoning_tokens, 3)
        self.assertTrue(req._is_reasoning_over)

    def test_prefill_zero_and_one(self):
        for cap in [0, 1, 2]:
            req, proc = make_req(cap=cap)
            step(req, proc, 10, prefill=True)
            self.assertEqual(req.reasoning_tokens, min(cap, 1))
            self.assertEqual(len(req.output_ids), 1)

    def test_opt_out_and_request_selected_end(self):
        req, proc = make_req()
        req.require_reasoning = False
        step(req, proc, [10, 10])
        self.assertEqual(req.reasoning_tokens, 0)
        req, proc = make_req()
        req.sampling_params.custom_params = {"test_end_ids": [8, 9]}
        step(req, proc, [8, 9, 10])
        self.assertEqual(req.reasoning_tokens, 2)

    def test_opt_out_does_not_slice_retained_tokens(self):
        class NoRetainedSlice(list):
            def __getitem__(self, key):
                # Finish resolution legitimately reads the new run via [-n:].
                # The added accounting slice has a nonnegative absolute start.
                if isinstance(key, slice) and key.start is not None and key.start >= 0:
                    raise AssertionError("unexpected retained-token allocation")
                return super().__getitem__(key)

        for prefill in [False, True]:
            with self.subTest(prefill=prefill):
                req, proc = make_req()
                req.require_reasoning = False
                req.output_ids = NoRetainedSlice()
                step(req, proc, 10 if prefill else [10, 10], prefill=prefill)
                self.assertEqual(req.reasoning_tokens, 0)

    def test_streaming_and_nonstream_usage(self):
        tree = ast.parse(
            (
                ROOT / "python/sglang/srt/entrypoints/openai/usage_processor.py"
            ).read_text(encoding="utf-8")
        )
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "UsageProcessor"
        )
        cls.decorator_list = []
        ns = dict(UsageInfo=NS, PromptTokensDetails=NS)
        compile_nodes([cls], ns)
        usage = ns["UsageProcessor"]
        req, proc = make_req(cap=4)
        for tokens in [[10, 10], [10, 10, 10]]:
            step(req, proc, tokens)
            counts = dict(
                prompt_tokens=8,
                completion_tokens=len(req.output_ids_through_stop),
                reasoning_tokens=req.reasoning_tokens,
            )
            unary = usage.calculate_response_usage([dict(meta_info=counts)])
            stream = usage.calculate_streaming_usage(
                {0: 8},
                {0: req.reasoning_tokens},
                {0: len(req.output_ids_through_stop)},
                {0: 0},
                1,
            )
            self.assertEqual(vars(unary), vars(stream))
            self.assertEqual(
                unary.completion_tokens_details["reasoning_tokens"],
                counts["completion_tokens"],
            )

    def test_sampling_mask_abort_does_not_count(self):
        req, proc = make_req()
        exec(
            DECODE,
            dict(
                req=req,
                self=proc,
                next_token_id=[10],
                sampling_mask_finish_reason=NSOURCE["FINISH_ABORT"]("mask"),
            ),
        )
        self.assertEqual(req.reasoning_tokens, 0)
        self.assertEqual(list(req.output_ids), [])

    def test_overlapping_terminator_and_normal_single_tokens(self):
        req, proc = make_req(ends=(90, 90, 91))
        for token in [10, 90, 90, 90, 91, 10]:
            step(req, proc, [token])
        self.assertEqual(req.reasoning_tokens, 5)
        self.assertTrue(req._is_reasoning_over)


if __name__ == "__main__":
    unittest.main()
