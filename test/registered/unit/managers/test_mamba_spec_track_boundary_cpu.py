"""CPU boundary regressions for the actual Mamba speculative checkpoint method.

Load only the pure method AST to avoid importing the CUDA runtime. The interval
provider is injected; all checkpoint decisions execute the production method.
GPU cached-vs-cold continuation coverage is in the upstream e2e companion.
"""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/managers/scheduler_components/batch_result_processor.py"
)


def load_method(interval):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_mamba_check_track_boundary"
    )
    namespace = {"mamba_track_grid": lambda page_size: interval}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace[method.name]


class TestMambaSpecTrackBoundaryCPU(unittest.TestCase):
    def boundary(self, cur, accepted, retained=None, non_draft=1, interval=128):
        method = load_method(interval)
        result = NS(
            num_correct_drafts_per_req_cpu=[accepted],
            grammar_retained_tokens=retained,
            num_non_draft_tokens_per_req=non_draft,
        )
        return method(
            NS(tree_cache=NS(page_size=1)),
            NS(seqlen=cur + 1),
            NS(spec_algorithm=NS(is_none=lambda: False)),
            result,
            0,
        )

    def test_grammar_truncation_does_not_invent_checkpoint_crossing(self):
        # Verify accepted four drafts but grammar kept one. Committed window
        # [129, 130] contains no checkpoint at 128; the old code donated one.
        self.assertEqual(self.boundary(130, 4, [[42]]), (False, 0))

    def test_empty_retained_window_cannot_donate_state(self):
        self.assertEqual(self.boundary(128, 4, [[]]), (False, 0))

    def test_uno_root_is_not_a_committed_verify_token(self):
        self.assertEqual(self.boundary(128, 4, [[42]], non_draft=2), (False, 0))
        self.assertEqual(self.boundary(128, 4, [[42, 43]], non_draft=2), (True, 128))

    def test_non_draft_count_larger_than_retained_clamps_to_zero(self):
        self.assertEqual(self.boundary(128, 4, [[42]], non_draft=3), (False, 0))

    def test_no_grammar_preserves_accepted_draft_boundary(self):
        self.assertEqual(self.boundary(130, 3), (True, 128))
        self.assertEqual(self.boundary(130, 0), (False, 0))
        self.assertEqual(self.boundary(130, 3, [None]), (True, 128))

    def test_committed_windows_around_both_deployed_grids(self):
        for interval in (128, 256):
            for cur in range(interval - 4, interval + 5):
                for committed in range(5):
                    with self.subTest(interval=interval, cur=cur, committed=committed):
                        checkpoints = [
                            position
                            for position in range(cur - committed + 1, cur + 1)
                            if position % interval == 0
                        ]
                        expected = (True, checkpoints[-1]) if checkpoints else (False, 0)
                        self.assertEqual(
                            self.boundary(cur, 4, [list(range(committed))], interval=interval),
                            expected,
                        )

    def test_absent_draft_result_does_not_track(self):
        result = NS(num_correct_drafts_per_req_cpu=None)
        actual = load_method(128)(
            NS(tree_cache=NS(page_size=1)), NS(seqlen=129),
            NS(spec_algorithm=NS(is_none=lambda: False)), result, 0,
        )
        self.assertEqual(actual, (False, 0))

    def test_non_spec_decode_keeps_overlap_adjustment(self):
        method = load_method(128)
        owner = NS(tree_cache=NS(page_size=1))
        req = NS(kv=NS(kv_committed_len=128), decode_batch_idx=2)
        batch = NS(spec_algorithm=NS(is_none=lambda: True), mamba_decode_batch_idx_cpu=[2])
        self.assertEqual(method(owner, req, batch, NS(), 0), (True, 128))
        req.decode_batch_idx = 3
        self.assertEqual(method(owner, req, batch, NS(), 0), (False, 0))


if __name__ == "__main__":
    unittest.main()
