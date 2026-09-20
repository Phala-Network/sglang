"""CPU source-method regression: no torch/GPU installation required.

Executes the actual PrefillAdder methods parsed from the selected checkout;
only request/pool/cache objects are lightweight stand-ins. The scheduler test
executes its actual continuation-admission branch, not a handwritten copy.
This is not a GPU allocation or end-to-end runtime test.
"""

import ast
import pathlib
import sys
import unittest
from types import SimpleNamespace as NS

SOURCE = (
    pathlib.Path(sys.argv.pop(1))
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else pathlib.Path(__file__).resolve().parents[2]
)
POLICY = SOURCE / "python/sglang/srt/managers/schedule_policy.py"


def load_methods(path, class_name, names, globals_):
    module = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    cls.bases = []
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        cls,
                    ],
                    type_ignores=[],
                )
            ),
            str(path),
            "exec",
        ),
        globals_,
    )
    return globals_[class_name]


class Range:
    def __init__(self, start, end):
        self.start, self.end, self.length = start, end, end - start


Req = load_methods(
    SOURCE / "python/sglang/srt/managers/schedule_batch.py",
    "Req",
    {"set_extend_range"},
    {"Range": Range},
)
PrefillAdder = load_methods(
    POLICY,
    "PrefillAdder",
    {
        "add_chunked_req",
        "rem_total_tokens",
        "cur_rem_tokens",
        "rem_swa_tokens",
        "ceil_paged_tokens",
        "_update_prefill_budget",
        "_mamba_gap_budget_for_req",
        "_get_running_request_total_token_offset",
        "_swa_budget_for_req",
        "_swa_reserved_tokens",
    },
    {"CLIP_MAX_NEW_TOKENS": 4096},
)


def make_case(
    free=11200,
    evictable=0,
    reservation=20000,
    remaining=15070,
    prefix=81920,
    page=64,
    mixed=0,
):
    a = PrefillAdder.__new__(PrefillAdder)
    a.page_size = page
    a.tree_cache = NS(
        evictable_size=lambda: evictable,
        full_evictable_size=lambda: evictable,
        swa_evictable_size=lambda: 0,
    )
    a.token_to_kv_pool_allocator = NS(
        available_size=lambda: free,
        full_available_size=lambda: free,
        swa_available_size=lambda: 100000,
    )
    a.dllm_config = None
    a.is_all_swa = a.is_hybrid_swa = a.is_hybrid_ssm_cache = a._swa_req_ring = False
    a.rem_total_token_offset = reservation + mixed
    a.cur_rem_token_offset = mixed
    a.rem_chunk_tokens = 16384 - mixed
    a.rem_input_tokens = 16384 - mixed
    a.rem_mamba_slots = None
    a.rem_swa_token_offset = 0
    a._mamba_slot_cost = 0
    a.exact_chunk_fill = False
    a.can_run_list = []
    a.prefill_delayer_single_pass = None
    a.log_hit_tokens = a.log_input_tokens = 0
    a.reprocessed_log_hit_tokens = a.reprocessed_log_input_tokens = 0
    r = Req.__new__(Req)
    r.prefix_indices = range(prefix)
    r.full_untruncated_fill_ids = range(prefix + remaining)
    r.set_extend_range(prefix, prefix)
    r.sampling_params = NS(max_new_tokens=1000)
    r.output_ids = []
    r.retracted_stain = False
    r.inflight_middle_chunks = 0
    r.init_next_round_input = lambda: None
    return a, r


class ContinuationRegression(unittest.TestCase):
    def assert_capacity(self, a, r, free):
        if a.can_run_list:
            # Exact paged allocation oracle, independent of admission formula.
            p = a.page_size
            pages = (r.extend_range.end + p - 1) // p - (
                r.extend_range.start + p - 1
            ) // p
            self.assertLessEqual(pages * p, free)
            self.assertGreater(r.extend_range.length, 0)
            self.assertGreaterEqual(a.cur_rem_tokens, 0)

    def test_incident_15070_against_11200_with_41_running(self):
        a, r = make_case()
        a.new_token_ratio = 0.5
        running = [
            NS(sampling_params=NS(max_new_tokens=1000), output_ids=[])
            for _ in range(41)
        ]
        a.rem_total_token_offset = sum(
            a._get_running_request_total_token_offset(x) for x in running
        )
        self.assertLess(a.rem_total_tokens, 0)
        retained = a.add_chunked_req(r)
        self.assert_capacity(a, r, 11200)
        self.assertIs(retained, r)
        self.assertEqual(r.extend_range.length, 11136)

    def test_nonnegative_estimate_still_reserves_paged_margin(self):
        a, r = make_case(reservation=0)
        a.add_chunked_req(r)
        self.assert_capacity(a, r, 11200)

    def test_hybrid_ssm_dcp8_uses_virtual_allocator_pages(self):
        # Scheduler page64 versus allocator virtual512/physical64. Allocation
        # oracle counts actual new virtual pages, independently of admission.
        for free in (512, 1024, 1536, 4096):
            for mixed in (0, 1, 65, 511):
                for prefix in (0, 64, 448, 512, 576):
                    with self.subTest(free=free, mixed=mixed, prefix=prefix):
                        a, r = make_case(free=free, mixed=mixed, prefix=prefix)
                        a.is_hybrid_ssm_cache = True
                        a.token_to_kv_pool_allocator.page_size = 512
                        a.add_chunked_req(r)
                        if a.can_run_list:
                            pages = (r.extend_range.end + 511) // 512 - (prefix + 511) // 512
                            self.assertLessEqual(pages * 512, free - mixed)
                            self.assertGreaterEqual(a.cur_rem_tokens, 0)

    def test_dcp8_final_tail_reserves_allocator_page_not_compute_page(self):
        a, r = make_case(free=1536, reservation=0, remaining=65, prefix=512)
        a.is_hybrid_ssm_cache = True
        a.token_to_kv_pool_allocator.page_size = 512
        self.assertIsNone(a.add_chunked_req(r))
        self.assertEqual(r.extend_range.length, 65)
        self.assertEqual(a.cur_rem_token_offset, 1024)

    def test_no_space_or_less_than_two_pages_parks(self):
        for free in (0, 1, 63, 64, 65, 127):
            with self.subTest(free=free):
                a, r = make_case(free=free)
                self.assertIs(a.add_chunked_req(r), r)
                self.assertEqual(a.can_run_list, [])
                self.assertEqual(r.extend_range.length, 0)

    def test_exact_page_margin_fits_and_completes(self):
        a, r = make_case(free=128, remaining=64)
        self.assertIsNone(a.add_chunked_req(r))
        self.assert_capacity(a, r, 128)
        self.assertEqual(r.extend_range.length, 64)

    def test_prefix_alignment_and_mixed_decode_reservation(self):
        for prefix in (0, 1, 63, 64, 65, 81920):
            for mixed in (0, 41):
                with self.subTest(prefix=prefix, mixed=mixed):
                    a, r = make_case(prefix=prefix, mixed=mixed)
                    a.add_chunked_req(r)
                    self.assert_capacity(a, r, 11200 - mixed)

    def test_evictable_capacity_is_counted(self):
        a, r = make_case(free=64, evictable=11136)
        a.add_chunked_req(r)
        self.assert_capacity(a, r, 11200)
        self.assertEqual(r.extend_range.length, 11136)

    def test_healthy_chunk_and_final_tail_unchanged(self):
        for remaining in (64, 15070, 40000):
            a, r = make_case(free=100000, remaining=remaining, reservation=0)
            retained = a.add_chunked_req(r)
            self.assertEqual(r.extend_range.length, min(remaining, 16384))
            self.assertEqual(retained is r, remaining > 16384)

    def test_park_recover_multiple_chunks_then_complete(self):
        a, r = make_case(free=0, remaining=40000)
        self.assertIs(a.add_chunked_req(r), r)
        self.assertEqual(a.can_run_list, [])
        original_end = len(r.full_untruncated_fill_ids)
        lengths = []
        for free in (100000, 100000, 100000):
            a, _ = make_case(free=free, reservation=0)
            retained = a.add_chunked_req(r)
            lengths.append(r.extend_range.length)
            r.prefix_indices = range(r.extend_range.end)
            if retained is None:
                break
        self.assertEqual(lengths, [16384, 16384, 7232])
        self.assertEqual(len(r.prefix_indices), original_end)
        self.assertIsNone(retained)

    def test_hybrid_swa_low_space_still_parks(self):
        a, r = make_case(free=100000, reservation=0)
        a.is_hybrid_swa = True
        a.token_to_kv_pool_allocator.swa_available_size = lambda: 64
        self.assertIs(a.add_chunked_req(r), r)
        self.assertEqual(a.can_run_list, [])

    def test_scheduler_park_returns_before_unrelated_admission(self):
        path = SOURCE / "python/sglang/srt/managers/scheduler.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        method = next(
            n
            for n in ast.walk(module)
            if isinstance(n, ast.FunctionDef) and n.name == "_get_new_batch_prefill_raw"
        )
        branch = next(
            n
            for n in method.body
            if isinstance(n, ast.If)
            and any(
                isinstance(x, ast.Call)
                and isinstance(x.func, ast.Attribute)
                and x.func.attr == "add_chunked_req"
                for x in ast.walk(n)
            )
        )
        fn = ast.parse("def run(self, adder, running_batch):\n    pass\n").body[0]
        fn.body = [
            branch,
            ast.Return(value=ast.Constant("continued_waiting_admission")),
        ]
        ns = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                str(path),
                "exec",
            ),
            ns,
        )
        a, r = make_case(free=0)
        scheduler, running = NS(chunked_req=r), object()
        self.assertEqual(ns["run"](scheduler, a, running), (None, running))
        self.assertIs(scheduler.chunked_req, r)
        self.assertEqual(r.inflight_middle_chunks, 0)


if __name__ == "__main__":
    print(f"source={SOURCE}", flush=True)
    unittest.main(verbosity=2)
