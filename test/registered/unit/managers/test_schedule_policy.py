import unittest
from array import array
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(rid, origin_input_text, origin_input_ids, sampling_params=None, **kwargs):
    if sampling_params is None:
        sampling_params = SamplingParams()
    return Req(
        rid,
        origin_input_text,
        array("q", origin_input_ids),
        sampling_params,
        **kwargs,
    )


class TestSchedulePolicyHRRN(CustomTestCase):
    def test_calc_priority_hrrn(self):
        """HRRN sorts by response ratio (waited_tokens / uncached_tokens).

        With three fresh reqs (waited_tokens = 0 for all), the ratio is 0 / uncached for each,
        and stable sort keeps original order among equal keys -- effectively pure SUF via the
        rid tie-breaker is avoided, so we set a non-zero arrival snapshot on some reqs to exercise
        the aging half of the formula.
        """
        tree_cache = RadixCache.create_simulated()

        # r_short: small uncached, just arrived (waited=0)
        # r_long:  large uncached, just arrived (waited=0)
        # r_aged:  medium uncached, arrived long ago (waited>>0)
        r_short = _make_req("short", "a", [1])
        r_long = _make_req("long", "a" * 10, list(range(10)))
        r_aged = _make_req("aged", "a" * 3, [1, 2, 3])

        # Fresh reqs arrived with the counter still at 0.
        r_short.arrival_processed_tokens = 0
        r_long.arrival_processed_tokens = 0
        r_aged.arrival_processed_tokens = 0

        waiting_queue = [r_long, r_aged, r_short]

        policy = SchedulePolicy(
            policy="hrrn",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        # processed_tokens=1000 -> waited_tokens is 1000 for every req here.
        # Ratios: short = 1000/1 = 1000; aged = 1000/3 ~= 333; long = 1000/10 = 100.
        # Highest ratio first -> short, aged, long.
        policy.calc_priority(waiting_queue, processed_tokens=1000)

        self.assertEqual(waiting_queue[0].rid, "short")
        self.assertEqual(waiting_queue[1].rid, "aged")
        self.assertEqual(waiting_queue[2].rid, "long")

    def test_calc_priority_hrrn_aging_overtakes_short(self):
        """A long request that has waited enough should overtake a
        just-arrived short request."""
        tree_cache = RadixCache.create_simulated()

        r_long_old = _make_req("long_old", "a" * 100, list(range(100)))
        r_short_new = _make_req("short_new", "a", [1])

        # long_old arrived at counter=0 and has been waiting; short_new
        # just arrived (its snapshot equals the current counter).
        r_long_old.arrival_processed_tokens = 0
        r_short_new.arrival_processed_tokens = 100000

        waiting_queue = [r_short_new, r_long_old]

        policy = SchedulePolicy(
            policy="hrrn",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        # processed_tokens=100000 -> long_old.waited = 100000, ratio = 100000/100 = 1000.
        #                            short_new.waited = 0, ratio = 0.
        # long_old should now be first.
        policy.calc_priority(waiting_queue, processed_tokens=100000)

        self.assertEqual(waiting_queue[0].rid, "long_old")
        self.assertEqual(waiting_queue[1].rid, "short_new")

    def test_calc_priority_hrrn_cached_length_affects_order(self):
        """Cached prefix length shortens uncached, so reqs with the same input
        length can sort differently by HRRN. Also verifies rid tie-break for
        equal-ratio reqs.

        Uses _sort_by_hrrn directly to bypass the prefix-match pass inside
        calc_priority, which would overwrite num_matched_prefix_tokens.
        """
        r_more_cached = _make_req("a", "x" * 100, list(range(100)))
        r_less_cached = _make_req("b", "x" * 100, list(range(100)))
        r_tie = _make_req("c", "x" * 100, list(range(100)))

        # r_more_cached: 90 cached -> uncached = 10 -> ratio = 1000 / 10 = 100.
        # r_less_cached:  0 cached -> uncached = 100 -> ratio = 1000 / 100 = 10.
        # r_tie:          0 cached -> uncached = 100 -> ratio = 1000 / 100 = 10 (ties with r_less_cached; rid "b" < "c").
        r_more_cached.num_matched_prefix_tokens = 90
        r_less_cached.num_matched_prefix_tokens = 0
        r_tie.num_matched_prefix_tokens = 0
        r_more_cached.arrival_processed_tokens = 0
        r_less_cached.arrival_processed_tokens = 0
        r_tie.arrival_processed_tokens = 0

        waiting_queue = [r_tie, r_less_cached, r_more_cached]
        SchedulePolicy._sort_by_hrrn(waiting_queue, set(), processed_tokens=1000)

        self.assertEqual(waiting_queue[0].rid, "a")
        self.assertEqual(waiting_queue[1].rid, "b")
        self.assertEqual(waiting_queue[2].rid, "c")

    def test_sort_by_hrrn_orders_by_priority_first_when_enabled(self):
        """With priority scheduling, HRRN orders by priority first and by
        response ratio within a priority; with it off (priority_sign 0) the
        priority is ignored and pure HRRN order is kept."""
        r_low_aged = _make_req("low", "x" * 10, list(range(10)), priority=1)
        r_high_fresh = _make_req("high", "x" * 100, list(range(100)), priority=2)
        r_high_cached = _make_req("warm", "x" * 100, list(range(100)), priority=2)
        for r in (r_low_aged, r_high_fresh, r_high_cached):
            r.num_matched_prefix_tokens = 0
            r.arrival_processed_tokens = 0
        r_high_cached.num_matched_prefix_tokens = 100  # no prefill work left
        r_low_aged.arrival_processed_tokens = 0  # waited 1000 tokens for 10 uncached: ratio 100
        r_high_fresh.arrival_processed_tokens = 1000  # just arrived: ratio 0

        # Priority scheduling off: the aged low-priority request wins on ratio.
        waiting_queue = [r_high_fresh, r_low_aged, r_high_cached]
        SchedulePolicy._sort_by_hrrn(waiting_queue, set(), processed_tokens=1000)
        self.assertEqual([r.rid for r in waiting_queue], ["warm", "low", "high"])

        # Priority scheduling on, higher values first: priority 2 before 1,
        # and within priority 2 the request with no prefill work drains first.
        waiting_queue = [r_high_fresh, r_low_aged, r_high_cached]
        SchedulePolicy._sort_by_hrrn(
            waiting_queue, set(), processed_tokens=1000, priority_sign=-1
        )
        self.assertEqual([r.rid for r in waiting_queue], ["warm", "high", "low"])

        # A temporarily deprioritized request still stays inside its priority.
        waiting_queue = [r_high_fresh, r_low_aged, r_high_cached]
        SchedulePolicy._sort_by_hrrn(
            waiting_queue, {"high"}, processed_tokens=1000, priority_sign=-1
        )
        self.assertEqual([r.rid for r in waiting_queue], ["warm", "high", "low"])


    def test_calc_priority_hrrn_tiers_and_direction(self):
        for enabled, low_first, expected in (
            (False, False, ["flex", "default", "priority"]),
            (True, False, ["priority", "default", "flex"]),
            (True, True, ["flex", "default", "priority"]),
        ):
            with self.subTest(enabled=enabled, low_first=low_first):
                reqs = [
                    _make_req("default", "x" * 10, list(range(10)), priority=1),
                    _make_req("priority", "x" * 100, list(range(100)), priority=2),
                    _make_req("flex", "x", [1], priority=0),
                ]
                for req in reqs:
                    req.num_matched_prefix_tokens = 0
                    req.arrival_processed_tokens = 0
                reqs[1].arrival_processed_tokens = 1000
                policy = SchedulePolicy(
                    policy="hrrn",
                    tree_cache=RadixCache.create_simulated(),
                    enable_hierarchical_cache=True,
                    enable_priority_scheduling=enabled,
                    schedule_low_priority_values_first=low_first,
                )
                with patch.object(policy, "_compute_prefix_matches", return_value=set()):
                    policy.calc_priority(reqs, processed_tokens=1000)
                self.assertEqual([req.rid for req in reqs], expected)

    def test_calc_priority_hrrn_large_queue_keeps_priority_in_fcfs_fallback(self):
        for enabled, low_first in ((False, False), (True, False), (True, True)):
            with self.subTest(enabled=enabled, low_first=low_first):
                reqs = [
                    _make_req(str(index), "x", [index], priority=index % 3)
                    for index in range(129)
                ]
                for index, req in enumerate(reqs):
                    req.time_stats.wait_queue_entry_time = float(index)
                expected = list(reqs)
                if enabled:
                    sign = 1 if low_first else -1
                    expected.sort(
                        key=lambda req: (
                            req.priority * sign,
                            req.time_stats.wait_queue_entry_time,
                        )
                    )
                policy = SchedulePolicy(
                    policy="hrrn",
                    tree_cache=RadixCache.create_simulated(),
                    enable_hierarchical_cache=True,
                    enable_priority_scheduling=enabled,
                    schedule_low_priority_values_first=low_first,
                )
                with patch.object(policy, "_compute_prefix_matches") as prefix_matches:
                    policy.calc_priority(reqs, processed_tokens=1000)
                prefix_matches.assert_not_called()
                self.assertEqual([req.rid for req in reqs], [req.rid for req in expected])


if __name__ == "__main__":
    unittest.main()
