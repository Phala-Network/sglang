"""Regression coverage for queue-time timestamps across request retractions."""

import pickle
import unittest
from unittest import mock

import sglang.srt.observability.req_time_stats as rts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestQueueTimeAcrossRetracts(CustomTestCase):
    def _new_stats(self) -> rts.SchedulerReqTimeStats:
        stats = rts.SchedulerReqTimeStats()
        stats.set_metrics_collector(mock.MagicMock())
        stats.trace_ctx = mock.MagicMock()
        stats.trace_ctx.tracing_enable = True
        stats.trace_slice = mock.MagicMock()
        return stats

    def test_retracts_keep_the_first_queue_interval(self):
        stats = self._new_stats()
        stats.scheduler_recv_time = 1.0

        # Initial admission: queue_time is 5.0 - 2.0.
        stats.set_wait_queue_entry_time(ts=2.0)
        stats.set_forward_entry_time(ts=5.0)

        # Ordinary requeues refresh the scheduling/trace timestamp, while the
        # telemetry timestamp remains paired with the first forward timestamp.
        stats.set_wait_queue_entry_time(ts=10.0)
        self.assertEqual(stats.wait_queue_entry_time, 10.0)
        self.assertEqual(stats.first_wait_queue_entry_time, 2.0)
        stats.set_forward_entry_time(ts=11.0)
        stats.set_wait_queue_entry_time(ts=20.0)
        self.assertEqual(stats.wait_queue_entry_time, 20.0)
        self.assertEqual(stats.first_wait_queue_entry_time, 2.0)
        stats.set_forward_entry_time(ts=22.0)

        self.assertEqual(stats.wait_queue_entry_time, 20.0)
        self.assertEqual(stats.first_wait_queue_entry_time, 2.0)
        self.assertEqual(stats.forward_entry_time, 5.0)
        self.assertEqual(stats.last_forward_entry_time, 22.0)
        self.assertEqual(stats.get_queueing_time(), 3.0)
        self.assertEqual(stats.convert_to_output_meta_info()["queue_time"], 3.0)

        # queue_time remains one first-admission metric, not a requeue metric.
        queue_time_calls = stats.metrics_collector.observe_queue_time.call_args_list
        self.assertEqual(
            [call[0] for call in queue_time_calls],
            [(3.0,)],
        )
        # The initial slices remain unchanged; later requeues retain retract events.
        self.assertEqual(
            stats.trace_slice.call_args_list,
            [
                mock.call(rts.RequestStage.REQUEST_PROCESS, 1.0, 2.0),
                mock.call(rts.RequestStage.PREFILL_WAITING, 2.0, 5.0),
            ],
        )
        self.assertEqual(
            [call[0][:2] for call in stats.trace_ctx.trace_event.call_args_list],
            [("retract", 1), ("retract", 1)],
        )

    def test_prefill_retry_reset_starts_a_new_queue_interval(self):
        stats = self._new_stats()
        stats.scheduler_recv_time = 1.0
        stats.set_wait_queue_entry_time(ts=2.0)
        stats.set_forward_entry_time(ts=5.0)

        # This is a new prefill attempt, unlike an ordinary retract/requeue.
        stats.reset_prefill_retry_time()
        self.assertEqual(stats.wait_queue_entry_time, 0.0)
        self.assertEqual(stats.first_wait_queue_entry_time, 0.0)
        self.assertEqual(stats.forward_entry_time, 0.0)

        stats.scheduler_recv_time = 9.0
        stats.set_wait_queue_entry_time(ts=10.0)
        stats.set_forward_entry_time(ts=14.0)

        self.assertEqual(stats.wait_queue_entry_time, 10.0)
        self.assertEqual(stats.first_wait_queue_entry_time, 10.0)
        self.assertEqual(stats.forward_entry_time, 14.0)
        self.assertEqual(stats.get_queueing_time(), 4.0)
        self.assertEqual(stats.convert_to_output_meta_info()["queue_time"], 4.0)
        queue_time_calls = stats.metrics_collector.observe_queue_time.call_args_list
        self.assertEqual(
            [call[0] for call in queue_time_calls],
            [(3.0,), (4.0,)],
        )
        self.assertEqual(
            stats.trace_slice.call_args_list[-2:],
            [
                mock.call(rts.RequestStage.REQUEST_PROCESS, 9.0, 10.0),
                mock.call(rts.RequestStage.PREFILL_WAITING, 10.0, 14.0),
            ],
        )

    def test_directly_populated_timestamps_fall_back_to_current_wait(self):
        # Old callers and serialized records have no first-wait telemetry field.
        stats = rts.SchedulerReqTimeStats(
            wait_queue_entry_time=2.0,
            forward_entry_time=5.0,
        )

        self.assertEqual(stats.first_wait_queue_entry_time, 0.0)
        self.assertEqual(stats.get_queueing_time(), 3.0)

    def test_first_queue_timestamp_survives_serialization_round_trip(self):
        stats = rts.SchedulerReqTimeStats(has_timing_data=True)
        stats.set_wait_queue_entry_time(ts=200.0)
        stats.set_forward_entry_time(ts=205.0)
        stats.set_wait_queue_entry_time(ts=210.0)

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            payload = pickle.dumps(stats)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_004.0):
            restored = pickle.loads(payload)

        self.assertEqual(restored.wait_queue_entry_time, 206.0)
        self.assertEqual(restored.first_wait_queue_entry_time, 196.0)
        self.assertEqual(restored.forward_entry_time, 201.0)
        self.assertEqual(restored.get_queueing_time(), 5.0)
        self.assertEqual(restored.convert_to_output_meta_info()["queue_time"], 5.0)


if __name__ == "__main__":
    unittest.main()
