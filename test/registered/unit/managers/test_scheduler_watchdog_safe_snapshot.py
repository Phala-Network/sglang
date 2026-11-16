"""The actual scheduler diagnostic producer retains numeric useful state."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

from sglang.srt.managers.scheduler_components import invariant_checker as module

MARKER = "SYNTHETIC_PRIVATE_REQUEST_SCHEMA_URL"


class TestSchedulerWatchdogSafeSnapshot(unittest.TestCase):
    def test_numeric_pool_queue_and_token_shapes_are_logged_without_request_data(self):
        stats = NS(full_available_size=17, full_evictable_size=19, full_num_used=23,
                   mamba_available_size=29, swa_available_size=MARKER)
        scheduler = NS(is_initializing=False, forward_ct=31,
                       waiting_queue=[MARKER] * 2,
                       pool_stats_observer=NS(get_pool_stats=MagicMock(return_value=stats)),
                       invariant_checker=NS(_check_all_pools=MagicMock(return_value=(False, MARKER))),
                       cur_batch_for_debug=NS(reqs=[NS(rid=MARKER, origin_input_ids=[MARKER] * 3,
                                                      output_ids=[MARKER] * 5)]))
        with patch.object(module, "WatchdogRaw") as constructor:
            module.create_scheduler_watchdog(scheduler, 10, soft=True)
        with patch("sglang.srt.distributed.device_communicators.pynccl_allocator.describe_registration_in_progress",
                   return_value=MARKER), self.assertLogs(module.logger, level="ERROR") as logs:
            result = constructor.call_args.kwargs["dump_info"]()
        output = "\n".join(logs.output)
        self.assertIsNone(result)
        self.assertNotIn(MARKER, output)
        self.assertIn("initializing=False registering=True forward_count=31 waiting=2", output)
        self.assertIn("'full_available_size': 17", output)
        self.assertIn("'full_evictable_size': 19", output)
        self.assertIn("'full_num_used': 23", output)
        self.assertIn("'mamba_available_size': 29", output)
        self.assertNotIn("swa_available_size", output)
        self.assertIn("request_token_shapes=[(3, 5)]", output)

    def test_initializing_snapshot_does_not_touch_uninitialized_pools(self):
        scheduler = NS(is_initializing=True, forward_ct=0, waiting_queue=[],
                       pool_stats_observer=MagicMock())
        with patch.object(module, "WatchdogRaw") as constructor:
            module.create_scheduler_watchdog(scheduler, 10, soft=True)
        with patch("sglang.srt.distributed.device_communicators.pynccl_allocator.describe_registration_in_progress",
                   return_value=None), self.assertLogs(module.logger, level="ERROR") as logs:
            constructor.call_args.kwargs["dump_info"]()
        scheduler.pool_stats_observer.get_pool_stats.assert_not_called()
        self.assertIn("initializing=True registering=False forward_count=0 waiting=0", "\n".join(logs.output))
