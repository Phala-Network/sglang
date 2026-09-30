import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.http_server import (
    _send_disaggregation_warmup_requests,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


class TestDisaggregationServerWarmup(unittest.IsolatedAsyncioTestCase):
    async def test_sends_concurrent_scalar_request_to_each_dp_rank(self):
        from sglang.srt.runtime_context import get_context

        # The warmup fan-out width comes from the published topology.
        override = get_context().override_server_args(dp_size=4)
        override.install()
        self.addCleanup(override.restore)
        server_args = SimpleNamespace(dp_size=4)
        all_started = asyncio.Event()
        calls = []
        sessions = []

        class Response:
            status = 200

            async def __aenter__(self):
                if len(calls) == server_args.dp_size:
                    all_started.set()
                await asyncio.wait_for(all_started.wait(), timeout=5)
                return self

            async def __aexit__(self, *args):
                pass

            async def read(self):
                return b""

        class Session:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                sessions.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def post(self, *args, **kwargs):
                calls.append((args, kwargs))
                return Response()

        with patch("sglang.srt.entrypoints.http_server.aiohttp.ClientSession", Session):
            status_codes = await _send_disaggregation_warmup_requests(
                url="http://localhost:30000",
                headers={"Authorization": "Bearer token"},
                ssl_verify=False,
                timeout=123,
            )

        self.assertEqual(status_codes, [200] * server_args.dp_size)
        self.assertEqual(len(calls), server_args.dp_size)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(
            sessions[0].kwargs["headers"], {"Authorization": "Bearer token"}
        )
        self.assertEqual(sessions[0].kwargs["timeout"].total, 123)

        calls_by_rank = {
            kwargs["json"]["routed_dp_rank"]: (args, kwargs) for args, kwargs in calls
        }
        self.assertEqual(set(calls_by_rank), set(range(server_args.dp_size)))

        for dp_rank, (args, kwargs) in calls_by_rank.items():
            self.assertEqual(args, ("http://localhost:30000/generate",))
            self.assertEqual(kwargs["json"]["input_ids"], [10, 11, 12, 13])
            self.assertEqual(kwargs["json"]["bootstrap_host"], FAKE_BOOTSTRAP_HOST)
            self.assertEqual(kwargs["json"]["bootstrap_room"], dp_rank)
            self.assertFalse(kwargs["ssl"])


if __name__ == "__main__":
    unittest.main()


class TestFreezeGcAfterWarmup(unittest.TestCase):
    def _run(self, outcomes):
        import requests

        from sglang.srt.entrypoints import http_server
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(api_key="k", admin_api_key=None)
        override.install()
        self.addCleanup(override.restore)

        class Ok:
            def raise_for_status(self):
                pass

        class Unauthorized:
            def raise_for_status(self):
                raise requests.exceptions.HTTPError("401")

        responses = {"ok": Ok(), "401": Unauthorized()}

        def post(url, headers, **kwargs):
            self.assertEqual(headers["Authorization"], "Bearer k")
            outcome = outcomes.pop(0)
            if outcome == "refused":
                raise requests.exceptions.ConnectionError()
            return responses[outcome]

        server_args = SimpleNamespace(url=lambda: "http://127.0.0.1:30000")
        with (
            patch.object(http_server.requests, "post", side_effect=post) as mock_post,
            patch.object(http_server.time, "sleep"),
            patch.object(http_server, "ssl_verify_of", return_value=True),
            patch.object(http_server.logger, "warning") as warning,
        ):
            http_server._freeze_gc_after_server_warmup(server_args)
        return mock_post.call_count, warning.call_count

    def test_waits_for_the_listener_after_skipped_warmup(self):
        self.assertEqual(self._run(["refused", "refused", "ok"]), (3, 0))

    def test_gives_up_at_once_on_an_http_error(self):
        self.assertEqual(self._run(["401"]), (1, 1))
