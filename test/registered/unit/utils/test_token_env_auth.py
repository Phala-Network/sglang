"""TOKEN opt-in through official CLI; runtime/diagnostic separation, no GPU."""
import asyncio
import json
import os
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.token_auth import redact_auth_argv, redact_auth_config
from sglang.srt.runtime_context import get_context, get_serving, publish, reset_context
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.srt.utils.auth import AuthLevel, decide_request_auth

TOKEN = "unit-test-token-only-7f9b"


class TokenEnvironmentAuthTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"PIG_AUTH_FROM_TOKEN": "0", "TOKEN": TOKEN})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(reset_context)
        reset_context()

    def resolve(self, **fields):
        args = ServerArgs(model_path="dummy", **fields)
        args.resolve_once()
        return args

    def test_absent_and_disabled_opt_in_preserve_legacy_keys(self):
        for mode in (None, "0"):
            with self.subTest(mode=mode):
                if mode is None:
                    os.environ.pop("PIG_AUTH_FROM_TOKEN", None)
                else:
                    os.environ["PIG_AUTH_FROM_TOKEN"] = mode
                args = self.resolve(api_key="old-api", admin_api_key="old-admin")
                self.assertEqual(args.resolved_dict()["api_key"], "old-api")
                self.assertEqual(args.resolved_dict()["admin_api_key"], "old-admin")

    def test_official_cli_resolves_auth_and_preserves_runtime_ipc(self):
        os.environ["PIG_AUTH_FROM_TOKEN"] = "1"
        args = prepare_server_args(["--model-path", "dummy", "--disable-overlap-schedule"])
        args.resolve_once()
        self.assertIsNone(args.api_key)  # raw input is deliberately unchanged
        self.assertIsNone(args.admin_api_key)
        self.assertEqual(args.resolved_dict()["api_key"], TOKEN)
        self.assertEqual(args.resolved_dict()["admin_api_key"], TOKEN)
        self.assertNotIn(TOKEN, args.launch_command)
        self.assertNotIn(TOKEN, json.dumps(args.diagnostic_dict()))
        copied = pickle.loads(pickle.dumps(args))
        os.environ.pop("TOKEN")
        publish(copied, role="tokenizer")  # resolved child does not re-read env
        self.assertEqual(get_serving().api_key, TOKEN)
        self.assertEqual(get_serving().admin_api_key, TOKEN)
        self.assertEqual(get_context().resolved_server_args_dict()["api_key"], TOKEN)

    def test_both_auth_levels_require_unified_credential(self):
        os.environ["PIG_AUTH_FROM_TOKEN"] = "1"
        publish(self.resolve(), role="tokenizer")
        for level in (AuthLevel.NORMAL, AuthLevel.ADMIN_OPTIONAL, AuthLevel.ADMIN_FORCE):
            for credential, expected in ((None, False), ("Bearer wrong", False), ("Bearer " + TOKEN, True)):
                with self.subTest(level=level, credential_present=credential is not None):
                    result = decide_request_auth(
                        method="POST", path="/generate", authorization_header=credential,
                        api_key=get_serving().api_key, admin_api_key=get_serving().admin_api_key,
                        auth_level=level,
                    )
                    self.assertEqual(result.allowed, expected)

    def test_missing_and_invalid_token_fail_before_model_resolution(self):
        os.environ["PIG_AUTH_FROM_TOKEN"] = "1"
        for value in (None, "", " leading", "trailing ", "two words", "a\nb", "a\rb", "a\tb", "a\x7fb", "nonascii-\u00e9"):
            with self.subTest(case=repr(value)):
                if value is None:
                    os.environ.pop("TOKEN", None)
                else:
                    os.environ["TOKEN"] = value
                with self.assertRaisesRegex(ValueError, "requires a nonempty printable ASCII"):
                    self.resolve()

    def test_invalid_opt_in_fails_closed_without_echoing_value(self):
        os.environ["PIG_AUTH_FROM_TOKEN"] = "invalid-sensitive-fixture"
        with self.assertRaises(ValueError) as caught:
            self.resolve()
        self.assertNotIn(os.environ["PIG_AUTH_FROM_TOKEN"], str(caught.exception))

    def test_all_explicit_keys_rejected_without_echoing_secret(self):
        os.environ["PIG_AUTH_FROM_TOKEN"] = "1"
        for fields in (
            {"api_key": TOKEN}, {"admin_api_key": TOKEN},
            {"api_key": TOKEN, "admin_api_key": TOKEN},
            {"api_key": "conflict-value"}, {"admin_api_key": "conflict-value"},
            {"api_key": ""},
        ):
            with self.assertRaises(ValueError) as caught:
                self.resolve(**fields)
            self.assertNotIn(TOKEN, str(caught.exception))
            self.assertNotIn("conflict-value", str(caught.exception))

    def test_explicit_cli_secrets_removed_from_launch_readback(self):
        args = prepare_server_args([
            "--model-path", "dummy", "--api-key", TOKEN,
            "--admin-api-key=" + TOKEN,
        ])
        self.assertNotIn(TOKEN, args.launch_command)
        self.assertEqual(args.resolved_dict()["api_key"], TOKEN)
        self.assertEqual(redact_auth_argv(["--api-key=x", "--port", "30000"]), ["--api-key=[REDACTED]", "--port", "30000"])
        abbreviated = prepare_server_args([
            "--model-path", "dummy", "--api-k", TOKEN, "--admin-api-k=" + TOKEN,
        ])
        self.assertNotIn(TOKEN, abbreviated.launch_command)

    def test_nested_readback_redaction_does_not_change_ipc_payload(self):
        raw = {"api_key": TOKEN, "internal_states": [{"server_args": {"admin_api_key": TOKEN, "api_key": None}}]}
        redacted = redact_auth_config(raw)
        self.assertNotIn(TOKEN, json.dumps(redacted))
        self.assertEqual(raw["api_key"], TOKEN)
        self.assertEqual(raw["internal_states"][0]["server_args"]["admin_api_key"], TOKEN)
        self.assertIsNone(redacted["internal_states"][0]["server_args"]["api_key"])

    def manager(self):
        os.environ["PIG_AUTH_FROM_TOKEN"] = "1"
        args = self.resolve()
        return SimpleNamespace(
            server_args=args, startup_time=0,
            get_internal_state=AsyncMock(return_value=[{"api_key": TOKEN, "nested": {"admin_api_key": TOKEN}}]),
        )

    def test_actual_http_server_info_redacts_nested_scheduler_state(self):
        from sglang.srt.entrypoints import http_server
        manager = self.manager()
        state = SimpleNamespace(tokenizer_manager=manager, scheduler_info={})
        with patch.object(http_server, "_global_state", state), patch.object(http_server, "describe_kv_events_publisher", return_value=None):
            result = asyncio.run(http_server.server_info())
        self.assertNotIn(TOKEN, json.dumps(result))
        self.assertEqual(manager.server_args.resolved_dict()["api_key"], TOKEN)

    def test_actual_engine_and_grpc_readbacks_redact(self):
        from sglang.srt.entrypoints.engine import Engine
        from sglang.srt.entrypoints import grpc_bridge
        manager = self.manager()
        loop = asyncio.new_event_loop()
        try:
            fake = SimpleNamespace(tokenizer_manager=manager, loop=loop,
                _scheduler_init_result=SimpleNamespace(scheduler_infos=[{}]))
            self.assertNotIn(TOKEN, json.dumps(Engine.get_server_info(fake)))
        finally:
            loop.close()
        fake = SimpleNamespace(tokenizer_manager=manager, scheduler_info={})
        with patch.object(grpc_bridge, "describe_kv_events_publisher", return_value=None):
            self.assertNotIn(TOKEN, grpc_bridge.RuntimeHandle.get_server_info(fake))

    def test_tokenizer_dump_redacts_without_changing_runtime_record(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        manager = self.manager()
        manager.resolved_config_dict = lambda base: {**base, "nested": {"admin_api_key": TOKEN}}
        result = TokenizerManager._dump_config_snapshot(manager)
        self.assertNotIn(TOKEN, json.dumps(result))
        self.assertEqual(manager.server_args.resolved_dict()["admin_api_key"], TOKEN)


if __name__ == "__main__":
    unittest.main()
