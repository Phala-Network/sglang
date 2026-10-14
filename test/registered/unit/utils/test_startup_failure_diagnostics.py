"""Fatal load errors remain diagnosable while credentials/locals stay private."""

import json
import logging

from sglang.srt.utils.framework_log_privacy import FrameworkPrivacyFilter
from sglang.srt.utils.startup_diagnostics import scheduler_exception_diagnostic


def fail_loader(message):
    private_prompt = "PRIVATE_FRAME_LOCAL_SENTINEL"
    raise RuntimeError(message)


def test_startup_keeps_failure_shape_and_code_locations_but_scrubs_secrets(monkeypatch):
    secret = "synthetic-token-12345678901234567890"
    api_key = "synthetic-api-09876543210987654321"
    monkeypatch.setenv("TOKEN", secret)
    try:
        fail_loader(f"expected shape (4, 3584), got (1, 3584); {secret}; {api_key}; Bearer other-credential")
    except RuntimeError as error:
        rendered = scheduler_exception_diagnostic(error, startup=True, credentials=(api_key,))
    assert secret not in rendered and api_key not in rendered
    assert "other-credential" not in rendered and "PRIVATE_FRAME_LOCAL_SENTINEL" not in rendered
    diagnostic = json.loads(rendered)
    assert diagnostic["exception_type"] == "builtins.RuntimeError"
    assert "expected shape (4, 3584), got (1, 3584)" in diagnostic["message"]
    assert diagnostic["frames"][-1]["function"] == "fail_loader"
    assert diagnostic["frames"][-1]["line"] > 0
    assert set(diagnostic["frames"][-1]) == {"file", "line", "function"}


def test_event_loop_omits_private_message_and_oversized_startup_is_bounded():
    try:
        fail_loader("PRIVATE_REQUEST_TEXT_SENTINEL" * 1000)
    except RuntimeError as error:
        runtime = json.loads(scheduler_exception_diagnostic(error, startup=False))
        startup = json.loads(scheduler_exception_diagnostic(error, startup=True))
    assert "message" not in runtime
    assert "PRIVATE_REQUEST_TEXT_SENTINEL" not in json.dumps(runtime)
    assert len(startup["message"]) == 1024


def test_framework_privacy_does_not_mask_safe_worker_diagnostics():
    record = logging.LogRecord("sglang.srt.managers.scheduler", logging.ERROR, __file__, 1,
                               "SCHEDULER_FAILURE %s", ('{"exception_type":"builtins.KeyError"}',), None)
    assert FrameworkPrivacyFilter().filter(record)
    assert "builtins.KeyError" in record.getMessage()
