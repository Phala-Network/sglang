"""Synthetic diagnostics: original response preserved, no request text logged."""
import responses_cpu_helpers
import asyncio
import json
import logging
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sglang.srt.entrypoints.openai import serving_base as base
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

PRIVATE = "synthetic-private-value\nFORGED-log-line"


def serving():
    obj = object.__new__(OpenAIServingChat)
    obj.chat_encoding_spec = "kimi_k3"
    obj.tokenizer_manager = SimpleNamespace(request_logger=SimpleNamespace(log_requests=False))
    obj._validate_request = lambda req: None
    obj._convert_to_internal_request = lambda req, raw: (SimpleNamespace(sampling_params={}), req)
    async def ok(*args):
        return "unchanged-success"
    obj._handle_streaming_request = ok
    obj._handle_non_streaming_request = ok
    return obj


@pytest.mark.parametrize("stream", [False, True])
def test_success_never_calls_diagnostics(stream):
    with patch.object(base, "_log_safe_request_rejection", side_effect=AssertionError("success logged")):
        assert asyncio.run(serving().handle_request(SimpleNamespace(stream=stream), None)) == "unchanged-success"


@pytest.mark.parametrize("stage", ["validation", "conversion", "json_constraint", "stream", "nonstream"])
@pytest.mark.parametrize("kind", ["value", "http"])
def test_error_has_one_safe_record_and_unchanged_response(stage, kind, caplog):
    obj = serving()
    exc = ValueError(PRIVATE) if kind == "value" else HTTPException(422, PRIVATE)
    def fail(*args):
        raise exc
    async def async_fail(*args):
        raise exc
    if stage == "validation":
        obj._validate_request = fail
    elif stage == "conversion":
        obj._convert_to_internal_request = fail
    elif stage == "json_constraint":
        obj._validate_active_json_constraint = fail
    else:
        setattr(obj, "_handle_streaming_request" if stage == "stream" else "_handle_non_streaming_request", async_fail)
    with caplog.at_level(logging.INFO, logger=base.__name__):
        result = asyncio.run(obj.handle_request(SimpleNamespace(stream=stage == "stream"), None))
    assert result.status_code == (400 if kind == "value" else 422)
    assert json.loads(result.body)["message"] == PRIVATE
    records = [r.getMessage() for r in caplog.records if r.name == base.__name__]
    assert len(records) == 1
    assert "stage=response" in records[0]
    assert "source=serving_base" in records[0]
    assert PRIVATE not in records[0] and "FORGED" not in records[0]
    assert "test_safe_rejection" not in records[0]


def test_validation_return_logs_without_changing_payload(caplog):
    obj = serving()
    obj._validate_request = lambda req: PRIVATE
    with caplog.at_level(logging.INFO, logger=base.__name__):
        response = asyncio.run(obj.handle_request(SimpleNamespace(stream=False), None))
    assert response.status_code == 400 and json.loads(response.body)["message"] == PRIVATE
    assert len(caplog.records) == 1 and "stage=response" in caplog.text
    assert PRIVATE not in caplog.text


@pytest.mark.parametrize("status", [400, HTTPStatus.BAD_REQUEST, 503])
def test_direct_error_uses_fixed_category(status, caplog):
    with caplog.at_level(logging.INFO, logger=base.__name__):
        response = serving().create_error_response(PRIVATE, status_code=status, param=PRIVATE)
    assert response.status_code == status
    assert len(caplog.records) == 1
    assert f"stage=response status={int(status)} source=unknown line=0" in caplog.text
    assert PRIVATE not in caplog.text


def test_untrusted_diagnostic_fields_are_not_logged(caplog):
    with caplog.at_level(logging.INFO, logger=base.__name__):
        base._log_safe_request_rejection(PRIVATE, PRIVATE, ValueError(PRIVATE))
    assert PRIVATE not in caplog.text
    assert "stage=unknown status=0" in caplog.text


def test_selected_schema_error_points_at_real_code(caplog):
    obj = serving()
    obj._convert_to_internal_request = lambda req, raw: (SimpleNamespace(sampling_params={"json_schema": json.dumps({"type":"array","uniqueItems":True})}), req)
    with caplog.at_level(logging.INFO, logger=base.__name__):
        response = asyncio.run(obj.handle_request(SimpleNamespace(stream=False), None))
    assert response.status_code == 400
    assert len(caplog.records) == 1
    assert "stage=response status=400 source=serving_base line=" in caplog.text
    assert "line=0" not in caplog.text
