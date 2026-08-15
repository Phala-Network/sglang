import asyncio
import json
import unittest

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from sglang.srt.entrypoints.http_server import app
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def request_for(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }
    )


class TestOpenAIV1ErrorEnvelope(unittest.TestCase):
    def assert_openai_error(self, response, expected_status: int):
        self.assertEqual(response.status_code, expected_status)
        body = json.loads(response.body)
        self.assertEqual(set(body), {"error"})
        self.assertEqual(set(body["error"]), {"message", "type", "param", "code"})
        self.assertEqual(body["error"]["code"], expected_status)
        self.assertIsNone(body["error"]["param"])

    def test_chat_request_validation_uses_nested_envelope(self):
        exc = RequestValidationError(
            errors=[
                {
                    "type": "value_error",
                    "loc": ("body", "tool_choice"),
                    "msg": "Value error, invalid tool choice",
                    "input": {"type": "invalid"},
                    "ctx": {"error": ValueError("invalid tool choice")},
                }
            ]
        )
        handler = app.exception_handlers[RequestValidationError]
        response = asyncio.run(handler(request_for("/v1/chat/completions"), exc))
        self.assert_openai_error(response, 400)
        self.assertIn("tool_choice", json.loads(response.body)["error"]["message"])

    def test_chat_http_exception_uses_nested_envelope(self):
        exc = HTTPException(status_code=401, detail="Invalid authentication")
        handler = app.exception_handlers[HTTPException]
        response = asyncio.run(handler(request_for("/v1/chat/completions"), exc))
        self.assert_openai_error(response, 401)

    def test_non_v1_validation_keeps_legacy_shape(self):
        exc = RequestValidationError(
            errors=[
                {
                    "type": "missing",
                    "loc": ("body", "text"),
                    "msg": "Field required",
                    "input": {},
                }
            ]
        )
        handler = app.exception_handlers[RequestValidationError]
        response = asyncio.run(handler(request_for("/generate"), exc))
        body = json.loads(response.body)
        self.assertNotIn("error", body)
        self.assertEqual(body["object"], "error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
