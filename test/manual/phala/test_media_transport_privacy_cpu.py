"""Actual requests/urllib3 loopback fixture; run only on the authorized dev CPU.

The runtime media downloader is extracted unchanged to avoid importing model
or CUDA modules. No external address, credential, or environment proxy is used.
"""

import ast
import http.server
import importlib.util
import io
import ipaddress
import logging
import os
import socket
import sys
import threading
import time
import unittest
from http.client import RemoteDisconnected
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.exceptions import MaxRetryError, ProtocolError
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"
MARKER = "SYNTHETIC_PRIVATE_MEDIA_URL_QUERY_AUTH_PROMPT_CIPHERTEXT"
SPEC = importlib.util.spec_from_file_location(
    "media_transport_privacy_source", SRT / "utils/framework_log_privacy.py"
)
PRIVACY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRIVACY)
TRANSPORT_LOGGERS = (
    "urllib3.connectionpool",
    "urllib3.poolmanager",
    "urllib3.util.retry",
)


class MediaHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/broken":
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        if path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/media?private=" + MARKER)
            self.end_headers()
            return
        if path == "/media":
            payload = b"media-fixture-payload"
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(503)
        self.end_headers()

    def log_message(self, *_):
        pass


class MediaTransportPrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MediaHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.streams = [io.StringIO(), io.StringIO()]
        self.handlers = []
        for stream in self.streams:
            handler = logging.StreamHandler(stream)
            handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
            self.handlers.append(handler)
            self.addCleanup(handler.close)
        for name in TRANSPORT_LOGGERS:
            logger = logging.getLogger(name)
            saved = logger.handlers[:], logger.level, logger.propagate, logger.disabled

            def restore(target=logger, state=saved):
                target.handlers, target.level, target.propagate, target.disabled = state

            self.addCleanup(restore)
            logger.handlers = self.handlers[:]
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            logger.disabled = False
        # Test the real installer against existing transport handlers.
        PRIVACY.configure_framework_log_privacy({"handlers": {}})

        self.session = requests.Session()
        self.addCleanup(self.session.close)
        self.session.trust_env = False
        self.session.headers["Authorization"] = "Bearer " + MARKER
        self.session.mount(
            "http://",
            HTTPAdapter(
                max_retries=Retry(
                    total=1,
                    connect=1,
                    read=1,
                    backoff_factor=0,
                )
            ),
        )
        namespace = {
            "requests": requests,
            "os": os,
            "time": time,
            "ipaddress": ipaddress,
            "urlparse": urlparse,
            "urljoin": urljoin,
            "_mm_http_session": threading.local(),
            "_DEFAULT_MEDIA_URL_MAX_FILE_SIZE_MB": 64,
            "_MAX_MEDIA_URL_REDIRECTS": 5,
            "_MEDIA_URL_REDIRECT_STATUS_CODES": {301, 302, 303, 307, 308},
            "_allowed_media_domains": frozenset(),
            "_media_url_max_file_size_bytes": 64 * 1024 * 1024,
        }
        path = SRT / "utils/common.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        selected = {
            "_normalize_media_domain",
            "configure_media_url_security",
            "_assert_media_url_allowed",
            "download_remote_media",
            "get_mm_http_session",
        }
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in selected
        ]
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[future, *functions], type_ignores=[])
        )
        exec(compile(module, str(path), "exec"), namespace)
        namespace["_mm_http_session"].session = self.session
        namespace["_mm_http_session"].pid = os.getpid()
        namespace["configure_media_url_security"](["127.0.0.1"], max_file_size_mb=1)
        self.download = namespace["download_remote_media"]

    def url(self, path):
        return f"http://127.0.0.1:{self.server.server_port}{path}?private={MARKER}"

    def output(self):
        outputs = [stream.getvalue() for stream in self.streams]
        self.assertEqual(outputs[0], outputs[1])
        self.assertNotIn(MARKER, outputs[0])
        return outputs[0]

    def test_actual_download_preserves_method_status_without_query_or_auth(self):
        self.assertEqual(
            self.download(self.url("/media"), timeout=5), b"media-fixture-payload"
        )
        output = self.output()
        self.assertIn("method=GET", output)
        self.assertIn("status=200", output)

    def test_actual_redirect_keeps_each_response_status(self):
        self.assertEqual(
            self.download(self.url("/redirect"), timeout=5), b"media-fixture-payload"
        )
        output = self.output()
        self.assertIn("status=302", output)
        self.assertIn("status=200", output)

    def test_actual_retry_keeps_known_type_without_query_or_exception_text(self):
        raw_retry_types = []

        def capture_retry_type(record):
            # urllib3 logs the original error, while Retry.increment receives
            # a ProtocolError wrapper. Capture only the class, never repr/URL.
            if (
                record.msg == "Retrying (%r) after connection broken by '%r': %s"
                and isinstance(record.args, tuple)
                and len(record.args) == 3
            ):
                raw_retry_types.append(type(record.args[1]))
            return True

        logger = logging.getLogger("urllib3.connectionpool")
        logger.addFilter(capture_retry_type)
        try:
            with self.assertRaises(requests.exceptions.ConnectionError) as caught:
                self.download(self.url("/broken"), timeout=5)
        finally:
            logger.removeFilter(capture_retry_type)
        self.assertEqual(raw_retry_types, [RemoteDisconnected])
        self.assertIsInstance(caught.exception.args[0], MaxRetryError)
        self.assertIsInstance(caught.exception.args[0].reason, ProtocolError)
        output = self.output()
        self.assertIn("Media HTTP retry", output)
        self.assertIn("type=RemoteDisconnected", output)
        self.assertNotIn("type=Exception", output)

    def test_real_http_error_keeps_status_and_does_not_log_error_response(self):
        with self.assertRaises(requests.exceptions.HTTPError):
            self.download(self.url("/error"), timeout=5)
        self.assertIn("status=503", self.output())

    def test_removing_filter_restores_real_query_leak(self):
        for handler in self.handlers:
            handler.filters = []
        self.download(self.url("/media"), timeout=5)
        self.assertIn(MARKER, self.streams[0].getvalue())

    def test_custom_retry_exception_name_and_text_are_private(self):
        logger = logging.getLogger("urllib3.connectionpool")
        for base in (RuntimeError, RemoteDisconnected):
            error_type = type(MARKER, (base,), {})
            logger.warning(
                "Retrying (%r) after connection broken by '%r': %s",
                MARKER,
                error_type(MARKER),
                MARKER,
            )
        self.assertIn("type=Exception", self.output())

    def test_exception_context_is_removed_and_type_survives_two_handlers(self):
        logger = logging.getLogger("urllib3.connectionpool")
        try:
            raise PermissionError(MARKER)
        except PermissionError:
            logger.error(MARKER, exc_info=sys.exc_info(), stack_info=True)
        self.assertIn("type=PermissionError", self.output())

    def test_actual_pool_full_warning_keeps_size_but_not_host(self):
        from urllib3.connectionpool import HTTPConnectionPool

        pool = HTTPConnectionPool(MARKER + ".example", maxsize=1)
        self.addCleanup(pool.close)
        connection = MagicMock()
        # The initialized pool queue already holds its one available slot.
        # Returning an extra connection exercises the real pool-full warning.
        pool._put_conn(connection)
        connection.close.assert_called()
        logger = logging.getLogger("urllib3.connectionpool")
        logger.warning("Media HTTP connection pool full (pool_size=%d)", MARKER)
        output = self.output()
        self.assertNotIn(MARKER.lower(), output)
        self.assertIn("WARNING Media HTTP connection pool full (pool_size=1)", output)
        self.assertIn("pool_size=0", output)


if __name__ == "__main__":
    unittest.main()
