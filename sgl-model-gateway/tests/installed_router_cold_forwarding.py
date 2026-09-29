"""Exercise the installed Python router extension against local HTTP workers.

Run inside the final runtime image with this file mounted read-only. The launcher
and extension must come from the image, with no source tree on PYTHONPATH.
"""

import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Worker(ThreadingHTTPServer):
    def __init__(self, role):
        super().__init__(("127.0.0.1", free_port()), WorkerHandler)
        self.role = role
        self.requests = []
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_port}"

    def snapshot(self):
        with self.lock:
            return list(self.requests)

    def close(self):
        self.shutdown()
        self.server_close()
        self.thread.join(timeout=5)


class WorkerHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def reply(self, status, payload, headers=None):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/health", "/health_generate"):
            self.reply(200, {"status": "healthy"})
        elif self.path == "/model_info":
            self.reply(
                200,
                {
                    "model_path": "mock-model-path",
                    "tokenizer_path": "mock-tokenizer-path",
                    "is_generation": True,
                    "preferred_sampling_params": {"temperature": 0.7},
                },
            )
        elif self.path == "/server_info":
            self.reply(
                200,
                {
                    "model_path": "mock-model-path",
                    "tokenizer_path": "mock-tokenizer-path",
                    "host": "127.0.0.1",
                    "port": self.server.server_port,
                    "dp_size": 1,
                    "tp_size": 1,
                    "internal_states": [
                        {"waiting_queue_size": 0, "running_queue_size": 0}
                    ],
                },
            )
        else:
            self.reply(404, {"error": self.path})

    def do_POST(self):
        if self.path != "/generate":
            self.reply(404, {"error": self.path})
            return
        data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        payload = json.loads(data)
        with self.server.lock:
            self.server.requests.append((payload, dict(self.headers)))
        self.reply(
            200,
            {
                "text": f"decode-from-{self.server.role}",
                "meta_info": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            {"x-worker-id": self.server.role},
        )


LAUNCH = """
from sglang_router.launch_router import launch_router
from sglang_router.router_args import RouterArgs
import sys
mode, port, prefill, decode = sys.argv[1:]
common = dict(host="127.0.0.1", port=int(port), policy="round_robin",
              worker_startup_timeout_secs=15, worker_startup_check_interval=1,
              request_timeout_secs=15, disable_retries=True)
if mode == "pd":
    args = RouterArgs(pd_disaggregation=True,
                      prefill_urls=[(prefill, None)], decode_urls=[decode], **common)
else:
    args = RouterArgs(worker_urls=[prefill], **common)
launch_router(args)
"""


def post(port, payload, headers=None):
    request = Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read()), dict(response.headers)
    except HTTPError as error:
        body = error.read().decode()
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            pass
        return error.code, body, dict(error.headers)


def wait_ready(port, process):
    for _ in range(150):
        if process.poll() is not None:
            raise AssertionError(f"router exited early: {process.returncode}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                return
        except (URLError, TimeoutError):
            time.sleep(0.1)
    raise AssertionError("router did not become ready")


def check_forwarded(requests, suffix, expected_flag):
    matches = [
        (body, headers)
        for body, headers in requests
        if body.get("rid") == f"cold-fixture-{suffix}"
    ]
    assert len(matches) == 1, (suffix, requests)
    body, headers = matches[0]
    assert body["extra_key"] == f"key-{suffix}", body
    assert body.get("cold_shared_read_bypass") == expected_flag, body
    assert {key.lower(): value for key, value in headers.items()}[
        "x-correlation-id"
    ] == f"header-{suffix}", headers


def run_mode(mode):
    prefill = Worker("prefill")
    decode = Worker("decode") if mode == "pd" else None
    port = free_port()
    stderr_log = tempfile.TemporaryFile(mode="w+t")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            LAUNCH,
            mode,
            str(port),
            prefill.url,
            decode.url if decode else prefill.url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=stderr_log,
        text=True,
    )
    try:
        wait_ready(port, process)
        for suffix, flag in (("true", True), ("false", False), ("omitted", None)):
            payload = {
                "text": "cold forwarding fixture",
                "stream": False,
                "rid": f"cold-fixture-{suffix}",
                "extra_key": f"key-{suffix}",
            }
            if flag is not None:
                payload["cold_shared_read_bypass"] = flag
            status, response, response_headers = post(
                port, payload, {"X-Correlation-Id": f"header-{suffix}"}
            )
            assert status == 200, (mode, suffix, status, response)
            expected_role = "decode" if mode == "pd" else "prefill"
            assert response["text"] == f"decode-from-{expected_role}", response
            assert {key.lower(): value for key, value in response_headers.items()}[
                "x-worker-id"
            ] == expected_role, response_headers

        before = (len(prefill.snapshot()), len(decode.snapshot()) if decode else 0)
        status, _, _ = post(
            port, {"text": "invalid cold flag", "cold_shared_read_bypass": "true"}
        )
        assert 400 <= status < 500, (mode, status)
        after = (len(prefill.snapshot()), len(decode.snapshot()) if decode else 0)
        assert after == before, (mode, before, after)

        for worker in (prefill, decode) if decode else (prefill,):
            requests = worker.snapshot()
            assert len(requests) == 3, (mode, worker.role, requests)
            for suffix, flag in (("true", True), ("false", None), ("omitted", None)):
                check_forwarded(requests, suffix, flag)
        print(
            f"{mode}: installed launcher forwarded 3 valid requests; invalid type rejected"
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if sys.exc_info()[0] is not None:
            stderr_log.seek(0)
            print(stderr_log.read()[-6000:], file=sys.stderr)
        stderr_log.close()
        prefill.close()
        if decode:
            decode.close()


if __name__ == "__main__":
    run_mode("pd")
    run_mode("regular")
