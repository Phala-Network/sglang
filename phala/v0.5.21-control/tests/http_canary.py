"""Actual installed SGLang app/Uvicorn loopback, no runtime/model startup."""
import contextlib
import http.client
import io
import json
import logging
import socket
import threading
import time
import sys

MARKER = "SYNTHETIC_PRIVATE_HTTP_CANARY"
capture = io.StringIO()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import uvicorn
    from fastapi import Request
    from sglang.srt.entrypoints.http_server import app
    from sglang.srt.utils.common import set_uvicorn_logging_configs


@app.get("/__privacy_probe/{value}")
async def privacy_ok(value: str):
    return {"ok": True}


@app.post("/__privacy_failure")
async def privacy_failure(request: Request):
    body = (await request.body()).decode()
    logging.getLogger("uvicorn.error").error("preformatted synthetic failure " + body)
    raise RuntimeError(body + request.headers.get("x-privacy-test", "") + str(request.query_params))


set_uvicorn_logging_configs()
config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="info",
                        lifespan="off", access_log=True, loop="asyncio", use_colors=True)
server = uvicorn.Server(config)
for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    for handler in logging.getLogger(name).handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(capture)
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
port = sock.getsockname()[1]
sock.listen(128)
thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
thread.start()
deadline = time.monotonic() + 20
while not server.started and thread.is_alive() and time.monotonic() < deadline:
    time.sleep(0.01)
assert server.started
connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
statuses = []
try:
    connection.request("GET", "/__privacy_probe/" + MARKER + "?query=" + MARKER)
    response = connection.getresponse()
    body = response.read()
    statuses.append(response.status)
    assert json.loads(body) == {"ok": True}
    connection.request("POST", "/__privacy_failure?query=" + MARKER,
                       body=MARKER, headers={"x-privacy-test": MARKER})
    response = connection.getresponse()
    response.read()
    statuses.append(response.status)
    assert statuses == [200, 500]
finally:
    connection.close()
    server.should_exit = True
    thread.join(timeout=15)
    sock.close()
text = capture.getvalue()
result = {"http_statuses": statuses, "marker_logged": MARKER in text,
          "traceback_logged": "Traceback (" in text, "log_characters": len(text),
          "access_event_present": "GET" in text and "200" in text and "POST" in text and "500" in text,
          "error_event_present": "RuntimeError" in text or "Exception in ASGI" in text,
          "thread_stopped": not thread.is_alive()}
result["passed"] = (not result["marker_logged"] and not result["traceback_logged"]
                    and result["access_event_present"] and result["error_event_present"]
                    and result["thread_stopped"])
print(json.dumps(result))
sys.exit(0 if result["passed"] else 1)
