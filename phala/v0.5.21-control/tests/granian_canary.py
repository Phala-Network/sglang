import asyncio
import contextlib
import http.client
import io
import json
import logging
import socket
import sys

capture = io.StringIO()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from fastapi import Request
    from granian.constants import Interfaces, HTTPModes
    from granian.log import LogLevels
    from granian.server.embed import Server
    from sglang.srt.entrypoints.http_server import app
    from sglang.srt.utils.framework_log_privacy import configure_granian_log_privacy

marker = "SYNTHETIC_PRIVATE_GRANIAN_CANARY"


@contextlib.asynccontextmanager
async def synthetic_lifespan(application):
    yield


app.router.lifespan_context = synthetic_lifespan


@app.get("/__privacy/{value}")
async def privacy_ok(value: str):
    return {"ok": True}


@app.post("/__privacy_fail")
async def privacy_fail(request: Request):
    raise RuntimeError((await request.body()).decode() + request.headers.get("x-test", ""))


def requests(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/__privacy/" + marker + "?q=" + marker)
    res = conn.getresponse()
    first = res.status
    assert json.loads(res.read()) == {"ok": True}
    conn.request("POST", "/__privacy_fail?q=" + marker, body=marker, headers={"x-test": marker})
    res = conn.getresponse()
    second = res.status
    res.read()
    conn.close()
    return [first, second]


async def main():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = Server(app, address="127.0.0.1", port=port, interface=Interfaces.ASGI,
                    http=HTTPModes.auto, log_level=LogLevels.info, log_access=True,
                    log_dictconfig=configure_granian_log_privacy())
    for name in ("_granian", "granian.access"):
        for handler in logging.getLogger(name).handlers:
            handler.setStream(capture)
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(500):
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                await writer.wait_closed()
                break
            except OSError:
                await asyncio.sleep(0.01)
        statuses = await asyncio.to_thread(requests, port)
    finally:
        server.stop()
        await asyncio.wait_for(task, timeout=15)
    text = capture.getvalue()
    result = {"statuses": statuses, "marker_logged": marker in text, "traceback_logged": "Traceback (" in text,
              "nonempty": bool(text), "method_status_present": all(x in text for x in ("GET", "POST", "200", "500"))}
    result["passed"] = statuses == [200, 500] and not result["marker_logged"] and not result["traceback_logged"] and result["method_status_present"]
    print(json.dumps(result))
    return result["passed"]


sys.exit(0 if asyncio.run(main()) else 1)
