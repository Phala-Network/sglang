"""A real four-worker server using SGLang's readiness and fatal lifecycle."""

import argparse
import json
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from sglang.srt.entrypoints import http_server
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.srt.utils.common import kill_process_tree
from sglang.srt.utils.engine_failure import mark_engine_failed
from sglang.srt.utils.uvicorn_supervisor import run_multi_worker_server
from sglang.srt.utils.watchdog import WatchdogRaw

state_path = os.environ["SGLANG_TEST_ENGINE_STATE"]
manager = TokenizerManager.__new__(TokenizerManager)
manager.engine_failure_path = state_path
manager.is_pause = False
manager.gracefully_exit = False
manager.server_status = ServerStatus.Up
http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
app = FastAPI()
app.get("/ready")(http_server.ready)
app.get("/health")(http_server.health_generate)


@app.middleware("http")
async def identify_worker(request, call_next):
    response = await call_next(request)
    response.headers["X-Worker-Pid"] = str(os.getpid())
    return response


def scheduler():
    def publish_and_wait():
        mark_engine_failed(state_path)
        # Test barrier: inspect every HTTP worker before allowing real SIGQUIT.
        while not Path(state_path + ".release").exists():
            time.sleep(0.01)

    WatchdogRaw(
        debug_name="FixtureScheduler",
        get_counter=lambda: 0,
        is_active=lambda: Path(state_path + ".trigger").exists(),
        watchdog_timeout=0.5,
        on_failure=publish_and_wait,
        dump_info=lambda: (_ for _ in ()).throw(AssertionError("hard diagnostic")),
    )
    while True:
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--ignore-quit", action="store_true")
    parser.add_argument("--exit-on-quit", action="store_true")
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    if args.exit_on_quit:
        signal.signal(signal.SIGQUIT, lambda *_: os._exit(0))
    elif args.ignore_quit:
        signal.signal(signal.SIGQUIT, lambda *_: None)
    else:
        signal.signal(signal.SIGQUIT, lambda *_: kill_process_tree(os.getpid()))
    child = mp.Process(target=scheduler)
    child.start()
    Path(state_path + ".pids").write_text(
        json.dumps({"parent": os.getpid(), "scheduler": child.pid})
    )
    kwargs = dict(
        host="127.0.0.1", port=args.port, workers=4, loop="asyncio", log_level="warning"
    )
    module = "engine_failure_fixture:app"
    if args.original:
        import uvicorn

        uvicorn.run(module, **kwargs)
    else:
        run_multi_worker_server(module, engine_failure_path=state_path, **kwargs)
    kill_process_tree(os.getpid(), include_parent=False, wait_timeout=10)


if __name__ == "__main__":
    main()
