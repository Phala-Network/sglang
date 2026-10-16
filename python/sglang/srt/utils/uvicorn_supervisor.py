"""Keep SGLang's fatal SIGQUIT handler when Uvicorn starts HTTP workers."""

import signal

import uvicorn
from sglang.srt.utils.engine_failure import mark_engine_failed
from uvicorn.supervisors import Multiprocess


class SGLangMultiprocess(Multiprocess):
    def __init__(self, *args, engine_failure_path=None, **kwargs):
        self._engine_sigquit_handler = signal.getsignal(signal.SIGQUIT)
        self._engine_failure_path = engine_failure_path
        super().__init__(*args, **kwargs)
        # Uvicorn 0.54 registers QUIT but has no handle_quit. SGLang owns
        # this fatal child-to-parent signal; TERM/INT remain Uvicorn-owned.
        signal.signal(signal.SIGQUIT, self._handle_engine_sigquit)

    def _handle_engine_sigquit(self, signum, frame):
        try:
            mark_engine_failed(self._engine_failure_path)
        finally:
            handler = self._engine_sigquit_handler
            if callable(handler):
                handler(signum, frame)
            else:
                signal.signal(signal.SIGQUIT, handler)
                signal.raise_signal(signal.SIGQUIT)

    def handle_quit(self):
        # Also cover a QUIT queued during the constructor's registration gap.
        self._handle_engine_sigquit(signal.SIGQUIT, None)


def run_multi_worker_server(app: str, *, engine_failure_path=None, **kwargs) -> None:
    config = uvicorn.Config(app, **kwargs)
    sock = config.bind_socket()
    try:
        SGLangMultiprocess(
            config, sockets=[sock], engine_failure_path=engine_failure_path
        ).run()
    finally:
        sock.close()
