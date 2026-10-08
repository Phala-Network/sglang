import copy
import io
import json
import logging
import logging.config

from uvicorn.config import LOGGING_CONFIG
from sglang.srt.utils.framework_log_privacy import (
    configure_framework_log_privacy, configure_granian_log_privacy,
)

marker = "SYNTHETIC_PRIVATE_FILTER_CANARY"
checks = []
for round_index in range(2):
    config = configure_framework_log_privacy(copy.deepcopy(LOGGING_CONFIG))
    for formatter in config["formatters"].values():
        formatter["use_colors"] = bool(round_index)
    configure_framework_log_privacy(config)
    logging.config.dictConfig(config)
    capture = io.StringIO()
    for name in ("uvicorn", "uvicorn.access"):
        for handler in logging.getLogger(name).handlers:
            handler.setStream(capture)
    logging.getLogger("uvicorn.access").info('%s - "%s %s HTTP/%s" %d', marker, "POST", "/" + marker, "1.1", 201)
    try:
        raise RuntimeError(marker)
    except RuntimeError:
        logging.getLogger("uvicorn.error").error("Exception in ASGI application", exc_info=True, stack_info=True)
    logging.getLogger("uvicorn.error").warning("HTTP diagnostic %s", marker)
    logging.getLogger("uvicorn.error").error("preformatted " + marker)
    logging.getLogger("uvicorn.error").info("Started server process [%d]", 123,
        extra={"color_message": "Started server process [\x1b[36m%d\x1b[0m]"})
    logging.getLogger("uvicorn.error").error("safe fixed event",
        extra={"color_message": marker})
    cached_record = logging.LogRecord("uvicorn.error", logging.ERROR, __file__, 1, "safe fixed event", (), None)
    cached_record.message = marker
    cached_record.color_message = marker
    logging.getLogger("uvicorn.error").handle(cached_record)
    logging.getLogger("uvicorn.asgi").warning("preformatted " + marker)
    malicious_exception = type(marker, (RuntimeError,), {})
    try:
        raise malicious_exception(marker)
    except RuntimeError:
        logging.getLogger("uvicorn.error").error("formatted exception " + marker, exc_info=True)
        logging.getLogger("uvicorn.access").error('%s - "%s %s HTTP/%s" %d',
            marker, "GET", marker, "1.1", 500, exc_info=True, stack_info=True)
    logging.getLogger("uvicorn.error").info("Application startup complete.")
    text = capture.getvalue()
    checks.append({"case": "uvicorn_config_worker_reload_colors_" + str(round_index),
                   "passed": marker not in text and "Traceback (" not in text and
                             "POST" in text and "201" in text and "RuntimeError" in text and
                             "type=Exception" in text and "Application startup complete." in text})
config = configure_granian_log_privacy()
logging.config.dictConfig(config)
capture = io.StringIO()
for name in ("_granian", "granian.access"):
    for handler in logging.getLogger(name).handlers:
        handler.setStream(capture)
logging.getLogger("granian.access").info("%(path)s %(query_string)s %(header)s %(method)s %(status)d", {
    "path": marker, "query_string": marker, "header": marker, "method": "GET", "status": 200})
logging.getLogger("_granian").error("ASGI failure " + marker)
text = capture.getvalue()
checks.append({"case": "granian_handler_filters", "passed": marker not in text and "GET" in text and "200" in text
               and "HTTP server event" in text})
print(json.dumps({"passed": all(row["passed"] for row in checks), "cases": checks}))
assert all(row["passed"] for row in checks)
