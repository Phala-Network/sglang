"""Privacy-safe framework diagnostics without changing request processing."""

import logging
from copy import deepcopy

_METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"}
)
_EXCEPTIONS = frozenset(
    {
        "RuntimeError",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AssertionError",
        "AttributeError",
        "TimeoutError",
        "ConnectionError",
        "OSError",
        "CancelledError",
        "HTTPException",
        "RequestValidationError",
    }
)
_STATIC_EVENTS = frozenset(
    {
        "Application startup complete.",
        "Application shutdown complete.",
        "Waiting for application startup.",
        "Waiting for application shutdown.",
        "Shutting down",
        "Invalid HTTP request received.",
        "ASGI callable returned without starting response.",
        "ASGI callable returned without completing response.",
        "ASGI callable returned without sending handshake.",
        "Unsupported upgrade request.",
    }
)


class FrameworkPrivacyFilter(logging.Filter):
    def filter(self, record):
        if not record.name.startswith(("uvicorn", "granian", "_granian")):
            return True
        # Uvicorn's colored formatter can substitute this alternate template.
        record.__dict__.pop("color_message", None)
        record.__dict__.pop("message", None)
        if record.name == "uvicorn.access":
            values = record.args if isinstance(record.args, tuple) else ()
            method = (
                values[1]
                if len(values) == 5 and type(values[1]) is str and values[1] in _METHODS
                else "OTHER"
            )
            status = values[4] if len(values) == 5 else 0
            status = status if type(status) is int and 100 <= status <= 599 else 0
            # AccessFormatter requires its native five-item tuple.
            record.msg = '%s - "%s %s HTTP/%s" %d'
            record.args = ("redacted", method, "<request>", "1.1", status)
        elif record.name == "granian.access":
            values = record.args if isinstance(record.args, dict) else {}
            method = values.get("method")
            method = method if type(method) is str and method in _METHODS else "OTHER"
            status = values.get("status", 0)
            status = status if type(status) is int and 100 <= status <= 599 else 0
            record.msg = "HTTP request completed (method=%s, status=%d)"
            record.args = (method, status)
        elif record.exc_info or record.exc_text or record.stack_info:
            name = (
                getattr(record.exc_info[0], "__name__", "") if record.exc_info else ""
            )
            name = name if name in _EXCEPTIONS else "Exception"
            record.msg = "HTTP application exception (type=%s)"
            record.args = (name,)
        elif record.args:
            record.msg = "HTTP server event (details redacted)"
            record.args = ()
        elif not isinstance(record.msg, str):
            record.msg = "HTTP server event (details redacted)"
        elif record.msg not in _STATIC_EVENTS:
            record.msg = "HTTP server event (details redacted)"
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def configure_framework_log_privacy(config=None):
    if config is None:
        from uvicorn.config import LOGGING_CONFIG

        config = LOGGING_CONFIG
    name = "sglang_framework_log_privacy"
    config.setdefault("filters", {})[name] = {
        "()": "sglang.srt.utils.framework_log_privacy.FrameworkPrivacyFilter",
    }
    for handler in config.get("handlers", {}).values():
        filters = [
            existing for existing in handler.get("filters", ()) if existing != name
        ]
        filters.append(name)
        handler["filters"] = filters
    return config


def configure_granian_log_privacy():
    from granian.log import LOGGING_CONFIG

    return configure_framework_log_privacy(deepcopy(LOGGING_CONFIG))
