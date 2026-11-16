"""Privacy-safe framework diagnostics without changing request processing."""

import logging
from copy import deepcopy

_METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"}
)
_MEDIA_TRANSPORT_LOGGERS = frozenset(
    {"urllib3.connectionpool", "urllib3.poolmanager", "urllib3.util.retry"}
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
        "PermissionError",
        "AccessDenied",
        "NoSuchProcess",
        "TimeoutExpired",
        "ProtocolError",
        "RemoteDisconnected",
        "NewConnectionError",
        "NameResolutionError",
        "ReadTimeoutError",
        "ConnectTimeoutError",
        "SSLError",
        "CancelledError",
        "HTTPException",
        "RequestValidationError",
        "HTTPError",
        "Timeout",
        "TooManyRedirects",
        "UnidentifiedImageError",
        "MemoryError",
        "OutOfMemoryError",
        "ImportError",
        "FileNotFoundError",
        "JSONDecodeError",
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


def safe_exception_type(error):
    """Return only known exception names; messages and custom names are private."""
    name = type(error).__name__
    return name if name in _EXCEPTIONS else "Exception"


class FrameworkPrivacyFilter(logging.Filter):
    def filter(self, record):
        media_transport = record.name in _MEDIA_TRANSPORT_LOGGERS
        if not media_transport and not record.name.startswith(
            ("uvicorn", "granian", "_granian")
        ):
            return True
        # Uvicorn's colored formatter can substitute this alternate template.
        record.__dict__.pop("color_message", None)
        record.__dict__.pop("message", None)
        for field, value in (
            ("request_line", "<request>"),
            ("client_addr", "redacted"),
            ("scope", "<request>"),
            ("headers", "<redacted>"),
            ("body", "<redacted>"),
        ):
            if field in record.__dict__:
                record.__dict__[field] = value
        if media_transport:
            values = record.args if isinstance(record.args, tuple) else ()
            message = record.msg if type(record.msg) is str else ""
            if message == '%s://%s:%s "%s %s %s" %s %s' and len(values) == 8:
                method, status = values[3], values[6]
                record.msg = "Media HTTP response (method=%s, status=%d)"
                record.args = (
                    method if type(method) is str and method in _METHODS else "OTHER",
                    status if type(status) is int and 100 <= status <= 599 else 0,
                )
            elif (
                message == "Media HTTP response (method=%s, status=%d)"
                and len(values) == 2
            ):
                method, status = values
                record.args = (
                    method if type(method) is str and method in _METHODS else "OTHER",
                    status if type(status) is int and 100 <= status <= 599 else 0,
                )
            elif (
                message
                == "Connection pool is full, discarding connection: %s. Connection pool size: %s"
                and len(values) == 2
            ) or (
                message == "Media HTTP connection pool full (pool_size=%d)"
                and len(values) == 1
            ):
                size = values[-1]
                record.msg = "Media HTTP connection pool full (pool_size=%d)"
                record.args = (
                    size if type(size) is int and 0 <= size <= 2147483647 else 0,
                )
            elif (
                message == "Retrying (%r) after connection broken by '%r': %s"
                and len(values) == 3
            ):
                record.msg = "Media HTTP retry (type=%s)"
                record.args = (safe_exception_type(values[1]),)
            elif (
                message == "Media HTTP retry (type=%s)"
                and len(values) == 1
                and type(values[0]) is str
                and (values[0] in _EXCEPTIONS or values[0] == "Exception")
            ):
                pass
            elif message == "Incremented Retry for (url='%s'): %r":
                record.msg, record.args = "Media HTTP retry state updated", ()
            elif message == "Redirecting %s -> %s":
                record.msg, record.args = "Media HTTP redirect", ()
            elif message in {
                "Media HTTP retry state updated",
                "Media HTTP redirect",
                "Media HTTP transport event (details redacted)",
            }:
                record.msg, record.args = message, ()
            elif record.exc_info or record.exc_text or record.stack_info:
                name = (
                    getattr(record.exc_info[0], "__name__", "")
                    if record.exc_info
                    else ""
                )
                record.msg = "Media HTTP transport error (type=%s)"
                record.args = (name if name in _EXCEPTIONS else "Exception",)
            elif (
                message == "Media HTTP transport error (type=%s)"
                and len(values) == 1
                and type(values[0]) is str
                and (values[0] in _EXCEPTIONS or values[0] == "Exception")
            ):
                pass
            else:
                record.msg, record.args = (
                    "Media HTTP transport event (details redacted)",
                    (),
                )
        elif record.name == "uvicorn.access":
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
            if (
                type(record.msg) is str
                and record.msg == "HTTP request completed (method=%s, status=%d)"
                and isinstance(record.args, tuple)
                and len(record.args) == 2
            ):
                # Another handler may have already sanitized this record.
                method, status = record.args
            else:
                values = record.args if isinstance(record.args, dict) else {}
                method, status = values.get("method"), values.get("status", 0)
            method = method if type(method) is str and method in _METHODS else "OTHER"
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
        elif (
            type(record.msg) is str
            and record.msg == "HTTP application exception (type=%s)"
            and isinstance(record.args, tuple)
            and len(record.args) == 1
            and type(record.args[0]) is str
            and (record.args[0] in _EXCEPTIONS or record.args[0] == "Exception")
        ):
            # Preserve only the fixed event and validated type on later handlers.
            pass
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
    # log_config=None (for example the dummy health server) uses current
    # handlers. Cover those without changing non-framework records or the
    # logger's level, routing, propagation, or lifecycle.
    handlers = set(logging.getLogger().handlers)
    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "uvicorn.asgi",
        "granian",
        "granian.access",
        "_granian",
        "urllib3",
        "urllib3.connectionpool",
        "urllib3.poolmanager",
        "urllib3.util.retry",
    ):
        handlers.update(logging.getLogger(name).handlers)
    for handler in handlers:
        if not any(
            isinstance(item, FrameworkPrivacyFilter) for item in handler.filters
        ):
            handler.addFilter(FrameworkPrivacyFilter())
    return config


def configure_granian_log_privacy():
    from granian.log import LOGGING_CONFIG

    return configure_framework_log_privacy(deepcopy(LOGGING_CONFIG))
