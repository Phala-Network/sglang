# SPDX-License-Identifier: Apache-2.0
"""Explicit environment auth and diagnostic-only secret redaction.

Runtime records and IPC retain actual credentials. Never feed the diagnostic
serializer back into a runtime configuration.
"""

from __future__ import annotations

import os

_AUTH_FIELDS = frozenset(("api_key", "admin_api_key", "ssl_keyfile_password"))
_AUTH_FLAGS = frozenset("--" + name.replace("_", "-") for name in _AUTH_FIELDS)


def handle_token_auth(server_args):
    """Resolve both HTTP credentials from TOKEN only after explicit opt-in."""
    mode = os.environ.get("PIG_AUTH_FROM_TOKEN")
    if mode in (None, "0"):
        return
    if mode != "1":
        raise ValueError("PIG_AUTH_FROM_TOKEN must be 0 or 1")

    token = os.environ.get("TOKEN")
    if not token or any(not 0x21 <= ord(char) <= 0x7E for char in token):
        raise ValueError(
            "PIG_AUTH_FROM_TOKEN requires a nonempty printable ASCII TOKEN without whitespace"
        )

    from sglang.srt.arg_groups.overrides import declare_resolution

    # Even matching explicit keys introduce a second source and can leak via
    # process argv. Raw None remains None after a declaration or pickle roundtrip.
    if server_args.api_key is not None or server_args.admin_api_key is not None:
        raise ValueError("Explicit API/admin keys cannot be combined with PIG_AUTH_FROM_TOKEN")
    declare_resolution(
        server_args, "handle_token_auth", api_key=token, admin_api_key=token
    )


def redact_auth_config(value):
    """Copy a diagnostic container, removing auth fields at every nesting level."""
    if isinstance(value, dict):
        return {
            key: ("[REDACTED]" if item is not None else None)
            if key in _AUTH_FIELDS
            else redact_auth_config(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_auth_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_auth_config(item) for item in value)
    return value


def redact_auth_argv(argv):
    """Hide explicit credential flags in stored diagnostic launch commands."""
    result = []
    hide_next = False
    for arg in argv:
        if hide_next:
            result.append("[REDACTED]")
            hide_next = False
        elif arg.startswith("--") and any(flag.startswith(arg) for flag in _AUTH_FLAGS):
            result.append(arg)
            hide_next = True
        elif "=" in arg and arg.startswith("--") and any(
            flag.startswith(arg.partition("=")[0]) for flag in _AUTH_FLAGS
        ):
            result.append(arg.partition("=")[0] + "=[REDACTED]")
        else:
            result.append(arg)
    return result
