"""Bounded fatal-worker diagnostics without traceback locals or request data."""

import json
import os
from pathlib import Path
import re

_CREDENTIAL_ENVS = (
    "TOKEN", "SGLANG_API_KEY", "SGLANG_ADMIN_API_KEY", "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN", "PHALA_CLOUD_API_KEY",
)


def scheduler_exception_diagnostic(error, *, startup, credentials=()):
    """Only constructor failures may include a scrubbed exception message.

    The event loop can carry private prompts/IDs in exception messages. Its
    diagnostic records type and code locations only. Neither phase emits
    frame locals, source lines, ServerArgs, request objects or environment.
    """
    secrets = [value for value in credentials if isinstance(value, str) and value]
    secrets.extend(os.environ[name] for name in _CREDENTIAL_ENVS if os.environ.get(name))

    def scrub(value):
        for secret in sorted(set(secrets), key=len, reverse=True):
            value = value.replace(secret, "<redacted>")
        return re.sub(r"(?i)\bbearer\s+[^\s,;'\"]+", "Bearer <redacted>", value)

    frames = []
    weight_context = {}
    tb = error.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        # Paths identify code, never local variables or the source line.
        filename = code.co_filename.replace("\\", "/")
        filename = (
            "sglang/" + filename.rsplit("/sglang/", 1)[1]
            if "/sglang/" in filename else Path(filename).name
        )
        frames.append({"file": filename, "line": tb.tb_lineno, "function": code.co_name})
        # Only trusted weight-loader frames can contribute parameter identity
        # and tensor metadata. Never stringify tensors or collect frame locals.
        module = tb.tb_frame.f_globals.get("__name__", "")
        if startup and module in (
            "sglang.srt.models.xing4_0",
            "sglang.srt.models.deepseek_common.deepseek_weight_loader",
            "sglang.srt.model_loader.weight_utils",
            "sglang.srt.layers.linear",
        ):
            for key in ("name", "target_name"):
                name = tb.tb_frame.f_locals.get(key)
                if isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.]{1,256}", name):
                    weight_context[key] = name
            for key in ("param", "loaded_weight"):
                tensor = tb.tb_frame.f_locals.get(key)
                if type(tensor).__module__ in ("torch", "torch.nn.parameter"):
                    weight_context[key + "_shape"] = [int(dim) for dim in tensor.shape]
                    weight_context[key + "_dtype"] = str(tensor.dtype)
        tb = tb.tb_next
    diagnostic = {
        "phase": "startup" if startup else "event_loop",
        "exception_type": type(error).__module__ + "." + type(error).__name__,
        "frames": frames[-24:],
    }
    if startup:
        if weight_context:
            diagnostic["weight_context"] = weight_context
        try:
            message = scrub(str(error))
        except Exception:
            message = "<message unavailable>"
        diagnostic["message"] = message[:1024]
    # Also remove an exact credential from unusual code filenames/class names.
    return scrub(json.dumps(diagnostic, ensure_ascii=True, separators=(",", ":")))
