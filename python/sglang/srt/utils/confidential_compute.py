# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Detect GPU CC for software dispatch; this does not change GPU security mode."""

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def is_confidential_compute() -> bool:
    forced = os.environ.get("SGLANG_CONFIDENTIAL_COMPUTE")
    if forced is not None:
        if forced not in ("0", "1"):
            raise ValueError("SGLANG_CONFIDENTIAL_COMPUTE must be '0' or '1'")
        return forced == "1"

    import torch

    if not torch.cuda.is_available():
        return False
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            state = pynvml.nvmlSystemGetConfComputeState()
            return int(state.ccFeature) != 0
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        logger.debug("GPU CC detection unavailable: %r", exc)
        return False
