# SPDX-License-Identifier: Apache-2.0
"""Model-only Xing RoPE addressing and Q scaling before FP8 conversion."""

import torch


def request_static_yarn_positions(forward_batch, physical_positions):
    positions = getattr(forward_batch, "rope_positions", None)
    return physical_positions if positions is None else positions


def scale_request_static_yarn_query(query, forward_batch, module):
    """Keep the native scalar/short Q; only long queries get the scale ratio.

    Called after MLA absorption, before backend Q quantization; MHA calls it
    on its projected BF16 Q before a fused backend RoPE/quantization hook.
    Neither K nor persistent latent KV is scaled. The table address selects
    the profile, so mixed batches and graph replay need only one model input.
    """
    ratio = getattr(module, "request_static_yarn_scale_ratio", None)
    if ratio is None:
        return query
    positions = getattr(forward_batch, "rope_positions", None)
    if positions is None:
        raise RuntimeError("Xing request-static YaRN model positions are missing")
    if query.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise RuntimeError("Xing request-static YaRN Q must be scaled before quantization")
    if query.shape[0] != positions.numel():
        raise RuntimeError("Xing request-static YaRN query/token layout mismatch")
    long_mask = (positions >= module.request_static_yarn_long_offset).view(
        -1, *([1] * (query.ndim - 1))
    )
    # FP32 multiplication matches the backend's FP32 scalar precision. Select
    # the original short value exactly, including signed zero, before cast.
    scaled = (query.float() * ratio).to(query.dtype)
    return torch.where(long_mask, scaled, query)
