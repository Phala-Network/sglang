# SPDX-License-Identifier: Apache-2.0
# Request-static table selection adapted from vLLM PR #53964, head
# 7ba5fd5e671e210b9e3d3f423ff2e1bf48314f45 (Apache-2.0).

"""Admission policy for immutable per-request mRoPE YaRN profiles."""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any


@dataclass(frozen=True)
class YarnProfile:
    factor: float
    offset: int
    profile_id: str
    budget: int


@dataclass(frozen=True)
class RequestStaticYarnConfig:
    factors: tuple[float, ...]
    original_max_position: int
    profiles: tuple[tuple[float, int, str], ...]

    @classmethod
    def from_hf_config(cls, hf_text_config: Any):
        rope = getattr(hf_text_config, "rope_parameters", None) or getattr(
            hf_text_config, "rope_scaling", None
        )
        factors = rope.get("request_static_factors") if isinstance(rope, dict) else None
        if factors is None:
            return None
        normalized = tuple(float(factor) for factor in factors)
        if (
            not normalized
            or any(not math.isfinite(f) or f < 1 for f in normalized)
            or normalized != tuple(sorted(set(normalized)))
        ):
            raise ValueError(
                "request_static_factors must be finite, sorted, unique and >= 1"
            )
        if (
            rope.get("rope_type", rope.get("type")) != "yarn"
            or "mrope_section" not in rope
        ):
            raise ValueError("Request-static YaRN requires yarn mRoPE")
        if float(rope.get("factor", 1)) != normalized[-1]:
            raise ValueError(
                "Largest request-static factor must equal configured YaRN factor"
            )
        original = rope.get("original_max_position_embeddings")
        if not isinstance(original, int) or isinstance(original, bool) or original <= 0:
            raise ValueError(
                "Request-static YaRN requires a positive original context length"
            )
        profiles = []
        offset = 0
        for factor in normalized:
            rows = 4 * original * factor
            if not rows.is_integer():
                raise ValueError("Request-static YaRN must have integral table sizes")
            params = {k: v for k, v in rope.items() if k != "request_static_factors"}
            params["factor"] = factor
            identity = {
                "rope": params,
                "head_dim": getattr(hf_text_config, "head_dim", None),
                "rope_theta": params.get(
                    "rope_theta", getattr(hf_text_config, "rope_theta", None)
                ),
                "partial_rotary_factor": params.get(
                    "partial_rotary_factor",
                    getattr(hf_text_config, "partial_rotary_factor", None),
                ),
                "layout": "sglang-mrope-request-static-v1",
            }
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            profiles.append((factor, offset, "yarn:" + digest))
            offset += int(rows)
        return cls(normalized, original, tuple(profiles))

    def validate_model_support(self, architectures, is_multimodal: bool):
        # These wrappers and their embedded MTP consume mrope_positions. Bare
        # CausalLM forwards do not, and prefill graphs allocate that input only
        # in multimodal mode. Reject unsupported combinations at model load.
        if not architectures or architectures[0] not in (
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        ):
            raise ValueError(
                "Request-static YaRN requires a Qwen3.5/3.8 ConditionalGeneration wrapper"
            )
        if not is_multimodal:
            raise ValueError(
                "Request-static YaRN requires multimodal mode; language-model-only "
                "and disabled multimodal mode are unsupported"
            )

    def select(self, budget: int, explicit_factor=None) -> YarnProfile:
        if not isinstance(budget, int) or budget < 0:
            raise ValueError("Request-static YaRN requires a nonnegative token budget")
        if explicit_factor is not None:
            if isinstance(explicit_factor, bool) or not isinstance(
                explicit_factor, (int, float)
            ):
                raise ValueError(
                    "custom_params.yarn_factor must be a configured numeric factor"
                )
            if explicit_factor not in self.factors:
                raise ValueError(
                    "custom_params.yarn_factor must be a configured factor"
                )
        for factor, offset, profile_id in self.profiles:
            if explicit_factor is not None and factor != explicit_factor:
                continue
            if budget <= self.original_max_position * factor:
                return YarnProfile(factor, offset, profile_id, budget)
        raise ValueError(
            "Prompt plus output budget exceeds the selected YaRN profile capacity"
        )


def admit_request_static_yarn(req, config, context_length: int):
    """Select once, before output clipping, cache lookup or chunked prefill."""
    explicit = (req.sampling_params.custom_params or {}).get("yarn_factor")
    if config is None:
        if explicit is not None:
            raise ValueError("custom_params.yarn_factor requires request-static YaRN")
        return
    if getattr(req, "rope_profile", None) is not None:
        return
    prompt_len = len(req.origin_input_ids)
    output_budget = req.sampling_params.max_new_tokens
    # None is the existing SGLang unbounded-output contract: reserve the whole
    # remaining server context rather than silently treating it as one token.
    if output_budget is None:
        output_budget = max(0, context_length - prompt_len)
    budget = prompt_len + output_budget
    if budget > context_length:
        raise ValueError(
            "Prompt plus requested output budget exceeds server context length"
        )
    profile = config.select(budget, explicit)
    mm = getattr(req, "multimodal_inputs", None)
    if mm is not None and mm.mrope_positions is not None:
        positions = mm.mrope_positions
        max_position = int(positions.max()) if positions.numel() else -1
        if (
            max_position + output_budget
            >= 4 * config.original_max_position * profile.factor
        ):
            raise ValueError(
                "Multimodal positions exceed the selected YaRN table capacity"
            )
    # Preserve the caller namespace without ambiguous string concatenation.
    req.extra_key = json.dumps(
        [req.extra_key, profile.profile_id], separators=(",", ":")
    )
    req.rope_profile = profile
