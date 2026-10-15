# SPDX-License-Identifier: Apache-2.0
# Request-static table selection adapted from vLLM PR #53964, head
# 7ba5fd5e671e210b9e3d3f423ff2e1bf48314f45 (Apache-2.0).

"""Admission policy for immutable per-request mRoPE and Xing MLA YaRN."""

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
    capacity: int = 0
    reserved_tokens: int = 0

    def metadata(self):
        return {
            "factor": self.factor,
            "id": self.profile_id,
            "identity": self.profile_id,
            "budget": self.budget,
            "admitted_budget": self.budget,
            "capacity": self.capacity,
            "reserved_tokens": self.reserved_tokens,
        }


@dataclass(frozen=True)
class RequestStaticYarnConfig:
    factors: tuple[float, ...]
    original_max_position: int
    profiles: tuple[tuple[float, int, str], ...]
    position_kind: str = "mrope"
    lookahead: int = 0

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
        is_xing = getattr(hf_text_config, "model_type", None) == "xing4_0"
        position_kind = "mla" if is_xing else "mrope"
        lookahead = rope.get("_request_static_yarn_lookahead", 0) if is_xing else 0
        if type(lookahead) is not int or lookahead not in (0, 8):
            raise ValueError("Xing request-static YaRN lookahead must be zero or eight")
        rope_type = rope.get("rope_type", rope.get("type"))
        if is_xing:
            if (
                rope_type not in ("yarn", "deepseek_yarn")
                or normalized != (64.0, 128.0)
                or "mrope_section" in rope
                or rope.get("original_max_position_embeddings") != 4096
                or rope.get("beta_fast", 32) != 32
                or rope.get("beta_slow", 1) != 1
                or rope.get("mscale", 1) != 1
                or rope.get("mscale_all_dim", 0) != 1
                or rope.get("attn_factor", 1) != 1
                or rope.get("extrapolation_factor", 1) != 1
                or not rope.get("apply_yarn_scaling", True)
                or not rope.get("apply_scale", True)
                or rope.get("rope_theta", getattr(hf_text_config, "rope_theta", 10000)) != 10000
                or rope.get("partial_rotary_factor", 1) != 1
                or getattr(hf_text_config, "qk_rope_head_dim", 64) != 64
                or getattr(hf_text_config, "qk_nope_head_dim", 128) != 128
                or not getattr(hf_text_config, "rope_interleave", True)
            ):
                raise ValueError(
                    "Xing request-static YaRN requires native factor 64 and long "
                    "factor 128, original 4096, and the official MLA RoPE parameters"
                )
        elif rope_type != "yarn" or "mrope_section" not in rope:
            raise ValueError("Request-static YaRN requires yarn mRoPE or Xing MLA")
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
            rows = (1 if is_xing else 4) * original * factor
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
                "layout": (
                    "sglang-xing-mla-request-static-v1"
                    if is_xing
                    else "sglang-mrope-request-static-v1"
                ),
            }
            if is_xing:
                # DeepSeek's attention constructor normalizes yarn to
                # deepseek_yarn in-place. They designate the same MLA math.
                params.pop("type", None)
                params["rope_type"] = "deepseek_yarn"
                identity["model_type"] = "xing4_0"
                identity["qk_nope_head_dim"] = getattr(
                    hf_text_config, "qk_nope_head_dim", 128
                )
                identity["qk_rope_head_dim"] = getattr(
                    hf_text_config, "qk_rope_head_dim", 64
                )
                identity["rope_interleave"] = True
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            profiles.append((factor, offset, "yarn:" + digest))
            offset += int(rows) + lookahead
        return cls(normalized, original, tuple(profiles), position_kind, lookahead)

    def validate_model_support(self, architectures, is_multimodal: bool):
        if self.position_kind == "mla":
            if not architectures or architectures[0] != "Xing4_0ForCausalLM":
                raise ValueError("Request-static MLA YaRN requires Xing4_0ForCausalLM")
            return
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

    def validate_runtime_support(self, args):
        """Fail before weight allocation for unported Xing runtime paths."""
        if self.position_kind != "mla":
            return
        backend = getattr(args, "attention_backend", None)
        prefill = getattr(args, "prefill_attention_backend", None) or backend
        decode = getattr(args, "decode_attention_backend", None) or backend
        if prefill != "trtllm_mla" or decode != "trtllm_mla":
            raise ValueError(
                "Xing request-static YaRN currently requires explicit TRTLLM MLA "
                "for both prefill and decode"
            )
        if getattr(args, "device", "cuda") != "cuda":
            raise ValueError("Xing request-static YaRN serving requires CUDA")
        if getattr(args, "dtype", "auto") not in ("auto", "bfloat16", "bf16"):
            raise ValueError("Xing request-static YaRN requires BF16 activations")
        if getattr(args, "kv_cache_dtype", "auto") not in (
            "auto", "bfloat16", "bf16", "fp8_e4m3"
        ):
            raise ValueError("Xing request-static YaRN supports BF16 or FP8 E4M3 KV")
        if (
            getattr(args, "tp_size", 1) not in (1, 2)
            or getattr(args, "pp_size", 1) != 1
            or getattr(args, "dp_size", 1) != 1
            or getattr(args, "attn_cp_size", 1) != 1
            or getattr(args, "dcp_size", 1) != 1
            or getattr(args, "enable_prefill_cp", False)
            or getattr(args, "enable_two_batch_overlap", False)
            or getattr(args, "enable_dp_attention", False)
            or getattr(args, "enable_dsa_prefill_cp", False)
            or getattr(args, "enable_mla_prefill_cp", False)
            or getattr(args, "enable_lora", False)
            or getattr(args, "disaggregation_mode", "null") != "null"
            or getattr(args, "enable_lmcache", False)
            or getattr(args, "enable_hierarchical_cache", False)
            or getattr(args, "enable_hisparse", False)
            or getattr(args, "cpu_offload_gb", 0)
        ):
            raise ValueError(
                "Xing request-static YaRN requires TP1/TP2, PP1, no CP/DP, "
                "LoRA, disaggregation or offload"
            )

        algorithm = getattr(args, "speculative_algorithm", None)
        if algorithm not in (None, "NONE"):
            if self.lookahead != 8:
                raise ValueError("Xing EAGLE request-static YaRN requires eight lookahead rows")
            if (
                algorithm != "EAGLE"
                or getattr(args, "tp_size", 1) != 1
                or getattr(args, "page_size", 64) != 64
                or getattr(args, "kv_cache_dtype", "auto") != "fp8_e4m3"
                or getattr(args, "quantization", None) is not None
                or getattr(args, "speculative_draft_attention_backend", None)
                not in (None, "trtllm_mla")
                or (
                    getattr(args, "speculative_num_steps", None),
                    getattr(args, "speculative_eagle_topk", None),
                    getattr(args, "speculative_num_draft_tokens", None),
                ) != (3, 1, 4)
                or getattr(args, "speculative_attention_mode", "prefill") != "prefill"
                or getattr(args, "speculative_adaptive", False)
                or getattr(args, "enable_multi_layer_eagle", False)
                or getattr(args, "speculative_use_rejection_sampling", False)
                or getattr(args, "speculative_draft_window_size", None) not in (None, 0)
            ):
                raise ValueError(
                    "Xing request-static EAGLE requires TP1, BF16 weights, FP8 E4M3 KV, "
                    "page64, TRTMLA and fixed single-layer EAGLE3/topk1/draft4 prefill"
                )

    def select(self, budget: int, explicit_factor=None, reserved_tokens=0) -> YarnProfile:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("Request-static YaRN requires a nonnegative token budget")
        if (
            isinstance(reserved_tokens, bool)
            or not isinstance(reserved_tokens, int)
            or reserved_tokens < 0
        ):
            raise ValueError("Request-static YaRN requires nonnegative reserved tokens")
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
            capacity = int(self.original_max_position * factor)
            # Xing's public RoPE context is actual input + requested output.
            # Paged-KV padding/headroom is an independent pool requirement.
            required = budget if self.position_kind == "mla" else budget + reserved_tokens
            if required <= capacity:
                return YarnProfile(
                    factor, offset, profile_id, budget, capacity, reserved_tokens
                )
        raise ValueError(
            "Prompt plus output budget exceeds the selected YaRN profile capacity"
        )


def admit_request_static_yarn(req, config, context_length: int, reserved_tokens=0):
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
    context_reserved = 0 if config.position_kind == "mla" else reserved_tokens
    if output_budget is None:
        output_budget = max(0, context_length - prompt_len - context_reserved)
    budget = prompt_len + output_budget
    if budget + context_reserved > context_length:
        raise ValueError(
            "Prompt plus requested output budget exceeds server context length"
        )
    profile = config.select(budget, explicit, reserved_tokens)
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
