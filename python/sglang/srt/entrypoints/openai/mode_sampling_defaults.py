"""Opt-in checkpoint sampling defaults selected by the resolved thinking mode."""

_SAMPLING_FIELDS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "presence_penalty",
    "frequency_penalty", "repetition_penalty",
})


def apply_mode_sampling_defaults(request, params, model_config, reasoning):
    if getattr(model_config, "sampling_defaults", "model") != "model":
        return params
    profiles = getattr(model_config.hf_config, "sglang_sampling_defaults_by_mode", None)
    if not profiles or not isinstance(reasoning, bool):
        return params
    mode = "thinking" if reasoning else "non_thinking"
    profile = profiles.get(mode, {})
    unknown = set(profile) - _SAMPLING_FIELDS
    if unknown:
        raise ValueError(f"Unsupported mode-specific sampling defaults: {sorted(unknown)}")
    result = dict(params)
    explicit = request.model_fields_set
    for name, value in profile.items():
        if name not in explicit or getattr(request, name, None) is None:
            result[name] = value
    return result
