# Xing4 on the maintained official SGLang v0.5.21 line

The complete source includes the PR39793 Xing MLA/MoE/mHC port, stable-interface
repairs, a registered official SentencePiece tokenizer, and request-static
YaRN 64/128. This continues the documented v0.5.21 compatibility exception;
shared v0.5.20 main uses incompatible private native/Governor interfaces.

Use the same immutable runtime image for BF16/FP8 weights and BF16/E4M3 KV.
The image uses `ENTRYPOINT ["sglang", "serve"]`: its command is flags only.
The pinned base, committed-only context exporter and build-time installed-byte
checks are in [Dockerfile](Dockerfile), [prepare_context.py](prepare_context.py)
and [install_source.py](install_source.py). Source/patch commits, Release,
immutable digest and validation are recorded in the published Release.

Dynamic serving requires BF16 activations, explicit `trtllm_mla` prefill/decode,
page64, and `--context-length 524288 --json-model-override-args` with one JSON
argument:

```json
{"max_position_embeddings":524288,"rope_scaling":{"rope_type":"yarn","factor":128,"request_static_factors":[64,128],"original_max_position_embeddings":4096,"beta_fast":32,"beta_slow":1,"mscale":1,"mscale_all_dim":1}}
```

The original input plus output budget selects factor64 through262144 and
factor128 through524288; larger budgets fail. Accepted profiles stay immutable
across chunks/retraction, and full math identity isolates prefix caches while
preserving caller namespace and cache salt. Independent KV padding/headroom
does not change the public budget or silently clip output. Native omitted
output keeps its default128; native explicit null and Chat omitted/null fill
the remaining public context. Metadata reports `yarn_profile` identity, factor,
budget, capacity and `reserved_tokens`. Explicit selection uses
`sampling_params.custom_params.yarn_factor` for native requests or top-level
`custom_params.yarn_factor` for Chat. Use
`chat_template_kwargs.spaces_between_special_tokens=false` for Chat decoding.

Static64 uses official unmodified model config and context262144. Static128
uses the JSON above without `request_static_factors`; it is a control, not a
request-static endpoint. Speculation, CP/DP/PP, LoRA, PD and offload are not
qualified for this dynamic path and are rejected before weight allocation.

Protected `TOKEN` environment authentication requires `PIG_AUTH_FROM_TOKEN=1`.
Model mounts contain only weights/JSON/SentencePiece/Jinja data and use
`trust_remote_code=false`. Container host is `0.0.0.0`; publish only approved
loopback/private addresses. Every public ingress path must authenticate.

[verify_cpu.sh](verify_cpu.sh) runs source or actual installed-package checks
on builder201. CPU driver stubs permit native imports only; CUDA/TileLang,
mixed graph/kernel equivalence, real512K, quality and capacity remain GPU gates.
The unified [ordered patch stack](https://github.com/Phala-Network/sglang-serving-patches/tree/codex/xing4-v0521-20261014/versions/v0.5.21)
preserves the published49-step prefix. No embedded Governor is selected.
