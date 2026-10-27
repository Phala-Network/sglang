# Request-static YaRN runtime

This release extends the existing official-v0.5.21 compatibility maintenance
line with request-fixed 1x/2x/4x YaRN for the Qwen3.5-family multimodal path.
The ordinary Qwen3.8-27B-FP8 model was qualified with FP8 E4M3 KV, FA3, EAGLE
and server context 1,048,576. It retains the complete phala-v0.5.21-5 runtime,
its native extensions and official dependencies; no embedded Governor is added.

The engine's shared main uses the v0.5.20 custom XGrammar/native/Governor
combination. Its private XGrammar 0.2.6+phala.union1 differs from this line's
official 0.2.7. Combining those unported interfaces would change the tested
runtime. This release therefore continues the established v0.5.21 maintenance
line rather than replacing main or claiming unrelated model qualification.

The HRRN priority maintenance release adds request-priority ordering within
the existing runtime. Enable `--enable-priority-scheduling`,
`--disable-priority-preemption` and `--default-priority-value 1`; higher
numeric values run first, with HRRN ordering within each tier. The large-queue
FCFS fallback also preserves priority and FCFS order within a tier. With
priority scheduling disabled, the existing ordering is unchanged. This is
the equivalent of engine PR124 plus the over-128-request fallback repair.

Use an explicit output budget when short requests should select 1x. Automatic
admission chooses the smallest profile covering the expanded input plus the
original output budget. The global context limit separately reserves the four
qualified EAGLE internal slots. `custom_params.yarn_factor` accepts 1, 2 or 4;
chat requests place it at the top level, native `/generate` inside
`sampling_params`. Profiles stay fixed for the request and isolate KV/SSM
cache identities. Illegal or insufficient profiles return HTTP 400 and retire
without entering model execution. Sessions, LMCache and PD modes are rejected
when request-static YaRN is enabled.

Set `text_config.rope_parameters.request_static_factors` to `[1, 2, 4]` through
`--json-model-override-args`, with `rope_type: yarn`, `factor: 4.0`,
`original_max_position_embeddings: 262144` and the model's existing mRoPE
parameters retained. Server context 1,048,576 allows input plus declared
output up to 1,048,572 with the qualified four EAGLE internal slots; pure input
has its separate guard. Retain `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`.
The release notes contain the exact qualified override and launch parameters.

`request-static-yarn.Dockerfile` builds from the immutable phala-v0.5.21-5 base.
Prepare a minimal context containing the 13 changed runtime modules, focused
tests, root LICENSE, the recipe/installer and `source-manifest.json` with
`engine_commit`, `tested_candidate_identity`, and `runtime_sha256`. Read every
input from the exact engine Git commit. The installer verifies actual installed
hashes, module resolution and updates the wheel RECORD during build; no runtime
installation or source mount is used. Base distribution/native versions stay
unchanged and OCI revision identifies the complete new engine source.

Build on the authorized builder:

```sh
docker build --network none \
  --build-arg ENGINE_COMMIT=FULL_ENGINE_COMMIT \
  --build-arg PATCHES_COMMIT=FULL_PATCHES_COMMIT \
  --build-arg RELEASE_VERSION=IMMUTABLE_RELEASE_TAG \
  -t ghcr.io/phala-network/sglang:IMMUTABLE_RELEASE_TAG \
  -f docker/phala/request-static-yarn.Dockerfile .
```

Qualify the actual image with direct `hf --help`, `hf download --help`,
`sglang serve --help`, installed source/dependency equivalence and focused
policy/admission regressions. Source tests, image publication and service
acceptance are separate. GPQA 166/198 belongs to context 1,000,000; the
1,048,576 addendum passed its five scoped runtime checks. TAU was not executed
because independent simulator credentials were unavailable.
