# Kimi model checkpoint-cache release

This opt-in repair retains the v0.5.21 ref5/DFlash mask maintenance runtime,
its dependencies and ordinary multithreaded safetensors loader. It adapts
[upstream PR40594](https://github.com/sgl-project/sglang/pull/40594) while
preserving the existing boolean shard-cache option and public CLI schema.

Set `drop_checkpoint_cache_after_model_load: true` in the JSON value of
`--model-loader-extra-config`, alongside `enable_multithread_load: true` and
`num_threads: 12`. After each complete model load/postprocess, synchronize
CUDA copies and the world CPU group, partition the current checkpoint files
among local ranks, advise DONTNEED, then synchronize before returning.
Model mode suppresses the old shard option if both are requested.

Supported here: default AUTO/SAFETENSORS loading, with actual safetensors files
on homogeneous ranks. PT fallback, FASTSAFETENSORS, ModelOpt and startup-overlap
use with this option are rejected. The option defaults off. This is an advisory
release of checkpoint file cache, not deletion or a global drop_caches action.
It does not change KV pools, RDMA registrations, serving kernels or PIG.

CPU qualification: 34 focused tests, plus a two-process real Gloo/mmap test
with exact copied values and full fixture file-cache eviction. CPU evidence
does not qualify CUDA/TP8, a full Kimi cold start, runtime latency or throughput.
A production rollout must still measure the entire load/pool-creation interval.
Memory pressure before model load completes may remain.

Build from the immutable runtime source commit: export the three paths in
`base-files.json` under `runtime/` in a temporary context, and copy this
Dockerfile, `base-files.json` and `install.py` into its root. Pass that complete
source commit as `ENGINE_REVISION`. The installer verifies the pinned base
files before replacing them. Final-image checks run without runtime source
mounts. Source/Release publication, image delivery and production acceptance
remain separate stages; never overwrite an existing image tag.
