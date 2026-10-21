# Coordinated checkpoint-cache release

An opt-in addition to the existing v0.5.21 ref5/modelcache1 maintenance runtime.
No serving kernels, protocol code, dependencies, GPU cache capacities, Mooncake
allocator, or external PIG change. The existing model-end release remains.

The normal buffered mmap loader accepts `checkpoint_cache_batch_shards` in
`--model-loader-extra-config`, default zero (disabled). Enabling it requires
`drop_checkpoint_cache_after_model_load: true`, buffered loading, no checkpoint
prefetch, and a single-node TP-only world with a Gloo CPU group. AUTO must resolve
to safetensors. Unsupported paths fail explicitly. Existing options stay valid.

Ranks first agree on file identity/order and batch size. After each bounded
file batch, synchronize CUDA copies and exchange inode mappings across ranks.
Advise only completed files that no participating rank still maps, with one
rank per file. A second exchange waits for advisory calls before the next batch.
Retained tensor mappings defer to later batches and the existing model-end
release; this is not a hard bound if a model retains many CPU weight views.
New collectives have a 300-second wait limit; failure aborts startup, never
silently disables the protection. Closing a generator does not initiate a new
collective. CPU failure tests use a shorter timeout.

The local-unmapped retry in upstream PR45204 informed the lifetime investigation;
it is not copied as a cross-rank completion signal. Two real CPU/Gloo ranks,
actual DefaultModelLoader integration, and mincore probes cover normal/skewed
consumption, retained tensors, differing file order/batch sizes, missing/corrupt/
changed files, consumer failures, map-read failure and a missing participant.
41 focused unit tests pass. These do not prove CUDA/TP8 full-model startup
pressure, loading speed or serving throughput. Full startup acceptance remains
required before production promotion.

For the proposed Kimi verification, restore the ordinary thread setting to 12
and test 12 files per batch in the same necessary cold start. About 12 x 16 GiB of
checkpoint data plus retained views is a planning estimate, not a measured
resident-page cap. No fastsafetensors, global drop_caches or cache-size reduction.

Build from the immutable complete source commit: export paths listed in
`base-files.json` under `runtime/` alongside this recipe, pass `ENGINE_REVISION`,
and use the pinned modelcache1 base. The installer checks old file hashes before
replacement. Source tag/Release must exist before publishing the image.
