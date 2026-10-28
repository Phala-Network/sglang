# Kimi ModelExpress candidate

This opt-in image adds ModelExpress weight loading to the existing Kimi
0.5.21 modelcache1 runtime. It does not upgrade the engine to the repository's
current main runtime, change inference settings, enable ModelExpress by default,
or qualify GPU-to-GPU loading. Deployment remains a separate operation.

## Frozen inputs

- Engine source: `cebb7c6ce132ef1030b3dbb271862658257b7ac2`,
  [published maintenance release](https://github.com/Phala-Network/sglang/releases/tag/phala-v0.5.21-3-k3protocol3-ref5-modelcache1).
- Base: `ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol3-ref5-modelcache1@sha256:c8abc4835c98fb2015c3867ca731809ca494332adb033b850fe1dc8caecd9011`.
- ModelExpress: `df8bc7ca1730fd40a2af0416b589d84fcd1b090c` plus
  `modelexpress.patch`, packaged as `0.8.0+phala.mx1`.
- No SGLang, Torch, NIXL, Mooncake, XGrammar or native extension replacement.
  The existing engine release and its serving-patches replay remain the engine
  source identity; this recipe only changes the external Python package layer.
  OCI `revision` identifies that engine, while `ai.phala.recipe.revision`
  identifies this recipe's main commit.

## Changes

1. Preserve model-level checkpoint-cache release when ModelExpress falls back
   to SGLang's native loader. The adapter records files but does not synchronize
   inside the rank-local fallback. At the common successful-load exits, every
   rank gathers the file union through the engine CPU process group, then calls
   the engine's existing synchronized, per-node cleanup. Mixed RDMA/native ranks
   therefore participate in the same collectives. All-RDMA loads do not advise
   any checkpoint files. Disabled cleanup retains upstream behavior.
2. Extend `MX_RDMA_NIC_PIN` with semicolon-separated per-device groups. Commas
   within each group select multiple UCX devices. For example, the two-GPU
   illustration `mlx5_0:1,mlx5_10:1;mlx5_1:1,mlx5_11:1` gives each GPU two HCAs.
   Existing single-device lists/auto/off semantics remain. Malformed groups fail
   before creating an agent. Real eight-rank mappings require topology review;
   this syntax alone does not prove UCX multi-rail bandwidth or RoCE health.
3. Allow protobuf 7 and make ModelStreamer/InstantTensor installation extras.
   No installed dependency is downgraded. Required dependency checks, generated
   protocol imports and an actual gRPC/protobuf round trip qualify the packaged
   P2P/native path; optional streaming integrations are not qualified.

The image disables InstantTensor and artifact transfer by default. It contains
no fastsafetensors and omits unused automatic RL plugin entrypoints. SGLang
selects ModelExpress explicitly through its existing loader interface; ordinary
startup does not acquire an RL logging/hook callback. Main weights are the intended P2P target; the independent
DFlash checkpoint remains native. GPU manifests, outputs, DFlash behavior,
mixed-strategy GPU recovery and donor inference impact still require testing.

## Build and image checks

Use a dedicated CPU builder directory. `prepare.py` verifies the fixed upstream
archive and applies the patch without fuzz. Build the wheel with the pinned base
using `pip wheel --no-deps --no-build-isolation`, writing it to this recipe's
`dist/`. Build the image once after source checks pass:

```sh
python3 prepare.py --output /path/to/new-mx-source
docker run --rm --network none --entrypoint python3 \
  -v /path/to/new-mx-source:/mx -v "$PWD/dist":/dist \
  PINNED_BASE -m pip wheel --no-deps --no-build-isolation --wheel-dir /dist /mx
docker build --network none --build-arg RECIPE_COMMIT=FULL_MAIN_COMMIT \
  -t ghcr.io/phala-network/sglang:phala-kimi-modelexpress-20261029-rc1 .
```

Wheel preparation may need a writable copy of the source for setuptools;
development bind mounts are not part of the final image. Run these final-image
checks by immutable builder image ID, with no source/executable mounts:

- Direct `--entrypoint hf`: `--help` and `download --help`.
- Direct `--entrypoint sglang`: `serve --help`.
- `python3 /opt/phala/modelexpress-tests/smoke.py`.
- `SGLANG_USE_CPU_ENGINE=1 python3 -m pytest -q
  /opt/phala/modelexpress-tests/test_candidate.py`.

CPU tests exercise two real Gloo ranks for mixed native/RDMA and all-RDMA file
cleanup. Only file advice is intercepted; engine synchronization executes.
These tests do not instantiate a model or prove RDMA/GPU correctness.

## Later deployment prerequisites

At least one compatible replica must first load and publish its weights using
ModelExpress. Existing unmodified servers cannot act as sources. A coordinator
and supported metadata backend must be separately configured on the management
network. Keep the existing Mooncake shared-KV store/master configuration.

Use the SGLang `remote_instance`/`modelexpress` interface with transport `nixl`,
an explicit immutable model revision, matching image/backend settings on both
ends, and the correct internal readiness URL (Kimi currently uses port19601).
Verify 20/21 per-rank NIC reachability and the installed UCX RMA multi-rail
configuration. Neither source publication nor a transport timeout should make
normal serving depend on coordinator availability; failure paths require
bounded validation before production promotion.

Keep the native fallback and local checkpoint. First-node cold starts and a new
incompatible software/layout generation do not receive a warm-peer benefit.
No inference speedup, five-minute startup, or production acceptance is claimed
by publishing this image.
