# Kimi K3.1 compatibility image

The official K3.1 image contains a dedicated source snapshot and native
extensions incompatible with the Phala 0.5.20 main runtime. This scoped variant
preserves those dependencies, replacing only the Mooncake Store/Transfer
Engine client modules and their private dependencies. Other main consumers
keep their existing runtime.

- Official source: `afcd7d6b487a9cbe73e25941971b4cc107e6f9ae` (image-exported Git bundle).
- Its parent is public upstream `db0dfc72286e6322fb79762d4e2d568b438af66c`;
  recover that history when importing the shallow image checkout.
- Official amd64 base: `lmsysorg/sglang:dev-kimi-k31-pro@sha256:4a098ad7414702a2271296ac148854448d10710ec3554ac93b57ef194999305a`.
- Upstream package metadata: `0.0.0.dev1+g41208373c.d20261104`; this is a developer
  K3.1 image, not a claimed SGLang 0.5.21/0.5.22 release.
- Native FlashMLA 96-head sparse decode/prefill packing, FlashInfer DynBlock,
  Rust TreeCore, Prims-TS, DeepGEMM and Dynamo remain as supplied by the official image.
- Mooncake CUDA13 `0.3.13.post1` retains its API with the opt-in GID rail-group repair.
  Its fixed source/artifact identities are in [native-client.json](native-client.json).
  Official torch, EP, PG, allocator and helper modules remain unchanged.
- XGrammar `0.2.7` and FlashInfer `0.7.2` remain as supplied.

The cache-read repair requires exact object-byte counts before reporting
a Mooncake cache restore as successful. Short, missing and oversized native
results cannot expose partially restored hybrid pages. PUT behavior is retained.
The source has native HRRN, metrics, readiness, Kimi protocol parsing and DFlash
grammar-mask reset; the old protocol/ref2/ref3 patch stack is not applied.

K3.1 tokenizer and processor implementations are packaged with SGLang and selected
explicitly for the K3 model type. Model checkpoint directories supply metadata
and BPE data; their `auto_map` entries do not execute checkpoint Python, including
when the caller sets `trust_remote_code=True`. The package records the original
HF revision and per-file identities.

Build on the authorized CPU builder using a clean archive of the release commit:

```sh
docker buildx build --load -f docker/phala/k31/Dockerfile \
  --build-context mooncake-client=<frozen-native-input-directory> \
  --build-arg PHALA_ENGINE_REVISION=<complete-engine-commit> \
  --build-arg PHALA_RELEASE_VERSION=<immutable-release-tag> \
  -t ghcr.io/phala-network/sglang:<immutable-release-tag> .
```

The named native input contains `native-client.tar.gz` and `cpu_smoke.py` from
the fixed Mooncake source. Build the artifact with that source's
`phala/k31/Dockerfile.builder`, or use the published artifact with the recorded
SHA256. The build checks archive/source/harness identities, installs only
`store.so`, `engine.so` and private libraries, and verifies that every other
official Mooncake native module remains byte-identical.

The complete tracked SGLang source replaces its Python tree at build time.
Production startup performs no dependency installation. Final-image
qualification uses no source/executable mounts or runtime package installation.
Test the official image verifier, focused Mooncake/Kimi/DFlash regressions,
the installed native P2P/TCP bytes and registered CPU pointer batch API,
`sglang serve --help`, and direct `hf --help` / `hf download --help` entrypoints.
Set `KIMI_K31_TEST_DATA` to the pinned data-only target/draft fixture for
`qualify.py`; this is required for actual config/tokenizer/processor selection,
UTF-8/XTML and image-preprocessing checks with checkpoint code execution blocked.
The native build also executes the 26 actual worker-policy CPU regressions.
CPU checks do not establish GPU/model, shared KV, RoCE or production acceptance.

Runtime requirements include device access to `/dev/infiniband`, unconfined NUMA
allocation with the appropriate capabilities, fixed master control address,
matching HCA names/subnets, `MC_ENABLE_DEST_DEVICE_AFFINITY=1`, and the opt-in
`MC_RDMA_RAIL_GROUPS=10.80.20.0/24;10.80.21.0/24`. Explicit GID groups constrain
first selection, retry and local handoff, including providers with different
HCA names. Unknown GIDs fail closed. `MC_ENABLE_HCA_PEER_AFFINITY` is not used
for CPU KV buffers. Record actual dual-rail transfer evidence at rollout.

The fixed official base ships torch 2.14.1 but only older PG bindings; importing
`mooncake.pg` reports that existing version mismatch. The original PG files
are retained and this Store client does not use them.

`fastsafetensors` 0.5.0 is installed from its fixed CPython 3.12 Linux wheel with
SHA256 validation and `--no-deps`; the other installed package versions and the
official base's existing dependency-check warnings remain unchanged. Final-image
qualification checks imports, SGLang bindings, CLI parsing and CPU nogds tensor
round trips. GPU loading requires a separate hardware smoke test. Standard
`--load-format safetensors` remains available; the optional nogds path is
`--load-format fastsafetensors --model-loader-extra-config '{"enable_gds":false}'`.
Production startup performs no source overlay, package install or dependency
downgrade.
