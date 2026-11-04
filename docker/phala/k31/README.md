# Kimi K3.1 compatibility image

The official K3.1 image contains a dedicated source snapshot and native
extensions incompatible with the Phala 0.5.20 main runtime. This scoped variant
preserves those dependencies. It does not upgrade other main consumers.

- Official source: `afcd7d6b487a9cbe73e25941971b4cc107e6f9ae` (image-exported Git bundle).
- Its parent is public upstream `db0dfc72286e6322fb79762d4e2d568b438af66c`;
  recover that history when importing the shallow image checkout.
- Official amd64 base: `lmsysorg/sglang:dev-kimi-k31-pro@sha256:4a098ad7414702a2271296ac148854448d10710ec3554ac93b57ef194999305a`.
- Upstream package metadata: `0.0.0.dev1+g41208373c.d20261104`; this is a developer
  K3.1 image, not a claimed SGLang 0.5.21/0.5.22 release.
- Native FlashMLA 96-head sparse decode/prefill packing, FlashInfer DynBlock,
  Rust TreeCore, Prims-TS, DeepGEMM and Dynamo remain as supplied by the official image.
- Mooncake CUDA13 `0.3.13.post1`, XGrammar `0.2.7` and FlashInfer `0.7.2` are retained.

The only Python runtime repair requires exact object-byte counts before reporting
a Mooncake cache restore as successful. Short, missing and oversized native
results cannot expose partially restored hybrid pages. PUT behavior is retained.
The source has native HRRN, metrics, readiness, Kimi protocol parsing and DFlash
grammar-mask reset; the old protocol/ref2/ref3 patch stack is not applied.

Build on the authorized CPU builder using a clean archive of the release commit:

```sh
docker build -f docker/phala/k31/Dockerfile \
  --build-arg PHALA_ENGINE_REVISION=<complete-engine-commit> \
  --build-arg PHALA_RELEASE_VERSION=<immutable-release-tag> \
  -t ghcr.io/phala-network/sglang:<immutable-release-tag> .
```

No dependency installer runs during build or production startup. The complete
tracked source replaces the Python tree at build time; existing compiled native
extensions remain. Final-image qualification uses no source/executable mounts.
Test the official image verifier, focused Mooncake/Kimi/DFlash regressions,
`sglang serve --help`, and direct `hf --help` / `hf download --help` entrypoints.
CPU checks do not establish GPU/model, shared KV, RoCE or production acceptance.

Runtime requirements include device access to `/dev/infiniband`, unconfined NUMA
allocation with the appropriate capabilities, fixed master control address,
matching HCA names/subnets, and `MC_ENABLE_DEST_DEVICE_AFFINITY=1`. The last flag
prevents normal same-topology transfers from choosing an unrelated peer HCA;
it is not a guarantee that every failure/retry remains on the same subnet.
The old `MC_RDMA_RAIL_GROUPS` switch is absent from this native package and must
not be reported as enabled. Record actual dual-rail transfer evidence at rollout.

The image may contain an upstream fastsafetensors package, but formal production
loading explicitly uses the standard safetensors loader. No runtime source
overlay, startup package install, strict-level2 override or dependency downgrade
is part of this release.
