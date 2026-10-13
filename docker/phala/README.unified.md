# Unified Runtime Build

This is the shared build path, not another model profile. Existing
profile-oriented preparation files are historical inputs and are not consumed
by this entry point. The only output repository is
`ghcr.io/phala-network/sglang`; source identity belongs to the full Phala engine.

## Prepare

Freeze a clean engine commit and patch-repository commit. For the R6 Governor
candidate, use engine `d752488ed6e55ff4bb68892e4aca8c22becea8f9` and serving
patches `bf43ae5913ea117f2938ce9f53369babc5088c87`; `stack.py verify` must
replay the active selector onto that engine tree. Build the complete SGLang
wheel from that engine with `SGLANG_BUILD_RUST_EXTS=all`; preserve the Cargo
locks and record the actual compiler/base environment. Do not reuse the older
`46107e6861` R6 developer wheel. Build or reuse the source-bound shared
XGrammar and KV wheels. Never substitute a Python-only wheel or stock XGrammar
for a missing component.

The input JSON uses schema `phala.sglang.unified-image-inputs.v1` and contains:

- `engine.commit` and `engine.version`, ending in `+phala.<commit-prefix-10>`.
- `patches_commit` and immutable `base_image` (`tag@sha256:...`).
- `kernel.version` and `kernel.wrapper_preimage_sha256`, read from that base.
- `wheels`: package name, path relative to the input file, and exact SHA256.
- `runtime_assets`: both Muse and Nemotron chat templates, each with a path
  relative to the input file and exact SHA256 of its published bytes.
- Optional `governor`: source repository, commit, Git tree, archive SHA256,
  version, ABI, `pig-governor` wheel hash, and native library hash. An
  enabled Governor input also requires `--governor-repo`; its default remains
  disabled until the deployment sets the runtime enable flag.
- Optional `shared_dependency_overrides`: protobuf, grpcio-tools, transformers,
  tokenizers and safetensors, each with a frozen version and reason. These are
  shared dependencies, not per-model options or permission to replace the
  Torch/native ABI.

```sh
python prepare_unified.py \
  --engine-repo /path/to/sglang \
  --patches-repo /path/to/sglang-serving-patches \
  --governor-repo /path/to/phala-inference-governor \
  --inputs /path/to/frozen-inputs.json \
  --output /new/build-context
```

Preparation executes the existing full-stack verifier. It checks wheel hashes,
metadata, every declared Rust extension's presence and changed runtime/JIT
source bytes against the engine. It extracts the matching kernel wrapper
directly from Git. It also verifies and stages both frozen chat templates at
their production `--chat-template` paths. When Governor is present it also
checks the clean source checkout, exact wheel and native hashes, version and
ABI, then stages both artifacts. Existing outputs and dirty source checkouts
are rejected.
Fixture wheels in the unit tests prove validation behavior, not runtime behavior.

## Build

Use the exact arguments recorded in `runtime-context.json`:

```sh
docker buildx build --platform linux/amd64 --network none --load \
  --build-arg SGLANG_BASE=PINNED_BASE \
  --build-arg ENGINE_COMMIT=FULL_ENGINE_COMMIT \
  --build-arg PATCHES_COMMIT=FULL_PATCH_COMMIT \
  --build-arg GOVERNOR_COMMIT=GOVERNOR_COMMIT \
  --build-arg GOVERNOR_VERSION=0.2.4 \
  --build-arg GOVERNOR_ABI=4 \
  --tag ghcr.io/phala-network/sglang:CANDIDATE_TAG \
  -f /new/build-context/Dockerfile.unified /new/build-context
```

The required base argument intentionally has no moving default. Wheels are
mounted only during the build. The runtime clears inherited `PYTHONPATH` and
uses the installed engine rather than an old editable source checkout.

The kernel wrapper is installed at build time with version/preimage/source
hash checks and an updated wheel RECORD. When Governor is present, the build
installs its hash-checked wheel, verifies the native library hash and ABI, and
sets `PIG_GOVERNOR_LIBRARY` to the staged library. The original binary package
version is recorded separately from the patched Python wrapper identity. A
failed Docker RUN discards the layer; sequential file updates are not claimed
to be a general filesystem transaction.

Governor remains disabled by default in the image. Runtime enablement, model
configuration, publication, deployment and routes remain separate concerns.

## Verify

Run on the authorized builder with the exact resulting image ID:

```sh
python verify_unified_image.py \
  --image sha256:EXACT_BUILDER_IMAGE_ID \
  --context /new/build-context/runtime-context.json \
  --output /new/final-image-checks
```

The final-image runner uses no source/executable mounts, package installation,
model downloads, credentials or SDK driver stubs. It checks direct `hf` help
entry points, SGLang CLI, installed versions and wrapper, four Rust imports,
CPU Torch/Rust cache round-trip, DeepSeek model import and image geometry,
native grammar bounds, generated protocols, exact chat-template bytes and
syntax, typed usage serialization, the optional Governor version/ABI/native
library contract, and an actual parser-error privacy canary.

Dependency findings are recorded, not converted to a pass. Inherited base
metadata conflicts require explicit classification; do not downgrade NCCL or
security-updated packages solely to silence pip. These CPU gates do not
qualify real CUDA, full serving behavior, hardware/topology, capacity or
production. Publication and GPU/rollout checks remain separate authorized
stages, with downloader and backend ultimately sharing one accepted digest.
