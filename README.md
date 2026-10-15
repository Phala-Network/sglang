# Phala SGLang

Phala's downstream [SGLang](https://github.com/sgl-project/sglang) engine is based
on upstream **v0.5.20**. It integrates shared serving fixes, model compatibility
changes and explicit opt-in [Governor](https://github.com/Phala-Network/phala-inference-governor)
hooks into one complete source tree.

This repository owns engine source. [TAIL](https://github.com/Phala-Network/pig-tail)
owns transport/attestation; Governor owns its controller and adapter. Model
weights, deployment configuration and rollout acceptance remain separate.

## Choose a version

`main` is the shared integration branch. Immutable
[source tags](https://github.com/Phala-Network/sglang/tags) identify release inputs;
[GHCR images](https://github.com/Phala-Network/sglang/pkgs/container/sglang) live at
`ghcr.io/phala-network/sglang`. Select an image digest and component identities
from its release receipt. A source tag or upstream feature list alone does not
establish hardware/model qualification.

The officialv0.5.21 compatibility maintenance line uses separately pinned native
interfaces; it does not replace this sharedv0.5.20 tree. Its
[Xing native EAGLE budget release](https://github.com/Phala-Network/sglang/releases/tag/phala-v0.5.21-xing4-yarn-rc5.3)
binds complete source48e9fc7d56930660999dc108cb5ecddee5245715 and the
[60-step ordered patch record](https://github.com/Phala-Network/sglang-serving-patches/blob/3f1ed96fa5d8f51b18fdce73d4b824c675f0af25/versions/v0.5.21/NATIVE-EAGLE-PUBLIC-BUDGET.md).
It repairs public262144 admission for the scoped BF16/E4M3 TP1 EAGLE recipe;
16 dedicated and166 affected CPU cases pass, with10 explicit CUDA skips.
Final-image and GPU/service gates remain separate. The
[existing compatibility exception](https://github.com/Phala-Network/sglang-serving-patches/blob/d0d730ee269fb67c06d5823499aa02f53ba5838e/versions/v0.5.21/XING4-DYNAMIC-YARN.md)
continues without changing shared Governor identity or previously published tags.

The PyPI package `sglang` is the upstream distribution; installing it does not
install this fork's changes. For development, follow the
[installation guide](https://docs.sglang.io/get_started/install.html) using this
checkout and the selected Phala release's pinned native dependencies.

## Run a selected image

This Bash template requires Docker with NVIDIA runtime, compatible GPUs, an
accepted image digest and a supported model. Set `SGLANG_IMAGE` to the receipt's
`ghcr.io/phala-network/sglang@sha256:...` reference and `MODEL` to your model:

```bash
: "${SGLANG_IMAGE:?Set an accepted immutable image reference}"
: "${MODEL:?Set a supported model}"
docker run --rm --gpus all --ipc=host -p 127.0.0.1:30000:30000 \
  --entrypoint sglang "$SGLANG_IMAGE" serve \
  --model-path "$MODEL" --host 0.0.0.0 --port 30000 \
  --schedule-policy hrrn --enable-metrics --enable-cache-report \
  --uvicorn-access-log-exclude-prefixes /metrics /health
```

After loading, `curl --fail http://127.0.0.1:30000/health` checks readiness.
Adjust topology, quantization, context and cache settings to your model/hardware.
This localhost example does not configure production authentication, attestation
or Governor; use the selected release's full configuration for those features.

## Contribute and validate

- [Phala responsibilities and release workflow](PHALA_REPOSITORY.md)
- [Historical source records](docs/phala/HISTORY.md)
- [CPU source checks](.github/workflows/phala-source-check.yml)
- [Full source lint](.github/workflows/lint.yml)
- [Upstream overview](README.upstream.md) and [documentation](https://docs.sglang.io/)

Keep common fixes shared and model-specific behavior scoped to its model or
topology. Run affected regressions before integration. GPU, native ABI,
final-image and deployment checks remain distinct from source lint.

## License

[Apache License 2.0](LICENSE). Preserve upstream attribution and applicable
third-party notices when redistributing the engine and its dependencies.

## Phala image builds

Image recipes and build tools belong in this engine repository; Compose and deployment records remain in [phala-models-compose](https://github.com/Phala-Network/phala-models-compose). See the [unified image builder](docker/phala/README.unified.md), [Kimi recovery recipe](docker/phala/kimi-recovery-3/README.md), and [DeepSeek offload repair recipe](docker/phala-offload-fix-20261012/README.md). Published image digests are unchanged by this source relocation.
