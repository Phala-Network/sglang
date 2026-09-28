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
