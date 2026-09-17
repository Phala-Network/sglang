# Use the Dockerfile frontend bundled with the recorded BuildKit version.
ARG BASE_IMAGE=ghcr.io/phala-network/sglang:dev-dsv41-phala-r4-async-chat@sha256:008012d040b4e954d7673e6767373acbfebe903f4b600bcf8ded876e2c5e762b
FROM ${BASE_IMAGE}

ARG BUILD_INPUTS_DIGEST
ARG BUILD_INPUTS_URL
ARG BUILDKIT_VERSION
ARG CREATED
ARG DOCKERFILE_DIGEST
ARG PHALA_PATCH_COMMIT
ARG PHALA_SOURCE_COMMIT
ARG SOURCE_DATE_EPOCH
ARG SOURCE_TREE_DIGEST

LABEL org.opencontainers.image.created="${CREATED}" \
      org.opencontainers.image.title="Phala SGLang DeepSeek V4.1 HiCache chunked write-through" \
      org.opencontainers.image.description="Back up completed chunked-prefill nodes through the existing HiCache write-through ACK path" \
      org.opencontainers.image.url="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.version="dev-dsv41-phala-r6-hicache-chunked" \
      org.opencontainers.image.revision="${PHALA_SOURCE_COMMIT}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Phala Network" \
      org.opencontainers.image.ref.name="dev-dsv41-phala-r6-hicache-chunked" \
      org.opencontainers.image.documentation="${BUILD_INPUTS_URL}" \
      org.phala.source.repository="https://github.com/Phala-Network/sglang" \
      org.phala.source.commit="${PHALA_SOURCE_COMMIT}" \
      org.phala.source.tree.digest="${SOURCE_TREE_DIGEST}" \
      org.phala.patch.commit="${PHALA_PATCH_COMMIT}" \
      org.phala.base.image="ghcr.io/phala-network/sglang:dev-dsv41-phala-r4-async-chat" \
      org.phala.base.digest="sha256:008012d040b4e954d7673e6767373acbfebe903f4b600bcf8ded876e2c5e762b" \
      org.phala.dockerfile.path="docker/phala-deepseek-v41-hicache-chunked/Dockerfile" \
      org.phala.dockerfile.digest="${DOCKERFILE_DIGEST}" \
      org.phala.builder.platform="linux/amd64" \
      org.phala.buildkit.version="${BUILDKIT_VERSION}" \
      org.phala.source.date.epoch="${SOURCE_DATE_EPOCH}" \
      org.phala.build.inputs.url="${BUILD_INPUTS_URL}" \
      org.phala.build.inputs.digest="${BUILD_INPUTS_DIGEST}"

COPY docker/phala-deepseek-v41-hicache-chunked/base.sha256 /opt/phala-release/hicache-chunked-base.sha256
RUN sha256sum -c /opt/phala-release/hicache-chunked-base.sha256

COPY python/sglang/srt/mem_cache/unified_radix_cache.py /sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_radix_cache.py
COPY docker/phala-deepseek-v41-hicache-chunked/source.sha256 /opt/phala-release/hicache-chunked-source.sha256
COPY docker/phala-deepseek-v41-hicache-chunked/dependencies.json /opt/phala-release/hicache-chunked-dependencies.json
COPY docker/phala-deepseek-v41-hicache-chunked/PATCH.md /opt/phala-release/hicache-chunked-patch.md
RUN sha256sum -c /opt/phala-release/hicache-chunked-source.sha256
RUN --network=none python3 - <<'PY'
import pathlib

source = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_radix_cache.py"
)
compile(source.read_text(), str(source), "exec")
cache_dir = source.parent / "__pycache__"
for cached in sorted(cache_dir.glob(source.stem + ".*.pyc")):
    assert cached.parent == cache_dir and cached.is_file()
    cached.unlink()
PY
RUN --network=none hf --help >/dev/null && hf download --help >/dev/null
