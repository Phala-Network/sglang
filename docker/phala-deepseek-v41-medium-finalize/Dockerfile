# Use the Dockerfile frontend bundled with the recorded BuildKit version.
ARG BASE_IMAGE=ghcr.io/phala-network/sglang:dev-dsv41-phala-r6-hicache-chunked@sha256:56bf231ff01dbf4296079dac9feb983918beb9563dda046bebff5916015e6efe
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
      org.opencontainers.image.title="Phala SGLang DeepSeek V4.1 medium target finalize" \
      org.opencontainers.image.description="Target-only 97-384 row MXFP4 finalize and TP all-reduce plane for DeepSeek V4.1" \
      org.opencontainers.image.url="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.version="dev-dsv41-phala-r7-medium-finalize" \
      org.opencontainers.image.revision="${PHALA_SOURCE_COMMIT}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Phala Network" \
      org.opencontainers.image.ref.name="dev-dsv41-phala-r7-medium-finalize" \
      org.opencontainers.image.documentation="${BUILD_INPUTS_URL}" \
      org.phala.source.repository="https://github.com/Phala-Network/sglang" \
      org.phala.source.commit="${PHALA_SOURCE_COMMIT}" \
      org.phala.source.tree.digest="${SOURCE_TREE_DIGEST}" \
      org.phala.patch.commit="${PHALA_PATCH_COMMIT}" \
      org.phala.base.image="ghcr.io/phala-network/sglang:dev-dsv41-phala-r6-hicache-chunked" \
      org.phala.base.digest="sha256:56bf231ff01dbf4296079dac9feb983918beb9563dda046bebff5916015e6efe" \
      org.phala.dockerfile.path="docker/phala-deepseek-v41-medium-finalize/Dockerfile" \
      org.phala.dockerfile.digest="${DOCKERFILE_DIGEST}" \
      org.phala.builder.platform="linux/amd64" \
      org.phala.buildkit.version="${BUILDKIT_VERSION}" \
      org.phala.source.date.epoch="${SOURCE_DATE_EPOCH}" \
      org.phala.build.inputs.url="${BUILD_INPUTS_URL}" \
      org.phala.build.inputs.digest="${BUILD_INPUTS_DIGEST}"

COPY docker/phala-deepseek-v41-medium-finalize/base.sha256 /opt/phala-release/dsv41-medium-finalize-base.sha256
RUN sha256sum -c /opt/phala-release/dsv41-medium-finalize-base.sha256

COPY python/sglang/kernels/ops/communication/all_reduce_fusion.py /sgl-workspace/sglang/python/sglang/kernels/ops/communication/all_reduce_fusion.py
COPY python/sglang/srt/environ.py /sgl-workspace/sglang/python/sglang/srt/environ.py
COPY python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py /sgl-workspace/sglang/python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py
COPY python/sglang/srt/models/deepseek_v2.py /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py
COPY test/registered/kernels/ops/communication/test_dsv41_medium_finalize_all_reduce.py /opt/phala-release/tests/test_dsv41_medium_finalize_all_reduce.py
COPY test/registered/unit/models/test_dsv41_medium_finalize_selection.py /opt/phala-release/tests/test_dsv41_medium_finalize_selection.py
COPY docker/phala-deepseek-v41-medium-finalize/source.sha256 /opt/phala-release/dsv41-medium-finalize-source.sha256
COPY docker/phala-deepseek-v41-medium-finalize/dependencies.json /opt/phala-release/dsv41-medium-finalize-dependencies.json
COPY docker/phala-deepseek-v41-medium-finalize/PATCH.md /opt/phala-release/dsv41-medium-finalize-patch.md
RUN sha256sum -c /opt/phala-release/dsv41-medium-finalize-source.sha256
RUN --network=none python3 - <<'PY'
import pathlib

files = [
    pathlib.Path("/sgl-workspace/sglang/python/sglang/kernels/ops/communication/all_reduce_fusion.py"),
    pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/environ.py"),
    pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py"),
    pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py"),
]
for source in files:
    compile(source.read_text(), str(source), "exec")
    cache_dir = source.parent / "__pycache__"
    for cached in sorted(cache_dir.glob(source.stem + ".*.pyc")):
        assert cached.parent == cache_dir and cached.is_file()
        cached.unlink()
PY
RUN --network=none hf --help >/dev/null && hf download --help >/dev/null
