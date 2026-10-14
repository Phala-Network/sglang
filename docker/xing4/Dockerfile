ARG BASE_IMAGE=ghcr.io/phala-network/sglang:phala-v0.5.21-7@sha256:6a74c66ab1c44d32d07d8f227fc5074a43a330251ea3e2b37ebe731a6178e5ee
FROM ${BASE_IMAGE}

ARG ENGINE_COMMIT
ARG PATCHES_COMMIT
ARG RELEASE_VERSION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_COMMIT}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.phala.serving-patches.revision="${PATCHES_COMMIT}" \
      org.phala.base-engine.revision="c2ce5382dad45fd85e39f5d4e1d93165225eb34d" \
      org.phala.upstream.revision="e00930c5489053f26d86b179cee0d087f846acbb"

COPY . /opt/phala-xing4/
RUN python /opt/phala-xing4/docker/xing4/install_source.py \
    && hf --help >/dev/null \
    && hf download --help >/dev/null \
    && sglang serve --help >/dev/null \
    && curl --version >/dev/null

ENTRYPOINT ["sglang", "serve"]
