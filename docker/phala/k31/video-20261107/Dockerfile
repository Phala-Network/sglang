# syntax=docker/dockerfile:1.7
# Python-only video adaptation over accepted image3; no installer/native rebuild.
FROM ghcr.io/phala-network/sglang:phala-dev-kimi-k31-3@sha256:08acd79a162875bc9bfded2ff42cd526a8b0b4e1c4f96e32c1997b61246f226b
ARG PHALA_ENGINE_REVISION
ARG PHALA_RELEASE_VERSION=phala-dev-kimi-k31-4
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${PHALA_ENGINE_REVISION}" \
      org.opencontainers.image.version="${PHALA_RELEASE_VERSION}" \
      io.phala.video.base="sha256:08acd79a162875bc9bfded2ff42cd526a8b0b4e1c4f96e32c1997b61246f226b"
WORKDIR /sgl-workspace/sglang
COPY rootfs/ /
RUN PYTHONDONTWRITEBYTECODE=1 python3 -B /opt/phala/k31-video/verify_installed.py
