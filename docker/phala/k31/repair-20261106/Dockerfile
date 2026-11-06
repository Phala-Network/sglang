# syntax=docker/dockerfile:1.7
# A Python-only layer over the accepted image2. No installers or native rebuild.
FROM ghcr.io/phala-network/sglang:phala-dev-kimi-k31-2@sha256:edc2bb25b3c7f7d67a902e0b7e3a99f1ef8a1371e8395f950ecbb06cb9f3db62
ARG PHALA_ENGINE_REVISION
ARG PHALA_RELEASE_VERSION=phala-dev-kimi-k31-3
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${PHALA_ENGINE_REVISION}" \
      org.opencontainers.image.version="${PHALA_RELEASE_VERSION}" \
      io.phala.repair.base="sha256:edc2bb25b3c7f7d67a902e0b7e3a99f1ef8a1371e8395f950ecbb06cb9f3db62"
WORKDIR /sgl-workspace/sglang
COPY rootfs/ /
RUN PYTHONDONTWRITEBYTECODE=1 python3 -B /opt/phala/k31-repair/verify_installed.py
