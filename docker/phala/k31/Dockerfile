# syntax=docker/dockerfile:1.7
# Preserve the exact K3.1 native stack; install reviewed Python source at build time.
FROM lmsysorg/sglang:dev-kimi-k31-pro@sha256:4a098ad7414702a2271296ac148854448d10710ec3554ac93b57ef194999305a
ARG PHALA_ENGINE_REVISION
ARG PHALA_RELEASE_VERSION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${PHALA_ENGINE_REVISION}" \
      org.opencontainers.image.version="${PHALA_RELEASE_VERSION}"
WORKDIR /sgl-workspace/sglang
# COPY preserves the official editable install's compiled Rust extension and
# generated version module, which are absent from the Git source archive.
COPY . /sgl-workspace/sglang/
RUN test -n "$PHALA_ENGINE_REVISION" \
    && rm -rf /sgl-workspace/sglang/.git \
    && printf '%s\n' "$PHALA_ENGINE_REVISION" > /sgl-workspace/sglang/.k31-source-commit \
    && python3 /opt/k31-image/verify_image.py
