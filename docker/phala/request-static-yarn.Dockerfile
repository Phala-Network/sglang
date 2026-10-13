ARG BASE_IMAGE=ghcr.io/phala-network/sglang:phala-v0.5.21-5@sha256:26ccb804414ee61251af7c51d02da39185b11fd625933809d345dcb082e800a3
FROM ${BASE_IMAGE}

ARG ENGINE_COMMIT
ARG PATCHES_COMMIT
ARG RELEASE_VERSION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_COMMIT}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.phala.serving-patches.revision="${PATCHES_COMMIT}"

# Context contains only source from ENGINE_COMMIT, its focused tests, LICENSE,
# this installer and a generated source-manifest.json. Installation is build-only.
COPY . /opt/phala-request-yarn/
RUN python /opt/phala-request-yarn/docker/phala/install_request_static_yarn.py

ENTRYPOINT ["sglang", "serve"]
