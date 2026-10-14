FROM ghcr.io/phala-network/sglang:phala-v0.5.21-6@sha256:cf0cf7a61358162e31dd96cd5591e8e0c42d2b5fcd0dd570ed02a5212710a26e

ARG ENGINE_COMMIT
ARG PATCHES_COMMIT
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_COMMIT}" \
      org.phala.serving-patches.revision="${PATCHES_COMMIT}" \
      org.phala.base-engine.revision="80afb70020aad5ce1135fcbb00864c0d761c8413"

# This build context contains only source-bound GGUF deltas and focused tests.
# The pinned released image supplies the unchanged official native dependencies.
COPY . /opt/phala-model894-yarn/
RUN python /opt/phala-model894-yarn/docker/phala-model894-yarn/install.py \
    && hf --help >/dev/null \
    && hf download --help >/dev/null \
    && sglang serve --help >/dev/null \
    && curl --version >/dev/null

ENTRYPOINT ["sglang", "serve"]
