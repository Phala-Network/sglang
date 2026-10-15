FROM ghcr.io/phala-network/sglang:phala-v0.5.21-7@sha256:6a74c66ab1c44d32d07d8f227fc5074a43a330251ea3e2b37ebe731a6178e5ee

ARG ENGINE_COMMIT
ARG PATCHES_COMMIT
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_COMMIT}" \
      org.opencontainers.image.version="phala-v0.5.21-8" \
      org.phala.serving-patches.revision="${PATCHES_COMMIT}" \
      org.phala.base-engine.revision="c2ce5382dad45fd85e39f5d4e1d93165225eb34d"

# prepare_context.py exports only immutable Git objects and their manifest.
COPY . /opt/phala-glm53-protocol/
RUN python /opt/phala-glm53-protocol/docker/phala-glm53-protocol/install.py \
      --engine-commit "${ENGINE_COMMIT}" --patches-commit "${PATCHES_COMMIT}" \
    && python -m pytest /opt/phala-glm53-protocol/test/registered/unit/entrypoints/openai/test_serving_chat.py -k glm53 -q \
    && hf --help >/dev/null \
    && hf download --help >/dev/null \
    && sglang serve --help >/dev/null \
    && curl --version >/dev/null

ENTRYPOINT ["sglang", "serve"]
