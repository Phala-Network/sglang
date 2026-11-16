ARG SGLANG_BASE=lmsysorg/sglang:v0.5.23-cu130@sha256:f7876bef696ccf9cf75a2bf7ccb82970f5fac577f082fb8ca95c9a59d2160eb5
FROM ${SGLANG_BASE}

ARG ENGINE_COMMIT
ARG ENGINE_VERSION
ARG PATCHES_COMMIT
ARG RECIPE_COMMIT
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_COMMIT}" \
      org.opencontainers.image.version="${ENGINE_VERSION}" \
      org.phala.serving-patches.revision="${PATCHES_COMMIT}" \
      org.phala.recipe.revision="${RECIPE_COMMIT}" \
      org.phala.governor.selected="false"

# prepare.py checks the fixed upstream base, changed package paths and deletions.
# Retain the official native dependencies and CLI, installing the complete srt.
COPY python/sglang/srt/ /sgl-workspace/sglang/python/sglang/srt/
