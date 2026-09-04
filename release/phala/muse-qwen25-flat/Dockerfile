ARG BASE_IMAGE="docker.io/lmsysorg/sglang:v0.5.18-cu130@sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1"
FROM ${BASE_IMAGE}

ARG CREATED_AT
ARG RELEASE_VERSION
ARG SOURCE_REPOSITORY
ARG SOURCE_REVISION
ARG SOURCE_TREE_DIGEST
ARG SOURCE_DATE_EPOCH
ARG PATCH_COMMIT
ARG RUNTIME_TREE_DIGEST
ARG RUNTIME_CONTENT_DIGEST
ARG RUNTIME_FILE_COUNT
ARG RUNTIME_SYMLINK_DIGEST
ARG RUNTIME_SYMLINK_COUNT
ARG BASE_IMAGE_REF
ARG BASE_IMAGE_DIGEST
ARG BASE_PLATFORM_DIGEST
ARG BASE_SOURCE_COMMIT
ARG DOCKERFILE_DIGEST
ARG BUILD_MANIFEST_URL
ARG BUILD_MANIFEST_DIGEST
ARG BUILD_CONTEXT_URI
ARG BUILD_RECIPE_DIGEST
ARG BUILDKIT_VERSION

LABEL org.opencontainers.image.created="${CREATED_AT}" \
      org.opencontainers.image.title="Phala GPT-OSS, Muse-Glimmer, and Qwen2.5 SGLang flat runtime" \
      org.opencontainers.image.description="SGLang v0.5.18 runtime for GPT-OSS-20B, Muse-Glimmer-30B, and Qwen2.5-7B, built from one public source commit on an immutable official SGLang base" \
      org.opencontainers.image.url="https://github.com/Phala-Network/sglang/pkgs/container/sglang" \
      org.opencontainers.image.source="${SOURCE_REPOSITORY}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Phala Network" \
      org.opencontainers.image.ref.name="${RELEASE_VERSION}" \
      org.opencontainers.image.documentation="${SOURCE_REPOSITORY}/tree/${SOURCE_REVISION}/release/phala/muse-qwen25-flat" \
      org.phala.source.repository="${SOURCE_REPOSITORY}" \
      org.phala.source.commit="${SOURCE_REVISION}" \
      org.phala.source.tree.digest="${SOURCE_TREE_DIGEST}" \
      org.phala.patch.commit="${PATCH_COMMIT}" \
      org.phala.source.runtime.tree.digest="${RUNTIME_TREE_DIGEST}" \
      org.phala.source.runtime.content.digest="${RUNTIME_CONTENT_DIGEST}" \
      org.phala.base.image="${BASE_IMAGE_REF}" \
      org.phala.base.digest="${BASE_IMAGE_DIGEST}" \
      org.phala.base.platform.digest="${BASE_PLATFORM_DIGEST}" \
      org.phala.base.source.commit="${BASE_SOURCE_COMMIT}" \
      org.phala.dockerfile.path="release/phala/muse-qwen25-flat/Dockerfile" \
      org.phala.dockerfile.digest="${DOCKERFILE_DIGEST}" \
      org.phala.dependency.lock.mode="inherited-from-immutable-official-base-no-added-packages" \
      org.phala.dependency.lock.digest="${BASE_IMAGE_DIGEST}" \
      org.phala.builder.platform="linux/amd64" \
      org.phala.buildkit.version="${BUILDKIT_VERSION}" \
      org.phala.source.date.epoch="${SOURCE_DATE_EPOCH}" \
      org.phala.build.manifest.url="${BUILD_MANIFEST_URL}" \
      org.phala.build.manifest.digest="${BUILD_MANIFEST_DIGEST}" \
      org.phala.build.context="${BUILD_CONTEXT_URI}" \
      org.phala.build.recipe.digest="${BUILD_RECIPE_DIGEST}" \
      org.phala.sbom.digest="buildkit-attestation" \
      org.phala.provenance.digest="buildkit-attestation" \
      org.phala.reproducible="candidate-two-clean-build-verification"

ENV SGLANG_BUILD_COMMIT="${SOURCE_REVISION}" \
    SGLANG_BUILD_URL="${SOURCE_REPOSITORY}/commit/${SOURCE_REVISION}" \
    SGLANG_IMAGE_TAG="${RELEASE_VERSION}"

WORKDIR /sgl-workspace/sglang

# The immutable upstream image supplies CUDA and compiled native dependencies.
# Replace the complete Python runtime from one public source commit in this
# single build, rather than stacking per-file overlay images.
RUN rm -rf /sgl-workspace/sglang/python/sglang
COPY --link python/sglang /sgl-workspace/sglang/python/sglang
COPY --link release/muse-glimmer-r5/chat_template.jinja /opt/muse-glimmer-chat-template.jinja

RUN set -eux; \
    cd /sgl-workspace/sglang/python/sglang; \
    actual_file_count="$(find . -type f | wc -l | tr -d ' ')"; \
    actual_symlink_count="$(find . -type l | wc -l | tr -d ' ')"; \
    actual_content_digest="$(find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print "sha256:" $1}')"; \
    actual_symlink_digest="$(find . -type l -printf '%P -> %l\n' | LC_ALL=C sort | sha256sum | awk '{print "sha256:" $1}')"; \
    test "${actual_file_count}" = "${RUNTIME_FILE_COUNT}"; \
    test "${actual_symlink_count}" = "${RUNTIME_SYMLINK_COUNT}"; \
    test "${actual_content_digest}" = "${RUNTIME_CONTENT_DIGEST}"; \
    test "${actual_symlink_digest}" = "${RUNTIME_SYMLINK_DIGEST}"

RUN set -eu; \
    mkdir -p /usr/share/phala; \
    printf '%s\n' \
      '{' \
      '  "schema_version": 1,' \
      "  \"source\": {\"repository\": \"${SOURCE_REPOSITORY}\", \"commit\": \"${SOURCE_REVISION}\", \"tree\": \"${SOURCE_TREE_DIGEST}\", \"runtime_tree\": \"${RUNTIME_TREE_DIGEST}\", \"runtime_content_digest\": \"${RUNTIME_CONTENT_DIGEST}\"}," \
      "  \"base\": {\"image\": \"${BASE_IMAGE_REF}\", \"index_digest\": \"${BASE_IMAGE_DIGEST}\", \"platform_digest\": \"${BASE_PLATFORM_DIGEST}\", \"source_commit\": \"${BASE_SOURCE_COMMIT}\"}," \
      "  \"build\": {\"context\": \"${BUILD_CONTEXT_URI}\", \"dockerfile\": \"release/phala/muse-qwen25-flat/Dockerfile\", \"dockerfile_digest\": \"${DOCKERFILE_DIGEST}\", \"recipe_digest\": \"${BUILD_RECIPE_DIGEST}\", \"platform\": \"linux/amd64\", \"source_date_epoch\": \"${SOURCE_DATE_EPOCH}\", \"buildkit_version\": \"${BUILDKIT_VERSION}\"}," \
      "  \"build_manifest\": {\"url\": \"${BUILD_MANIFEST_URL}\", \"input_digest\": \"${BUILD_MANIFEST_DIGEST}\"}" \
      '}' \
      > /usr/share/phala/sglang-build-input-manifest.json; \
    printf '%s\n' "${SOURCE_REVISION}" > /etc/.GIT_REV
