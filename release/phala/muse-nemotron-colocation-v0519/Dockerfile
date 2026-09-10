ARG BASE_IMAGE="docker.io/lmsysorg/sglang:v0.5.19-cu130@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"
FROM ${BASE_IMAGE} AS source_builder

ARG SOURCE_DATE_EPOCH
ARG RELEASE_VERSION
ARG SOURCE_REVISION
ARG RUST_TOOLCHAIN="1.92.0"

ENV PATH="/root/.cargo/bin:${PATH}" \
    CARGO_INCREMENTAL=0 \
    PYTHONHASHSEED=0 \
    SETUPTOOLS_SCM_PRETEND_VERSION="0.5.19+phala.r2" \
    SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH}" \
    SGLANG_BUILD_RUST_EXTS=all

WORKDIR /build/sglang

# The source tree is the only input for the Python runtime and its Rust
# extensions.  Build-only Python packages are hash-pinned and remain in this
# discarded stage.  Cargo dependencies are locked by both rust/Cargo.lock and
# rust/mem-cache/Cargo.lock; rustup verifies the pinned toolchain artifacts.
COPY --link release/phala/muse-nemotron-colocation-v0519/build-requirements.txt /tmp/build-requirements.txt
RUN python3 -m pip install --no-cache-dir --no-deps --require-hashes \
      -r /tmp/build-requirements.txt \
    && rustup toolchain install "${RUST_TOOLCHAIN}" --profile minimal --no-self-update

COPY --link python python
COPY --link rust rust
COPY --link proto proto
COPY --link README.md LICENSE ./

RUN set -eux; \
    cp README.md LICENSE python/; \
    export RUSTUP_TOOLCHAIN="${RUST_TOOLCHAIN}-x86_64-unknown-linux-gnu"; \
    export RUSTFLAGS="-C link-arg=-Wl,--build-id=none --remap-path-prefix=/build/sglang=."; \
    python3 -m pip install --no-deps --no-build-isolation -e python; \
    suffix="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"; \
    test -f "python/sglang/srt/rust_extensions/_server${suffix}"; \
    test -f "python/sglang/srt/rust_extensions/_grpc${suffix}"; \
    test -f "python/sglang/srt/rust_extensions/_multimodal${suffix}"; \
    test -f "python/sglang/srt/mem_cache/rust_tree_core/mem_cache${suffix}"; \
    test -f python/sglang/_version.py; \
    rm -f python/README.md python/LICENSE; \
    find python/sglang -type d -name '__pycache__' -prune -exec rm -rf '{}' +; \
    find python/sglang -type f -name '*.pyc' -delete; \
    find python/sglang -type f -exec touch -d "@${SOURCE_DATE_EPOCH}" '{}' +

FROM ${BASE_IMAGE}

ARG CREATED_AT=""
ARG RELEASE_VERSION=""
ARG SOURCE_REPOSITORY=""
ARG SOURCE_REVISION=""
ARG SOURCE_TREE_DIGEST=""
ARG SOURCE_DATE_EPOCH=""
ARG PATCH_COMMIT=""
ARG RUNTIME_TREE_DIGEST=""
ARG RUNTIME_CONTENT_DIGEST=""
ARG RUNTIME_FILE_COUNT=""
ARG RUNTIME_SYMLINK_DIGEST=""
ARG RUNTIME_SYMLINK_COUNT=""
ARG BASE_IMAGE_REF=""
ARG BASE_IMAGE_DIGEST=""
ARG BASE_PLATFORM_DIGEST=""
ARG BASE_SOURCE_COMMIT=""
ARG DOCKERFILE_DIGEST=""
ARG BUILD_MANIFEST_URL=""
ARG BUILD_MANIFEST_DIGEST=""
ARG BUILD_CONTEXT_URI=""
ARG BUILD_RECIPE_DIGEST=""
ARG BUILDKIT_VERSION=""
ARG SBOM_STATUS=""
ARG PROVENANCE_STATUS=""
ARG REPRODUCIBILITY_STATUS=""
ARG BUILD_REQUIREMENTS_DIGEST=""
ARG RUST_TOOLCHAIN="1.92.0"

LABEL org.opencontainers.image.created="${CREATED_AT}" \
      org.opencontainers.image.title="Phala Muse-Glimmer-30B and Nemotron 3.5 Lightning SGLang v0.5.19 flat runtime" \
      org.opencontainers.image.description="SGLang v0.5.19 runtime for Muse-Glimmer-30B and Nemotron 3.5 Lightning, built from one public source commit on an immutable official SGLang base" \
      org.opencontainers.image.url="https://github.com/Phala-Network/sglang/pkgs/container/sglang" \
      org.opencontainers.image.source="${SOURCE_REPOSITORY}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Phala Network" \
      org.opencontainers.image.ref.name="${RELEASE_VERSION}" \
      org.opencontainers.image.documentation="${SOURCE_REPOSITORY}/tree/${SOURCE_REVISION}/release/phala/muse-nemotron-colocation-v0519" \
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
      org.phala.dockerfile.path="release/phala/muse-nemotron-colocation-v0519/Dockerfile" \
      org.phala.dockerfile.digest="${DOCKERFILE_DIGEST}" \
      org.phala.dependency.lock.mode="immutable-base-plus-hash-pinned-build-tools-and-cargo-locks" \
      org.phala.dependency.lock.digest="${BUILD_REQUIREMENTS_DIGEST}" \
      org.phala.rust.toolchain="${RUST_TOOLCHAIN}" \
      org.phala.builder.platform="linux/amd64" \
      org.phala.buildkit.version="${BUILDKIT_VERSION}" \
      org.phala.source.date.epoch="${SOURCE_DATE_EPOCH}" \
      org.phala.build.manifest.url="${BUILD_MANIFEST_URL}" \
      org.phala.build.manifest.digest="${BUILD_MANIFEST_DIGEST}" \
      org.phala.build.context="${BUILD_CONTEXT_URI}" \
      org.phala.build.recipe.digest="${BUILD_RECIPE_DIGEST}" \
      org.phala.sbom.digest="${SBOM_STATUS}" \
      org.phala.provenance.digest="${PROVENANCE_STATUS}" \
      org.phala.reproducible="${REPRODUCIBILITY_STATUS}" \
      ai.sglang.build.commit="${SOURCE_REVISION}" \
      ai.sglang.build.url="${SOURCE_REPOSITORY}/commit/${SOURCE_REVISION}" \
      ai.sglang.image.tag="${RELEASE_VERSION}"

ENV SGLANG_BUILD_COMMIT="${SOURCE_REVISION}" \
    SGLANG_BUILD_URL="${SOURCE_REPOSITORY}/commit/${SOURCE_REVISION}" \
    SGLANG_IMAGE_TAG="${RELEASE_VERSION}" \
    SGLANG_RUST_BUILD_MODE=never \
    PYTHONPATH="/sgl-workspace/sglang/python"

WORKDIR /sgl-workspace/sglang

# Replace the complete Python runtime tree with the tree built above from this
# public source commit, including all four native Rust extension modules.
# CUDA dependencies remain those of the immutable official v0.5.19 base.
RUN rm -rf /sgl-workspace/sglang/python/sglang
COPY --link --from=source_builder /build/sglang/python/sglang /sgl-workspace/sglang/python/sglang

COPY --link release/phala/muse-nemotron-colocation-v0519/muse-glimmer-chat-template.jinja /opt/muse-glimmer-chat-template.jinja

RUN set -eux; \
    cd /sgl-workspace/sglang/python/sglang; \
    actual_file_count="$(find . -type f | wc -l | tr -d ' ')"; \
    actual_symlink_count="$(find . -type l | wc -l | tr -d ' ')"; \
    actual_content_digest="$(find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print "sha256:" $1}')"; \
    actual_symlink_digest="$(find . -type l -printf '%P -> %l\n' | LC_ALL=C sort | sha256sum | awk '{print "sha256:" $1}')"; \
    test -z "${RUNTIME_FILE_COUNT}" || test "${actual_file_count}" = "${RUNTIME_FILE_COUNT}"; \
    test -z "${RUNTIME_SYMLINK_COUNT}" || test "${actual_symlink_count}" = "${RUNTIME_SYMLINK_COUNT}"; \
    test -z "${RUNTIME_CONTENT_DIGEST}" || test "${actual_content_digest}" = "${RUNTIME_CONTENT_DIGEST}"; \
    test -z "${RUNTIME_SYMLINK_DIGEST}" || test "${actual_symlink_digest}" = "${RUNTIME_SYMLINK_DIGEST}"

RUN set -eu; \
    mkdir -p /usr/share/phala; \
    printf '%s\n' \
      '{' \
      '  "schema_version": 1,' \
      "  \"source\": {\"repository\": \"${SOURCE_REPOSITORY}\", \"commit\": \"${SOURCE_REVISION}\", \"tree\": \"${SOURCE_TREE_DIGEST}\", \"runtime_tree\": \"${RUNTIME_TREE_DIGEST}\", \"runtime_content_digest\": \"${RUNTIME_CONTENT_DIGEST}\"}," \
      "  \"base\": {\"image\": \"${BASE_IMAGE_REF}\", \"index_digest\": \"${BASE_IMAGE_DIGEST}\", \"platform_digest\": \"${BASE_PLATFORM_DIGEST}\", \"source_commit\": \"${BASE_SOURCE_COMMIT}\"}," \
      "  \"build\": {\"context\": \"${BUILD_CONTEXT_URI}\", \"dockerfile\": \"release/phala/muse-nemotron-colocation-v0519/Dockerfile\", \"dockerfile_digest\": \"${DOCKERFILE_DIGEST}\", \"recipe_digest\": \"${BUILD_RECIPE_DIGEST}\", \"platform\": \"linux/amd64\", \"source_date_epoch\": \"${SOURCE_DATE_EPOCH}\", \"buildkit_version\": \"${BUILDKIT_VERSION}\"}," \
      "  \"build_manifest\": {\"url\": \"${BUILD_MANIFEST_URL}\", \"input_digest\": \"${BUILD_MANIFEST_DIGEST}\"}" \
      '}' \
      > /usr/share/phala/sglang-build-input-manifest.json; \
    printf '%s\n' "${SOURCE_REVISION}" > /etc/.GIT_REV
