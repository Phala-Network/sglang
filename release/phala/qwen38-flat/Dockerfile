ARG BASE_IMAGE="docker.io/lmsysorg/sglang:v0.5.19-cu130@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"
FROM ${BASE_IMAGE} AS xgrammar-build
ARG SOURCE_DATE_EPOCH
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=1 \
    TZ=UTC \
    LC_ALL=C.UTF-8
COPY release/phala/qwen38-flat/xgrammar-r5.patch /build-inputs/xgrammar-r5.patch
COPY release/phala/qwen38-flat/canonicalize_wheel.py /build-inputs/canonicalize_wheel.py
RUN set -eu; \
    echo 'cda7644a43f71f6d12152df16865055e5a429cc5aa04c82fe934206b981a9af7  /build-inputs/xgrammar-r5.patch' | sha256sum -c -; \
    echo '55549be27c102fefb87bad0d00ad0d7ffe5a073c07a66fe5929c8bdf58600ec1  /build-inputs/canonicalize_wheel.py' | sha256sum -c -; \
    git init /opt/xgrammar-source; \
    cd /opt/xgrammar-source; \
    git remote add origin https://github.com/mlc-ai/xgrammar.git; \
    git fetch --depth=1 origin 5b4e9ce9e72524037ae24ecd831b9b6604d2eb48; \
    git checkout --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = '5b4e9ce9e72524037ae24ecd831b9b6604d2eb48'; \
    git submodule update --init --depth=1 3rdparty/dlpack; \
    test "$(git -C 3rdparty/dlpack rev-parse HEAD)" = 'bbd2f4d32427e548797929af08cfe2a9cbb3cf12'; \
    git apply --check /build-inputs/xgrammar-r5.patch; \
    git apply /build-inputs/xgrammar-r5.patch
# No build-time package resolution: compiler, CMake, scikit-build-core and TVM
# FFI are inherited from the exact official base. Build cannot access a network.
RUN --network=none python -m pip wheel --no-deps --no-build-isolation \
    --config-settings=cmake.build-type=RelWithDebInfo \
    --config-settings=cmake.define.CMAKE_CXX_FLAGS=-frandom-seed=phala-xgrammar-r5 \
    --wheel-dir=/wheels /opt/xgrammar-source \
    && python /build-inputs/canonicalize_wheel.py /wheels/*.whl

FROM scratch AS xgrammar-wheel
COPY --from=xgrammar-build /wheels /

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
ARG SBOM_STATUS
ARG PROVENANCE_STATUS
ARG REPRODUCIBILITY_STATUS
ARG DEPENDENCY_LOCK_DIGEST

LABEL org.opencontainers.image.created="${CREATED_AT}" \
      org.opencontainers.image.title="Phala Qwen3.8-27B SGLang v0.5.19 flat runtime" \
      org.opencontainers.image.description="Qwen3.8-27B SGLang v0.5.19 runtime built directly from one public source commit on an immutable official SGLang base" \
      org.opencontainers.image.url="https://github.com/Phala-Network/sglang/pkgs/container/sglang" \
      org.opencontainers.image.source="${SOURCE_REPOSITORY}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Phala Network" \
      org.opencontainers.image.ref.name="${RELEASE_VERSION}" \
      org.opencontainers.image.documentation="${SOURCE_REPOSITORY}/tree/${SOURCE_REVISION}/release/phala/qwen38-flat" \
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
      org.phala.dockerfile.path="release/phala/qwen38-flat/Dockerfile" \
      org.phala.dockerfile.digest="${DOCKERFILE_DIGEST}" \
      org.phala.dependency.lock.mode="immutable-official-base-with-source-built-pinned-xgrammar" \
      org.phala.dependency.lock.digest="${DEPENDENCY_LOCK_DIGEST}" \
      org.phala.builder.platform="linux/amd64" \
      org.phala.buildkit.version="${BUILDKIT_VERSION}" \
      org.phala.source.date.epoch="${SOURCE_DATE_EPOCH}" \
      org.phala.build.manifest.url="${BUILD_MANIFEST_URL}" \
      org.phala.build.manifest.digest="${BUILD_MANIFEST_DIGEST}" \
      org.phala.build.context="${BUILD_CONTEXT_URI}" \
      org.phala.build.recipe.digest="${BUILD_RECIPE_DIGEST}" \
      org.phala.sbom.digest="${SBOM_STATUS}" \
      org.phala.provenance.digest="${PROVENANCE_STATUS}" \
      org.phala.reproducible="${REPRODUCIBILITY_STATUS}"

ENV SGLANG_BUILD_COMMIT="${SOURCE_REVISION}" \
    SGLANG_BUILD_URL="${SOURCE_REPOSITORY}/commit/${SOURCE_REVISION}" \
    SGLANG_IMAGE_TAG="${RELEASE_VERSION}"

WORKDIR /sgl-workspace/sglang

COPY --from=xgrammar-build /wheels /opt/phala-xgrammar-wheels
COPY release/phala/qwen38-flat/native-dependency.json /usr/share/phala/xgrammar-source.json
RUN --network=none python -m pip install --no-deps --no-compile --force-reinstall \
      /opt/phala-xgrammar-wheels/xgrammar-0.2.1+phala.r5-*.whl \
    && rm -rf /opt/phala-xgrammar-wheels

# Replace the complete Python runtime tree in one build from the public source
# commit. Native CUDA dependencies remain those of the immutable official base.
RUN rm -rf /sgl-workspace/sglang/python/sglang
COPY --link python/sglang /sgl-workspace/sglang/python/sglang

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
      "  \"build\": {\"context\": \"${BUILD_CONTEXT_URI}\", \"dockerfile\": \"release/phala/qwen38-flat/Dockerfile\", \"dockerfile_digest\": \"${DOCKERFILE_DIGEST}\", \"recipe_digest\": \"${BUILD_RECIPE_DIGEST}\", \"platform\": \"linux/amd64\", \"source_date_epoch\": \"${SOURCE_DATE_EPOCH}\", \"buildkit_version\": \"${BUILDKIT_VERSION}\"}," \
      "  \"build_manifest\": {\"url\": \"${BUILD_MANIFEST_URL}\", \"input_digest\": \"${BUILD_MANIFEST_DIGEST}\"}" \
      '}' \
      > /usr/share/phala/sglang-build-input-manifest.json; \
    printf '%s\n' "${SOURCE_REVISION}" > /etc/.GIT_REV
