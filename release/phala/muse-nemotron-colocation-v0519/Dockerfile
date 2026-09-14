ARG BASE_IMAGE="docker.io/lmsysorg/sglang:v0.5.19-cu130@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"
FROM ${BASE_IMAGE} AS source_builder

ARG SOURCE_DATE_EPOCH
ARG RELEASE_VERSION
ARG SOURCE_REVISION
ARG RUST_TOOLCHAIN="1.92.0"
ARG PYTHON_PACKAGE_VERSION="0.5.19+phala.mn.r5"

ENV PATH="/root/.cargo/bin:${PATH}" \
    CARGO_INCREMENTAL=0 \
    PYTHONHASHSEED=0 \
    SETUPTOOLS_SCM_PRETEND_VERSION="${PYTHON_PACKAGE_VERSION}" \
    PYTHONDONTWRITEBYTECODE=1 \
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
COPY --link release/phala/muse-nemotron-colocation-v0519/verify_wheel.py /opt/phala/verify_wheel.py

RUN set -eux; \
    cp README.md LICENSE python/; \
    export RUSTUP_TOOLCHAIN="${RUST_TOOLCHAIN}-x86_64-unknown-linux-gnu"; \
    export RUSTFLAGS="-C link-arg=-Wl,--build-id=none --remap-path-prefix=/build/sglang=."; \
    python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir /wheelhouse ./python; \
    python3 /opt/phala/verify_wheel.py /wheelhouse /build/sglang/python/sglang "${PYTHON_PACKAGE_VERSION}"

FROM ${BASE_IMAGE} AS xgrammar_builder

ARG SOURCE_DATE_EPOCH
ENV SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH}" \
    PYTHONHASHSEED=0 \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /build/xgrammar
ADD --checksum=sha256:9d54c310b88f9f362655de1586529add59cf625db0af7c5c07f228287777ce84 https://files.pythonhosted.org/packages/82/21/03c58293ed7d539252affb6499121dcd63265774e030f049ef4147b46f5b/xgrammar-0.2.6.tar.gz /tmp/xgrammar-0.2.6.tar.gz
COPY --link release/phala/muse-nemotron-colocation-v0519/xgrammar-strict-order.patch /tmp/xgrammar-strict-order.patch
COPY --link release/phala/muse-nemotron-colocation-v0519/xgrammar-patched-files.sha256 /tmp/xgrammar-patched-files.sha256
COPY --link release/phala/muse-nemotron-colocation-v0519/xgrammar-bounded-string-escapes.patch /tmp/xgrammar-bounded-string-escapes.patch
COPY --link release/phala/muse-nemotron-colocation-v0519/xgrammar-order-goldens.patch /tmp/xgrammar-order-goldens.patch
COPY --link release/phala/muse-nemotron-colocation-v0519/xgrammar-r4-patched-files.sha256 /tmp/xgrammar-r4-patched-files.sha256
COPY --link release/phala/muse-nemotron-colocation-v0519/canonicalize_wheel.py /opt/phala/canonicalize_wheel.py
RUN set -eu; \
    echo 'f1755d033e8b56cc9e280714cf9fe38d7866e096539d044b05d861f4c37a2fdf  /tmp/xgrammar-strict-order.patch' | sha256sum -c -; \
    tar -xzf /tmp/xgrammar-0.2.6.tar.gz --strip-components=1; \
    patch -p1 --fuzz=0 < /tmp/xgrammar-strict-order.patch; \
    sha256sum -c /tmp/xgrammar-patched-files.sha256; \
    echo '0b86f69f83464c2c6ea461dd4febb7ab0421ac3175ec11a43468513068cabf4e  /tmp/xgrammar-bounded-string-escapes.patch' | sha256sum -c -; \
    patch -p1 --fuzz=0 < /tmp/xgrammar-bounded-string-escapes.patch; \
    echo '7bb323ea5d6772ed90b649301e54f5e93a35762b4c408031c6f3782eb3909e7a  /tmp/xgrammar-order-goldens.patch' | sha256sum -c -; \
    patch -p1 --fuzz=0 < /tmp/xgrammar-order-goldens.patch; \
    sha256sum -c /tmp/xgrammar-r4-patched-files.sha256; \
    CXXFLAGS='-ffile-prefix-map=/build/xgrammar=. -fdebug-prefix-map=/build/xgrammar=.' \
    LDFLAGS='-Wl,--build-id=none' \
    python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir /wheelhouse .; \
    python3 /opt/phala/canonicalize_wheel.py /wheelhouse/xgrammar-*.whl

FROM ${BASE_IMAGE}

ARG SOURCE_DATE_EPOCH
ARG PYTHON_PACKAGE_VERSION="0.5.19+phala.mn.r5"
COPY --link --from=xgrammar_builder /wheelhouse /tmp/phala-xgrammar-wheelhouse
RUN python3 -m pip install --no-cache-dir --no-deps --no-compile --force-reinstall /tmp/phala-xgrammar-wheelhouse/*.whl \
    && PYTHONDONTWRITEBYTECODE=1 python3 -c 'from importlib.metadata import version; assert version("xgrammar") == "0.2.6+phala.strictorder2"' \
    && rm -rf /tmp/phala-xgrammar-wheelhouse

ARG CREATED_AT=""
ARG RELEASE_VERSION=""
ARG SOURCE_REPOSITORY=""
ARG SOURCE_REVISION=""
ARG SOURCE_TREE_DIGEST=""
ARG SOURCE_DATE_EPOCH=""
ARG PATCH_COMMIT=""
ARG BASE_IMAGE_REF=""
ARG BASE_IMAGE_DIGEST=""
ARG BASE_PLATFORM_DIGEST=""
ARG BASE_SOURCE_COMMIT=""
ARG DOCKERFILE_DIGEST=""
ARG BUILD_INPUTS_URL=""
ARG BUILD_INPUTS_DIGEST=""
ARG BUILD_CONTEXT_URI=""
ARG BUILD_RECIPE_DIGEST=""
ARG BUILDKIT_VERSION=""
ARG BUILD_REQUIREMENTS_DIGEST=""
ARG RUST_TOOLCHAIN="1.92.0"

LABEL org.opencontainers.image.created="${CREATED_AT}" \
      org.opencontainers.image.title="Phala Muse-Glimmer-30B and Nemotron 3.5 Lightning SGLang v0.5.19 flat runtime" \
      org.opencontainers.image.description="SGLang v0.5.19 runtime for Muse-Glimmer-30B and Nemotron 3.5 Lightning, built from one frozen source commit on an immutable official SGLang base" \
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
      org.phala.base.image="${BASE_IMAGE_REF}" \
      org.phala.base.digest="${BASE_IMAGE_DIGEST}" \
      org.phala.base.platform.digest="${BASE_PLATFORM_DIGEST}" \
      org.phala.base.source.commit="${BASE_SOURCE_COMMIT}" \
      org.phala.dockerfile.path="release/phala/muse-nemotron-colocation-v0519/Dockerfile" \
      org.phala.dockerfile.digest="${DOCKERFILE_DIGEST}" \
      org.phala.dependency.lock.mode="immutable-base-plus-hash-pinned-runtime-build-tools-and-cargo-locks" \
      org.phala.dependency.lock.digest="${BUILD_REQUIREMENTS_DIGEST}" \
      org.phala.rust.toolchain="${RUST_TOOLCHAIN}" \
      org.phala.builder.platform="linux/amd64" \
      org.phala.buildkit.version="${BUILDKIT_VERSION}" \
      org.phala.source.date.epoch="${SOURCE_DATE_EPOCH}" \
      org.phala.build.inputs.url="${BUILD_INPUTS_URL}" \
      org.phala.build.inputs.digest="${BUILD_INPUTS_DIGEST}" \
      org.phala.build.context="${BUILD_CONTEXT_URI}" \
      org.phala.build.recipe.digest="${BUILD_RECIPE_DIGEST}" \
      ai.sglang.build.commit="${SOURCE_REVISION}" \
      ai.sglang.build.url="${SOURCE_REPOSITORY}/commit/${SOURCE_REVISION}" \
      ai.sglang.image.tag="${RELEASE_VERSION}"

ENV SGLANG_BUILD_COMMIT="${SOURCE_REVISION}" \
    SGLANG_BUILD_URL="${SOURCE_REPOSITORY}/commit/${SOURCE_REVISION}" \
    SGLANG_IMAGE_TAG="${RELEASE_VERSION}" \
    SGLANG_RUST_BUILD_MODE=never \
    PYTHONPATH="/sgl-workspace/sglang/python"

WORKDIR /sgl-workspace/sglang

# Install the complete source-built wheel, including all four native extensions
# and its matching distribution metadata. Preserve the established import path
# as an in-image symlink; no source is loaded from a host mount or at startup.
COPY --link --from=source_builder /wheelhouse /tmp/phala-sglang-wheelhouse
RUN set -eu; \
    PYTHONPATH= python3 -m pip install --no-deps --no-cache-dir --no-compile --force-reinstall /tmp/phala-sglang-wheelhouse/sglang-*.whl; \
    rm -rf /sgl-workspace/sglang/python/sglang /sgl-workspace/sglang/python/sglang.egg-info; \
    package_dir="$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/sglang")')"; \
    test -d "${package_dir}"; \
    ln -s "${package_dir}" /sgl-workspace/sglang/python/sglang; \
    EXPECTED_VERSION="${PYTHON_PACKAGE_VERSION}" PYTHONDONTWRITEBYTECODE=1 python3 -c 'import importlib.metadata as m; import os; import sglang; versions = [d.version for d in m.distributions() if d.metadata.get("Name") == "sglang"]; print("Installed SGLang distributions:", versions); assert versions == [os.environ["EXPECTED_VERSION"]]; assert m.version("sglang") == os.environ["EXPECTED_VERSION"] == sglang.__version__; assert "xgrammar==0.2.6" in m.requires("sglang")'; \
    rm -rf /tmp/phala-sglang-wheelhouse

COPY --link release/phala/muse-nemotron-colocation-v0519/muse-glimmer-chat-template.jinja /opt/muse-glimmer-chat-template.jinja
COPY --link release/phala/muse-nemotron-colocation-v0519/nemotron-lightning-chat-template.jinja /opt/phala/nemotron-lightning-chat-template.jinja
COPY --link release/phala/muse-nemotron-colocation-v0519/xgrammar-source.lock.json /usr/share/phala/xgrammar-source.lock.json
COPY --link --from=source_builder /wheelhouse/wheel-verification.json /usr/share/phala/sglang-wheel-verification.json

RUN set -eu; \
    mkdir -p /usr/share/phala; \
    printf '%s\n' \
      '{' \
      '  "schema_version": 1,' \
      "  \"source\": {\"repository\": \"${SOURCE_REPOSITORY}\", \"commit\": \"${SOURCE_REVISION}\", \"tree\": \"${SOURCE_TREE_DIGEST}\", \"package_version\": \"${PYTHON_PACKAGE_VERSION}\"}," \
      "  \"base\": {\"image\": \"${BASE_IMAGE_REF}\", \"index_digest\": \"${BASE_IMAGE_DIGEST}\", \"platform_digest\": \"${BASE_PLATFORM_DIGEST}\", \"source_commit\": \"${BASE_SOURCE_COMMIT}\"}," \
      "  \"build\": {\"context\": \"${BUILD_CONTEXT_URI}\", \"dockerfile\": \"release/phala/muse-nemotron-colocation-v0519/Dockerfile\", \"dockerfile_digest\": \"${DOCKERFILE_DIGEST}\", \"recipe_digest\": \"${BUILD_RECIPE_DIGEST}\", \"platform\": \"linux/amd64\", \"source_date_epoch\": \"${SOURCE_DATE_EPOCH}\", \"buildkit_version\": \"${BUILDKIT_VERSION}\"}," \
      "  \"build_inputs\": {\"url\": \"${BUILD_INPUTS_URL}\", \"digest\": \"${BUILD_INPUTS_DIGEST}\"}" \
      '}' \
      > /usr/share/phala/sglang-build-input-manifest.json; \
    printf '%s\n' "${SOURCE_REVISION}" > /etc/.GIT_REV
