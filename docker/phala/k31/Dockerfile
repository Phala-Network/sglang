# syntax=docker/dockerfile:1.7
# Preserve the exact K3.1 native stack; install reviewed Python source at build time.
FROM lmsysorg/sglang:dev-kimi-k31-pro@sha256:4a098ad7414702a2271296ac148854448d10710ec3554ac93b57ef194999305a
ARG PHALA_ENGINE_REVISION
ARG PHALA_RELEASE_VERSION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${PHALA_ENGINE_REVISION}" \
      org.opencontainers.image.version="${PHALA_RELEASE_VERSION}"
LABEL io.phala.mooncake.source="https://github.com/Phala-Network/Mooncake" \
      io.phala.mooncake.revision="467533b2e80a101dfabd4789ed5aa8ab251c24a6"
WORKDIR /sgl-workspace/sglang
# COPY preserves the official editable install's compiled Rust extension and
# generated version module, which are absent from the Git source archive.
COPY . /sgl-workspace/sglang/
COPY --from=mooncake-client /cpu_smoke.py /opt/phala/mooncake-k31/cpu_smoke.py
# The named context contains the immutable, CPU-qualified native artifact.
# Its SHA256 and complete source identity are checked before extracting only
# store.so, engine.so and private dependencies into the installed package.
RUN --mount=type=bind,from=mooncake-client,source=/native-client.tar.gz,target=/opt/testdata/native-client.tar.gz,ro \
    python3 docker/phala/k31/install_native.py /opt/testdata/native-client.tar.gz
RUN python3 -m pip install --no-deps --no-cache-dir --require-hashes \
    -r docker/phala/k31/fastsafetensors-requirements.txt
RUN test -n "$PHALA_ENGINE_REVISION" \
    && mkdir -p /opt/testdata \
    && rm -rf /sgl-workspace/sglang/.git \
    && printf '%s\n' "$PHALA_ENGINE_REVISION" > /sgl-workspace/sglang/.k31-source-commit \
    && python3 /opt/k31-image/verify_image.py
