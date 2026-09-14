# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ghcr.io/phala-network/sglang:dev-dsv41-phala-r1@sha256:067b5487093e47d52b6fe25242374f7cacc859b92c7d07dd37f6be19c81329c1
FROM redis:7.2.8-bookworm@sha256:73d68753a2c3e9be8473dc06290ec2236058f358db8bf1fd9a8492f3e1874dd1 AS redis-test-tools
FROM ${BASE_IMAGE} AS builder
ARG SOURCE_DATE_EPOCH
ENV RUSTUP_TOOLCHAIN=stable \
    CARGO_BUILD_JOBS=24 \
    PYO3_PYTHON=/opt/sglang/bin/python3 \
    CARGO_INCREMENTAL=0
WORKDIR /build/sgl-model-gateway
COPY LICENSE /build/LICENSE
COPY sgl-model-gateway/ ./
COPY docker/phala-deepseek-v41-pdstream/Cargo.lock ./Cargo.lock
COPY docker/phala-deepseek-v41-pdstream/source.sha256 /build/source.sha256
RUN cd /build && sha256sum -c source.sha256 \
    && rustc --version | grep -F 'rustc 1.98.1 ' \
    && cargo --version | grep -F 'cargo 1.98.1 '
RUN cargo fetch --locked
COPY --from=redis-test-tools /usr/local/bin/redis-server /usr/local/bin/redis-server
RUN --network=none redis-server --version
RUN --network=none cargo test --frozen --release -p sgl-model-gateway --test routing_tests --test reliability_tests --no-run
RUN --network=none cargo test --frozen --release -p sgl-model-gateway --test routing_tests -- --test-threads=1
RUN --network=none cargo test --frozen --release -p sgl-model-gateway --test reliability_tests upstream_cancel_test -- --test-threads=1
RUN --network=none cargo build --frozen --release -p sgl-model-gateway-python \
    && cargo build --frozen --release -p sgl-model-gateway --bin sgl-model-gateway

FROM ${BASE_IMAGE}
ARG SOURCE_DATE_EPOCH
ARG PHALA_SOURCE_COMMIT
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${PHALA_SOURCE_COMMIT}" \
      org.opencontainers.image.version="dev-dsv41-phala-r3-pdstream" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="P/D gateway streaming repair; SGLang model runtime inherited unchanged from r1"
RUN printf '%s\n' \
      '1e1f7680641818c7ccb62f50f5d2875172b00af4b8da927666f671bf5b565234  /opt/sglang/lib/python3.12/site-packages/sglang_router/sglang_router_rs.abi3.so' \
      'c47dd2737c1486e18e2b05df83054909879d0ce04573b8b7298e285e53b083a2  /opt/sglang/bin/sgl-model-gateway' \
      | sha256sum -c -
COPY --from=builder /build/sgl-model-gateway/target/release/libsglang_router_rs.so /opt/sglang/lib/python3.12/site-packages/sglang_router/sglang_router_rs.abi3.so
COPY --from=builder /build/sgl-model-gateway/target/release/sgl-model-gateway /opt/sglang/bin/sgl-model-gateway
COPY docker/phala-deepseek-v41-pdstream/source.sha256 /opt/phala-release/pdstream-source.sha256
COPY docker/phala-deepseek-v41-pdstream/Cargo.lock /opt/phala-release/pdstream-Cargo.lock
RUN python3 -c 'import sglang_router.sglang_router_rs' \
    && python3 -m sglang_router.launch_router --help >/dev/null \
    && sgl-model-gateway --version \
    && hf --help >/dev/null && hf download --help >/dev/null
