FROM ghcr.io/phala-network/sglang:v0.5.19-nemotron-3-5-lightning-single-r8@sha256:be8469903ccfb7cf3f512d83c99a5b38981c82db4b75ab7facf944c748d65db2
ARG SOURCE_DATE_EPOCH
ARG SOURCE_REVISION
ARG RELEASE_VERSION
RUN echo '80b68264779a93cf91011d8bc365f7b287ddfce572c09382b074cb5111e23751  /sgl-workspace/sglang/python/sglang/srt/managers/schedule_policy.py' | sha256sum -c - \
 && echo '9f413ce51409c48138b915274b538f10bcb7b20499f87ab8ab1c2d950d667c07  /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py' | sha256sum -c -
COPY python/sglang/srt/managers/schedule_policy.py /sgl-workspace/sglang/python/sglang/srt/managers/schedule_policy.py
COPY python/sglang/srt/managers/scheduler.py /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py
RUN echo 'c0d82a8d3554a8fecbd069703ae6f767f5c75ba292094f8153ac7d29be8ee368  /sgl-workspace/sglang/python/sglang/srt/managers/schedule_policy.py' | sha256sum -c - \
 && echo '3d40880a8545c453fb70388331bed71589ab7eba6edb8a772aee0ca7658bdc2b  /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py' | sha256sum -c -
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0"
