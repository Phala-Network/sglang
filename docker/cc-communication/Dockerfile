FROM ghcr.io/phala-network/sglang:glm53-nvfp4-main-ccf9fe-c1ff6d9b-r1@sha256:174f2d0c1beddf47f41a3db401b6651134443b1a4c11332458cc19ad9c73b83d
ARG SOURCE_REVISION
ARG SOURCE_DATE_EPOCH
ARG VERSION=glm53-nvfp4-cc-ipc-r2-candidate
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision=$SOURCE_REVISION \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.title="GLM-5.3 CC IPC fusion candidate" \
      org.opencontainers.image.description="Pinned SGLang CC dispatch and guarded FlashInfer 0.6.18 IPC allocations; hardware acceptance pending" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Phala Network" \
      org.phala.source.commit=$SOURCE_REVISION \
      org.phala.base.digest="sha256:174f2d0c1beddf47f41a3db401b6651134443b1a4c11332458cc19ad9c73b83d" \
      org.phala.source.date.epoch=$SOURCE_DATE_EPOCH
COPY . /opt/phala/cc-communication/
RUN python3 /opt/phala/cc-communication/install_overlay.py
ENV SGLANG_SOURCE_ROOT=/sgl-workspace/sglang/python/sglang \
    FLASHINFER_SOURCE_ROOT=/opt/sglang/lib/python3.12/site-packages/flashinfer
RUN python3 /opt/phala/cc-communication/test_flashinfer_cc_contract.py
