FROM ghcr.io/phala-network/sglang:muse-glimmer-30b-nemotron-3-5-lightning-sglang-v0.5.19-phala-r7-f97b21d20d-20260914@sha256:52e5e119b977719396d692a5f755b370125773aae28808f96821b3848be85e22
ARG SOURCE_DATE_EPOCH
ARG SOURCE_REVISION
ARG RELEASE_VERSION
RUN echo '6b9dd58cfca505b5abf63fbfe6c9e0248c38328fd9f8001e3ebf21d4356fc239  /sgl-workspace/sglang/python/sglang/srt/function_call/qwen3_coder_detector.py' | sha256sum -c -
COPY python/sglang/srt/function_call/qwen3_coder_detector.py /sgl-workspace/sglang/python/sglang/srt/function_call/qwen3_coder_detector.py
RUN echo 'f720e2c7d780ae6cf79c32040d976bd4fd57e02958b1162272a1388911f57078  /sgl-workspace/sglang/python/sglang/srt/function_call/qwen3_coder_detector.py' | sha256sum -c -
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0"
