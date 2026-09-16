FROM ghcr.io/phala-network/sglang:muse-glimmer-30b-nemotron-3-5-lightning-sglang-v0.5.19-phala-r3-01433fd331-20260912@sha256:e772d234b28fe70d3ceaec603b85ef95f07a4f5e8f5be7b1a4ec3e8772113b59
ARG SOURCE_DATE_EPOCH
ARG SOURCE_REVISION
ARG RELEASE_VERSION
RUN echo '437b392484f80f208f275eb71abbc6be6f52939fbe99c0408421e9828589df84  /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py' | sha256sum -c -
COPY python/sglang/srt/entrypoints/openai/serving_chat.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py
RUN echo 'c841df268e20ee598f85186887faf6adcefd470d54e68a2976b7103ac3f99650  /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py' | sha256sum -c -
RUN echo 'cc83aeb512c78b227224bf3d201eca0570b321fd6802ee6b9622e8a2a38b5007  /opt/muse-glimmer-chat-template.jinja' | sha256sum -c -
COPY release/phala/muse-nemotron-colocation-v0519/muse-glimmer-chat-template.jinja /opt/muse-glimmer-chat-template.jinja
RUN echo '900db3effc316e33295ec3d7dfa2df83ea2735228cba73adba8fecc2e83343f7  /opt/muse-glimmer-chat-template.jinja' | sha256sum -c -
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0"
