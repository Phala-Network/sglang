FROM ghcr.io/phala-network/sglang:muse-glimmer-30b-nemotron-3-5-lightning-sglang-v0.5.19-phala-r3-01433fd331-20260912@sha256:e772d234b28fe70d3ceaec603b85ef95f07a4f5e8f5be7b1a4ec3e8772113b59
ARG SOURCE_DATE_EPOCH
ARG SOURCE_REVISION
ARG RELEASE_VERSION
RUN echo 'b9150bc84144afe1e82b0db1b52c910794b2438af02e2c5b8376bc626fde6a5a  /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py' | sha256sum -c -
COPY python/sglang/srt/entrypoints/openai/serving_chat.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py
RUN echo 'c841df268e20ee598f85186887faf6adcefd470d54e68a2976b7103ac3f99650  /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py' | sha256sum -c -
RUN echo '0bef34aa26e59fd316164486e6001ec121a49a1f0f353ed7fd683c0ef1c6fb1a  /opt/muse-glimmer-chat-template.jinja' | sha256sum -c -
COPY release/phala/muse-nemotron-colocation-v0519/muse-glimmer-chat-template.jinja /opt/muse-glimmer-chat-template.jinja
RUN echo '900db3effc316e33295ec3d7dfa2df83ea2735228cba73adba8fecc2e83343f7  /opt/muse-glimmer-chat-template.jinja' | sha256sum -c -
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0"
