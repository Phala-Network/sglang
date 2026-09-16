FROM ghcr.io/phala-network/sglang:muse-glimmer-30b-nemotron-3-5-lightning-sglang-v0.5.19-phala-r3-01433fd331-20260912@sha256:e772d234b28fe70d3ceaec603b85ef95f07a4f5e8f5be7b1a4ec3e8772113b59
ARG SOURCE_DATE_EPOCH
ARG SOURCE_REVISION
ARG RELEASE_VERSION
RUN echo '437b392484f80f208f275eb71abbc6be6f52939fbe99c0408421e9828589df84  /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py' | sha256sum -c -
COPY python/sglang/srt/entrypoints/openai/serving_chat.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py
RUN echo '2d6140b0c06016ced68324386affce8a24c2a57d64cabfd8906fd84214993304  /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py' | sha256sum -c -
RUN echo 'cc83aeb512c78b227224bf3d201eca0570b321fd6802ee6b9622e8a2a38b5007  /opt/muse-glimmer-chat-template.jinja' | sha256sum -c -
COPY release/phala/muse-nemotron-colocation-v0519/muse-glimmer-chat-template.jinja /opt/muse-glimmer-chat-template.jinja
RUN echo '900db3effc316e33295ec3d7dfa2df83ea2735228cba73adba8fecc2e83343f7  /opt/muse-glimmer-chat-template.jinja' | sha256sum -c -
RUN echo 'f3ffa0fbdf3cadc722386a743a7fbd7d4b332b0be083a2a554822e20778a7603  /sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py' | sha256sum -c -
COPY python/sglang/srt/managers/tokenizer_manager.py /sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py
RUN echo '8fdab38dc7f46d8a667e5f690c63d9dfc542a75c396df4314db70b847b9a2d2a  /sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py' | sha256sum -c -
RUN echo '341a9bb86dc852b20ca9b8649c849985e986f8f85b9165887e77f1513ae57033  /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py' | sha256sum -c -
COPY python/sglang/srt/managers/scheduler.py /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py
RUN echo '9f413ce51409c48138b915274b538f10bcb7b20499f87ab8ab1c2d950d667c07  /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py' | sha256sum -c -
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0"
