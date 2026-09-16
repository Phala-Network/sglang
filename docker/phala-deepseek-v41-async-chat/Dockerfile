# Use the Dockerfile frontend bundled with the recorded BuildKit version.
ARG BASE_IMAGE=ghcr.io/phala-network/sglang:dev-dsv41-phala-r3-pdstream@sha256:8d496a07f28301612fe8938552aefdcaa196f952b07589882d69954e13322357
FROM ${BASE_IMAGE}
ARG SOURCE_DATE_EPOCH
ARG PHALA_SOURCE_COMMIT
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${PHALA_SOURCE_COMMIT}" \
      org.opencontainers.image.version="dev-dsv41-phala-r4-async-chat" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Phala Network" \
      org.opencontainers.image.description="Opt-in asynchronous DeepSeek V4.1 chat conversion; inherits the qualified r3 gateway"

COPY docker/phala-deepseek-v41-async-chat/base.sha256 /opt/phala-release/async-chat-base.sha256
RUN sha256sum -c /opt/phala-release/async-chat-base.sha256
COPY python/sglang/srt/entrypoints/openai/serving_base.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_base.py
COPY python/sglang/srt/entrypoints/openai/serving_chat.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py
COPY python/sglang/srt/managers/async_dynamic_batch_tokenizer.py /sgl-workspace/sglang/python/sglang/srt/managers/async_dynamic_batch_tokenizer.py
COPY docker/phala-deepseek-v41-async-chat/source.sha256 /opt/phala-release/async-chat-source.sha256
COPY docker/phala-deepseek-v41-async-chat/dependencies.json /opt/phala-release/async-chat-dependencies.json
RUN sha256sum -c /opt/phala-release/async-chat-source.sha256
RUN --network=none python3 - <<'PY'
import pathlib

root = pathlib.Path('/sgl-workspace/sglang/python/sglang/srt')
for relative in ['entrypoints/openai/serving_base.py', 'entrypoints/openai/serving_chat.py',
                 'managers/async_dynamic_batch_tokenizer.py']:
    source = root / relative
    compile(source.read_text(), str(source), 'exec')
    cache_dir = source.parent / '__pycache__'
    for cached in sorted(cache_dir.glob(source.stem + '.*.pyc')):
        assert cached.parent == cache_dir and cached.is_file()
        cached.unlink()
PY
RUN --network=none hf --help >/dev/null && hf download --help >/dev/null
