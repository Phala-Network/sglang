# syntax=docker/dockerfile:1
FROM lmsysorg/sglang:dev-dsv41@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860
ARG SOURCE_DATE_EPOCH
ARG PHALA_SOURCE_COMMIT
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${PHALA_SOURCE_COMMIT}" \
      org.opencontainers.image.version="dev-dsv41-phala-r1" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="DeepSeek V4.1 source-integrated compatibility and speculative usage accounting repair"
COPY docker/phala-deepseek-v41-r1/base-source.sha256 /opt/phala-release/base-source.sha256
RUN sha256sum -c /opt/phala-release/base-source.sha256
COPY python/sglang/srt/phala_compat/ /sgl-workspace/sglang/python/sglang/srt/phala_compat/
COPY python/sglang/srt/entrypoints/openai/encoding_dsv41.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/encoding_dsv41.py
COPY python/sglang/srt/entrypoints/openai/protocol.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/protocol.py
COPY python/sglang/srt/entrypoints/openai/serving_chat.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py
COPY python/sglang/srt/entrypoints/openai/usage_processor.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/usage_processor.py
COPY python/sglang/srt/managers/schedule_batch.py /sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py
COPY python/sglang/srt/managers/scheduler.py /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py
COPY python/sglang/srt/mem_cache/unified_cache/components/swa_component.py /sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_cache/components/swa_component.py
COPY python/sglang/srt/multimodal/processors/base_processor.py /sgl-workspace/sglang/python/sglang/srt/multimodal/processors/base_processor.py
COPY python/sglang/srt/utils/__init__.py /sgl-workspace/sglang/python/sglang/srt/utils/__init__.py
COPY python/sglang/srt/utils/common.py /sgl-workspace/sglang/python/sglang/srt/utils/common.py
COPY docker/phala-deepseek-v41-r1/overlay-source.sha256 /opt/phala-release/overlay-source.sha256
RUN sha256sum -c /opt/phala-release/overlay-source.sha256 \
    && python -m compileall -q /sgl-workspace/sglang/python/sglang/srt/phala_compat \
    && hf --help >/dev/null && hf download --help >/dev/null
