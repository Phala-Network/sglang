ARG BASE_IMAGE=lmsysorg/sglang:v0.5.19-cu130@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9
FROM ${BASE_IMAGE}

ARG SOURCE_REVISION
ARG SOURCE_DATE_EPOCH
ARG RELEASE_VERSION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.title="SGLang Qwen3.8 GGUF and Gemma4 qualification" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.phala.base.image="lmsysorg/sglang:v0.5.19-cu130" \
      org.phala.base.digest="sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"

COPY release/phala/qwen38-gemma4-v0519/base-files.sha256 /tmp/base-files.sha256
RUN cd /sgl-workspace/sglang && sha256sum -c /tmp/base-files.sha256
COPY python/sglang/srt/layers/linear.py /sgl-workspace/sglang/python/sglang/srt/layers/linear.py
COPY python/sglang/srt/layers/quantization/gguf.py /sgl-workspace/sglang/python/sglang/srt/layers/quantization/gguf.py
COPY python/sglang/srt/model_loader/gguf_name_maps.py /sgl-workspace/sglang/python/sglang/srt/model_loader/gguf_name_maps.py
COPY python/sglang/srt/model_loader/gguf_vision.py /sgl-workspace/sglang/python/sglang/srt/model_loader/gguf_vision.py
COPY python/sglang/srt/model_loader/loader.py /sgl-workspace/sglang/python/sglang/srt/model_loader/loader.py
COPY python/sglang/srt/models/qwen3_5.py /sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py
COPY python/sglang/srt/models/qwen3_5_text.py /sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_text.py
COPY python/sglang/srt/function_call/qwen3_coder_detector.py /sgl-workspace/sglang/python/sglang/srt/function_call/qwen3_coder_detector.py
COPY python/sglang/srt/function_call/schema_argument_coercion.py /sgl-workspace/sglang/python/sglang/srt/function_call/schema_argument_coercion.py
COPY python/sglang/srt/constrained/llguidance_backend.py /sgl-workspace/sglang/python/sglang/srt/constrained/llguidance_backend.py
COPY python/sglang/srt/function_call/utils.py /sgl-workspace/sglang/python/sglang/srt/function_call/utils.py
COPY python/sglang/srt/entrypoints/openai/serving_chat.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py
COPY python/sglang/srt/entrypoints/openai/mode_sampling_defaults.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/mode_sampling_defaults.py
COPY test/registered/unit/test_qwen35_gguf_weight_transform.py /opt/phala/tests/test_qwen35_gguf_weight_transform.py
COPY test/registered/unit/test_qwen_coder_schema_contract.py /opt/phala/tests/test_qwen_coder_schema_contract.py
COPY test/registered/unit/test_guidance_reasoner_callbacks.py /opt/phala/tests/test_guidance_reasoner_callbacks.py
COPY test/registered/unit/test_named_tool_reference_scope.py /opt/phala/tests/test_named_tool_reference_scope.py
COPY test/registered/unit/test_mode_sampling_defaults.py /opt/phala/tests/test_mode_sampling_defaults.py
COPY test/registered/unit/test_qwen35_gguf_vision.py /opt/phala/tests/test_qwen35_gguf_vision.py
COPY test/registered/unit/test_gguf_prefill_dispatch.py /opt/phala/tests/test_gguf_prefill_dispatch.py
COPY release/phala/qwen38-gemma4-v0519/runtime-files.sha256 /opt/phala/runtime-files.sha256
COPY release/phala/qwen38-gemma4-v0519/qwen-vision-config.json /root/.cache/huggingface/hub/models--HauhauCS--Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/snapshots/993a5971fda8f30dd1b7eb2654792ba4415c7460/config.json
RUN cd /sgl-workspace/sglang && sha256sum -c /opt/phala/runtime-files.sha256 \
    && python3 -m compileall -q python/sglang/srt/model_loader/gguf_name_maps.py \
       python/sglang/srt/model_loader/loader.py python/sglang/srt/layers/linear.py \
       python/sglang/srt/models/qwen3_5.py python/sglang/srt/models/qwen3_5_text.py \
    && hf --help >/dev/null && hf download --help >/dev/null && curl --version >/dev/null

ENTRYPOINT []
