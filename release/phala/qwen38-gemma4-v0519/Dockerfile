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
COPY python/sglang/srt/model_loader/gguf_name_maps.py /sgl-workspace/sglang/python/sglang/srt/model_loader/gguf_name_maps.py
COPY python/sglang/srt/model_loader/loader.py /sgl-workspace/sglang/python/sglang/srt/model_loader/loader.py
COPY python/sglang/srt/models/qwen3_5.py /sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py
COPY python/sglang/srt/models/qwen3_5_text.py /sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_text.py
COPY test/registered/unit/test_qwen35_gguf_weight_transform.py /opt/phala/tests/test_qwen35_gguf_weight_transform.py
COPY release/phala/qwen38-gemma4-v0519/runtime-files.sha256 /opt/phala/runtime-files.sha256
RUN cd /sgl-workspace/sglang && sha256sum -c /opt/phala/runtime-files.sha256 \
    && python3 -m compileall -q python/sglang/srt/model_loader/gguf_name_maps.py \
       python/sglang/srt/model_loader/loader.py python/sglang/srt/layers/linear.py \
       python/sglang/srt/models/qwen3_5.py python/sglang/srt/models/qwen3_5_text.py \
    && hf --help >/dev/null && hf download --help >/dev/null && curl --version >/dev/null

ENTRYPOINT []
