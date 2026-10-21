FROM ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol3-ref5-modelcache1@sha256:c8abc4835c98fb2015c3867ca731809ca494332adb033b850fe1dc8caecd9011
ARG ENGINE_REVISION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_REVISION}" \
      org.opencontainers.image.version="v0.5.21-phala-3-k3protocol3-ref5-modelcache2"
COPY base-files.json install.py /tmp/kimi-model-cache2/
COPY runtime/ /tmp/kimi-model-cache2/runtime/
RUN /opt/sglang/bin/python /tmp/kimi-model-cache2/install.py && rm -rf /tmp/kimi-model-cache2
