FROM ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol3-ref5-dflashmask1@sha256:eca5f09ba72beddb55fa79e6e975827d31c7444400b6e41c96be241f1370326d
ARG ENGINE_REVISION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_REVISION}" \
      org.opencontainers.image.version="v0.5.21-phala-3-k3protocol3-ref5-modelcache1"
COPY base-files.json install.py /tmp/kimi-model-cache/
COPY runtime/ /tmp/kimi-model-cache/runtime/
RUN /opt/sglang/bin/python /tmp/kimi-model-cache/install.py && rm -rf /tmp/kimi-model-cache
