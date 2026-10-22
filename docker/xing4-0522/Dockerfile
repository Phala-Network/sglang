FROM docker.io/lmsysorg/sglang@sha256:5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e
ARG ENGINE_REVISION
ARG IMAGE_VERSION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_REVISION}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
COPY python/ /sgl-workspace/sglang/python/
COPY test/registered/unit/ /sgl-workspace/sglang/test/registered/unit/
COPY docker/xing4-0522/ /opt/phala/xing4-0522/
COPY candidate-source.json /opt/phala/candidate-source.json
COPY vendor/xgrammar-0.2.7-cp312-cp312-linux_x86_64.whl /tmp/xgrammar-0.2.7-cp312-cp312-linux_x86_64.whl
RUN python3 /opt/phala/xing4-0522/install.py --revision "${ENGINE_REVISION}"
