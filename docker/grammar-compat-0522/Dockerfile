FROM ghcr.io/phala-network/sglang@sha256:b07063aca12994e5dd7c1db64416cfe607c439244cb2eada2688525db6d6b1be
ARG ENGINE_REVISION
ARG IMAGE_VERSION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_REVISION}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
COPY python/ /sgl-workspace/sglang/python/
COPY test/ /sgl-workspace/sglang/test/
COPY candidate-source.json /opt/phala/candidate-source.json
COPY verify.py /opt/phala/grammar-compat-0522/verify.py
RUN python3 /opt/phala/grammar-compat-0522/verify.py --revision "${ENGINE_REVISION}"
