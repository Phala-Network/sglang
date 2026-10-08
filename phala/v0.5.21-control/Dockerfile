FROM docker.io/lmsysorg/sglang@sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75
COPY . /opt/phala-control/
RUN python3 /opt/phala-control/install.py
ENV SGLANG_BUILD_COMMIT=e00930c5489053f26d86b179cee0d087f846acbb SGLANG_IMAGE_TAG=v0.5.21-phala-control
LABEL org.phala.sglang.base.commit=e00930c5489053f26d86b179cee0d087f846acbb
