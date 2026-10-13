FROM ghcr.io/phala-network/sglang:v0.5.21-phala-3@sha256:d5abbec01e3e6d6f571cf8250871d72373a178dca1d05b398965c759798139cf
ARG ENGINE_REVISION
ARG MOONCAKE_REVISION
ARG PATCHES_REVISION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision=$ENGINE_REVISION \
      io.phala.sglang.revision=$ENGINE_REVISION \
      io.phala.mooncake.revision=$MOONCAKE_REVISION \
      io.phala.mooncake.source="https://github.com/Phala-Network/Mooncake" \
      io.phala.serving-patches.revision=$PATCHES_REVISION
COPY runtime/ /opt/sglang/lib/python3.12/site-packages/mooncake/
COPY verify-mooncake-recovery.py /opt/phala/verify-mooncake-recovery.py
COPY overlay/ /opt/phala/kimi-recovery-3/overlay/
COPY source-lock.json install-overlay.py /opt/phala/kimi-recovery-3/
COPY test_mamba_load_back_admission_cpu.py /sgl-workspace/sglang/test/registered/unit/mem_cache/
RUN /opt/sglang/bin/python /opt/phala/kimi-recovery-3/install-overlay.py && \
    /opt/sglang/bin/python /opt/phala/verify-mooncake-recovery.py --allow-missing-driver
