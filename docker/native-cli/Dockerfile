FROM ghcr.io/phala-network/sglang:glm53-nvfp4-cc-ipc-lifecycle-r3-883836dc@sha256:da624990a68052e2721bf43ebbe15d24be7bb6d27ecbbcccd0ba31a6439d88ef

ARG SOURCE_REVISION
ARG SOURCE_DATE_EPOCH
ARG VERSION
ARG REPAIR_SCRIPT_SHA256
ARG TEST_SCRIPT_SHA256
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision=$SOURCE_REVISION \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.title="GLM-5.3 native HF CLI packaging repair" \
      org.opencontainers.image.description="Restore the existing HF console entrypoint without changing packages or r3 serving repairs" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Phala Network" \
      org.phala.source.commit=$SOURCE_REVISION \
      org.phala.base.image="ghcr.io/phala-network/sglang:glm53-nvfp4-cc-ipc-lifecycle-r3-883836dc" \
      org.phala.base.digest="sha256:da624990a68052e2721bf43ebbe15d24be7bb6d27ecbbcccd0ba31a6439d88ef" \
      org.phala.source.date.epoch=$SOURCE_DATE_EPOCH

COPY restore_hf_cli.py test_native_cli.py /opt/phala/native-cli/
RUN printf '%s  %s\n' "$REPAIR_SCRIPT_SHA256" /opt/phala/native-cli/restore_hf_cli.py \
      "$TEST_SCRIPT_SHA256" /opt/phala/native-cli/test_native_cli.py | sha256sum -c - \
    && PYTHONDONTWRITEBYTECODE=1 python3 /opt/phala/native-cli/restore_hf_cli.py \
    && PYTHONDONTWRITEBYTECODE=1 python3 /opt/phala/native-cli/test_native_cli.py
