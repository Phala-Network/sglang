ARG SGLANG_BASE=ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol3-ref5-modelcache1@sha256:c8abc4835c98fb2015c3867ca731809ca494332adb033b850fe1dc8caecd9011
FROM ${SGLANG_BASE}
ARG RECIPE_COMMIT
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="cebb7c6ce132ef1030b3dbb271862658257b7ac2" \
      org.opencontainers.image.version="phala-kimi-modelexpress-20261029-rc1" \
      ai.phala.recipe.revision="${RECIPE_COMMIT}" \
      ai.phala.modelexpress.upstream="df8bc7ca1730fd40a2af0416b589d84fcd1b090c" \
      ai.phala.release.stage="cpu-qualified-candidate"
COPY dist/modelexpress-0.8.0+phala.mx1-py3-none-any.whl /tmp/mx-wheel/
RUN python3 -m pip install --no-index --no-deps /tmp/mx-wheel/*.whl \
    && rm -rf /tmp/mx-wheel
COPY test_candidate.py /opt/phala/modelexpress-tests/test_candidate.py
COPY smoke.py /opt/phala/modelexpress-tests/smoke.py
ENV MX_INSTANT_TENSOR=0 MX_ARTIFACT_TRANSFER=0
