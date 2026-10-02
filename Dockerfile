FROM ghcr.io/phala-network/sglang:logfix-qwen36-v0519-r3-astra-20260923-final@sha256:7c47bb5274d5bb98c23348f28c1fabc510c2064b04156924416e026c0ed116dc
LABEL org.phala.framework.logprivacy.sha256="b62eafff3bb0f6af564da7bcf2e3e381f2622a779a8b6e863a25841666918640" \
      org.phala.base.image="ghcr.io/phala-network/sglang:logfix-qwen36-v0519-r3-astra-20260923-final@sha256:7c47bb5274d5bb98c23348f28c1fabc510c2064b04156924416e026c0ed116dc"
COPY framework_log_privacy.py /sgl-workspace/sglang/python/sglang/srt/utils/framework_log_privacy.py
COPY patch_framework.py /opt/phala-framework-log-privacy/patch_framework.py
RUN python3 /opt/phala-framework-log-privacy/patch_framework.py > /opt/phala-framework-log-privacy/receipt.json
