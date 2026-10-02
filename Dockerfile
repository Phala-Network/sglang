FROM ghcr.io/phala-network/sglang:logfix-glm53-v0520-r3-astra-20260923-final@sha256:c8b06dbaddf9537e7531957675cec860087eb61740445ab305c1a85caa18a388
LABEL org.phala.framework.logprivacy.sha256="b62eafff3bb0f6af564da7bcf2e3e381f2622a779a8b6e863a25841666918640" \
      org.phala.base.image="ghcr.io/phala-network/sglang:logfix-glm53-v0520-r3-astra-20260923-final@sha256:c8b06dbaddf9537e7531957675cec860087eb61740445ab305c1a85caa18a388"
COPY framework_log_privacy.py /sgl-workspace/sglang/python/sglang/srt/utils/framework_log_privacy.py
COPY patch_framework.py /opt/phala-framework-log-privacy/patch_framework.py
RUN python3 /opt/phala-framework-log-privacy/patch_framework.py > /opt/phala-framework-log-privacy/receipt.json
