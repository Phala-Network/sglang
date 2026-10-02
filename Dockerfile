FROM ghcr.io/phala-network/sglang:logfix-nemotron-v0519-r3-astra-20260923-final@sha256:075db1ac10f6b0a3a673b6a98e72cbd0386c4cca1438603e32c1cbe5241642f7
LABEL org.phala.framework.logprivacy.sha256="b62eafff3bb0f6af564da7bcf2e3e381f2622a779a8b6e863a25841666918640" \
      org.phala.base.image="ghcr.io/phala-network/sglang:logfix-nemotron-v0519-r3-astra-20260923-final@sha256:075db1ac10f6b0a3a673b6a98e72cbd0386c4cca1438603e32c1cbe5241642f7"
COPY framework_log_privacy.py /sgl-workspace/sglang/python/sglang/srt/utils/framework_log_privacy.py
COPY patch_framework.py /opt/phala-framework-log-privacy/patch_framework.py
RUN python3 /opt/phala-framework-log-privacy/patch_framework.py > /opt/phala-framework-log-privacy/receipt.json
