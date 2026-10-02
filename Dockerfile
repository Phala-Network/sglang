FROM ghcr.io/phala-network/sglang:logfix-deepseek-v0520-r5@sha256:d2dfdc79d78e925170c05bad961269d49ced55652a7bc45f209503df7c87d502
COPY framework_log_privacy.py /opt/phala-source/python/sglang/srt/utils/framework_log_privacy.py
COPY patch_framework.py /tmp/deepseek-framework-log-patch.py
RUN python3 /tmp/deepseek-framework-log-patch.py > /opt/phala/deepseek-framework-integration.json && rm /tmp/deepseek-framework-log-patch.py
COPY shared-inputs.json /opt/phala/deepseek-framework-shared-inputs.json
LABEL io.phala.logfix.framework="deepseek-r5-framework-r4"
