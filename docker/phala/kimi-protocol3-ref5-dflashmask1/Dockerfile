FROM ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol3-ref5@sha256:31c9f4a2ef390161664f639b0245a60c4963d734bc8b6157097c46bd5ab3cd95

ARG ENGINE_REVISION
ARG BASE_REF5_DIGEST=sha256:31c9f4a2ef390161664f639b0245a60c4963d734bc8b6157097c46bd5ab3cd95
COPY dflash_utils.py /tmp/dflash_utils.py

LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_REVISION}" \
      org.opencontainers.image.version="v0.5.21-phala-3-k3protocol3-ref5-dflashmask1" \
      io.phala.sglang.parent.ref5.digest="${BASE_REF5_DIGEST}" \
      io.phala.sglang.dflash-grammar-mask-fix="sglang-pr-43518" \
      io.phala.sglang.dflash-grammar-mask-source="python/sglang/srt/speculative/dflash_utils.py"

RUN /opt/sglang/bin/python3 -c "import pathlib, shutil, sglang.srt.speculative.dflash_utils as m; p=pathlib.Path(m.__file__); shutil.copy2('/tmp/dflash_utils.py', p); compile(p.read_text(), str(p), 'exec')" \
    && rm /tmp/dflash_utils.py
