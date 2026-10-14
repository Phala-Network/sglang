# Xing4.0 on stable SGLang

This image delta extends the maintained official v0.5.21 runtime with Xing4.0
MLA, MoE, mHC and protocol support from upstream PR39793 at
`aa737beda5878755dffe40836de014663ca2a106`. The source is based on the published
`phala-v0.5.21-7` compatibility line. Native dependencies are unchanged.

Build on the authorized CPU builder, from a clean committed source checkout:

```bash
python docker/xing4/prepare_context.py ../xing4-context
docker build -f ../xing4-context/docker/xing4/Dockerfile \
  --build-arg ENGINE_COMMIT="$(git rev-parse HEAD)" \
  --build-arg PATCHES_COMMIT=<ordered-patch-commit> \
  --build-arg RELEASE_VERSION=<immutable-version> \
  -t ghcr.io/phala-network/sglang:<immutable-version> ../xing4-context
```

All executable source is installed during the build. GPU qualification uses the
image identity without source mounts or startup installation. Direct `hf --help`,
`hf download --help` and `sglang serve --help` are required final-image checks.
The focused CPU checks are in `test/registered/unit/models/test_xing4_native.py`;
CUDA attention, TileLang, real checkpoint loading, FP8 storage and model quality
still require scoped B300 qualification. No embedded Governor is selected.

The official SentencePiece `Xing4_0Tokenizer` is installed and registered in
the image. Model mounts need only fixed weights, JSON, `tokenizer.model` and
Jinja data; loading uses `trust_remote_code=False` and never imports Python
from the model mount. Tokenizer reconstruction and real vocabulary probes are
in `test_xing4_tokenizer.py`; the latter take `XING4_TOKENIZER_ASSETS` pointing
to fixed BF16/FP8 data assets on the CPU builder.

For protected backend authentication, set `PIG_AUTH_FROM_TOKEN=1` and supply
`TOKEN` through the protected process environment. Do not put keys in CLI
arguments. The maintained resolution pipeline uses that environment for API
and admin authentication and redacts diagnostics. Backend health/metrics
probes are exempt, so public ingress must enforce authentication on every path.
