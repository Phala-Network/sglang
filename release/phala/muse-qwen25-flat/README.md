# GPT-OSS-20B, Muse-Glimmer-30B, and Qwen2.5-7B flat SGLang runtime

This release target rebuilds the complete `python/sglang` runtime from the
public commit identified by `org.opencontainers.image.revision`. The immutable
official `lmsysorg/sglang:v0.5.18-cu130` image supplies CUDA and compiled native
dependencies; no Python or operating-system dependency is installed at
container startup.

The image is intended for all three:

- `openai/gpt-oss-20b` with the GPT-OSS reasoning/tool parsers and xgrammar.
  The runtime also recognizes constrained Harmony tool calls whose assistant
  start token was consumed by the prompt before non-streaming parsing.
- `RedHatAI/Muse-Glimmer-30B-FP8-block` with the Muse reasoning/tool parsers,
  llguidance, multimodal processing, and the Phala Muse reliability repairs.
- `RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic` with the Qwen tool parser,
  llguidance, and standard OpenAI-compatible structured output.

Publishing must use BuildKit/buildx for `linux/amd64` with the immutable base,
`--pull=false`, `--sbom=true`, `--provenance=mode=max`, manifest/index OCI
annotations, a version tag and a source-revision tag. The release evidence must
record the exact source tree, Dockerfile, runtime content, official base index
and platform digests, BuildKit version, `SOURCE_DATE_EPOCH`, build command,
SBOM/provenance referrers, and a clean second-build reproducibility comparison.
