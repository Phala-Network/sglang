# Qwen3.8-27B flat SGLang image

This release definition has one immutable official SGLang base and one public
source commit. It must not use a Phala `r12`, `r13`, `r14`, or other overlay
image as its base.

BuildKit uses the public repository URL plus the complete commit SHA as its Git
context, so provenance records the VCS source directly. The Dockerfile removes
the base image's entire
`/sgl-workspace/sglang/python/sglang` directory and copies the complete
`python/sglang` tree from that archive. It does not install Python or operating
system dependencies; native CUDA components and dependency versions are
inherited unchanged from the digest-pinned official base.

The release process must use `linux/amd64`, `--pull=false`, the source commit
timestamp as `SOURCE_DATE_EPOCH`, `--sbom=true`, and
`--provenance=mode=max`. Publish both a version tag and a source-revision tag,
attach the sanitized build manifest as an OCI referrer, and verify the runtime
manifest with two clean BuildKit builders before promotion.

The Dockerfile intentionally has no external `# syntax=` frontend tag. The
release pins the BuildKit image itself and uses its built-in Dockerfile
frontend, so a floating frontend image is not an undeclared build input.

The immutable base for this release line is:

```text
docker.io/lmsysorg/sglang:dev-qwen38-27b-dflash2@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe
```

For `linux/amd64`, that index resolves to:

```text
sha256:b91d664a8e4825afc16ab831c6035a6c88ac20ef8bd26da4fe2b9813a9f44376
```
