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
timestamp as `SOURCE_DATE_EPOCH`, `--provenance=mode=max`, and a digest-pinned
SBOM generator. For this release line, use:

```text
--attest=type=sbom,generator=docker.io/docker/buildkit-syft-scanner:stable-1@sha256:ae4f3b554449e7e25548e7d8ccc029d17357348e30c6e3df01b92bc93654d6a9
```

Publish both a version tag and a source-revision tag, attach the sanitized
build manifest as an OCI referrer, and verify the runtime manifest with two
clean BuildKit builders before promotion.

The Dockerfile intentionally has no external `# syntax=` frontend tag. The
release pins the BuildKit image itself and uses its built-in Dockerfile
frontend, so a floating frontend image is not an undeclared build input.

The immutable base for this release line is:

```text
docker.io/lmsysorg/sglang:v0.5.19-cu130@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9
```

For `linux/amd64`, that index resolves to:

```text
sha256:37bbbd3444732a464bbc68dee4fb0164e0ce9e18e2f027f3fc967f1152d3c262
```
