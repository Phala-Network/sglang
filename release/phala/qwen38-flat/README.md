# Qwen3.8-27B flat SGLang image

This release definition has one immutable official SGLang base and one public
source commit. It must not use a Phala `r12`, `r13`, `r14`, or other overlay
image as its base.

BuildKit uses the public repository URL plus the complete commit SHA as its Git
context, so provenance records the VCS source directly. The Dockerfile removes
the base image's entire
`/sgl-workspace/sglang/python/sglang` directory and copies the complete
`python/sglang` tree from that archive. Native CUDA components and all dependencies
except XGrammar remain those of the digest-pinned official base. XGrammar is
built from the full upstream commit and the hash-guarded public patch recorded
in `native-dependency.json`. No source fork image, runtime installer, source
mount or mutable dependency tag is used.

The r5 patch backports the empty XML parameter-zone fix from upstream PR #837
to XGrammar 0.2.1. It also enforces required names omitted from `properties`,
retaining typed additional-property constraints and rejecting contradictory
closed schemas before generation. Undeclared required names combined with
`patternProperties`/`propertyNames` are explicitly unsupported; clients must
declare those properties. The Qwen XML parser retains the original types of
typed additional parameters in both streaming and non-streaming output.

The native build uses the official base's compiler, CMake, scikit-build-core
and TVM FFI. `pip wheel --no-deps --no-build-isolation` runs without network
access. The final installation is also offline and never occurs at startup.
The effective upstream `RelWithDebInfo` mode is preserved explicitly. A fixed
GCC random seed makes bundled LTO static archives deterministic, and the
hash-guarded wheel canonicalizer sorts ZIP members and regenerates RECORD
without changing any other member's bytes. Two independent native wheel
builds and two final runtime image builds must compare equal before release.
Qualify the installed native grammar and SGLang parser together with
`test/registered/unit/function_call/test_qwen_xml_empty_required_schema.py`;
serialized structural-tag equality alone does not establish grammar behavior.

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
