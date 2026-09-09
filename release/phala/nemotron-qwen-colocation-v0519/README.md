# Nemotron 3.5 Lightning and Qwen3.5-9B SGLang v0.5.19 runtime

This release profile rebuilds the complete `python/sglang` runtime tree from a
single public `Phala-Network/sglang` source commit on the immutable official
SGLang v0.5.19 CUDA 13.0 base image.  That includes `_version.py` and the four
native Rust modules used by the HTTP server, gRPC, multimodal processing and
the unified radix-cache tree core.

The source commit is based on upstream SGLang `v0.5.19` commit
`0bcd822377da7b5718e674eaf9c870d349424dd1` and carries exactly these two
upstream changes:

- [#30430](https://github.com/sgl-project/sglang/pull/30430), which fuses the
  Nemotron latent-MoE projection with the shared-expert add;
- [#38117](https://github.com/sgl-project/sglang/pull/38117), which uses
  Gumbel-max sampling on the unseeded main-sampler path.

Runtime Python, CUDA, FlashInfer, SGL kernel and all other CUDA dependencies are
inherited from the digest-pinned official base image.  A discarded builder
stage installs only three hash-pinned build tools and uses the source tree's
Cargo locks with Rust 1.92.0 to compile the source commit's native extensions.
No package installation or compilation occurs when the released image starts.
The build verifies the final flat runtime file count, content digest, symlink
count and symlink digest before producing the image.

Publication requires OCI index annotations, an SPDX SBOM attestation, SLSA
provenance in max mode, a Phala build-manifest referrer, registry readback, and
two clean builds whose runnable platform manifest, config and layer descriptors
are identical after normalizing attestation-only differences.
