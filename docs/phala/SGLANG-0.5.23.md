# Official v0.5.23 version maintenance

The complete v0.5.23 source candidate is available on the
[`codex/sglang-v0.5.23-maintenance`](https://github.com/Phala-Network/sglang/tree/codex/sglang-v0.5.23-maintenance)
version-maintenance branch at
[`35f00f8cfaaa56fa3f13edb9e81ae0ae1d4504ed`](https://github.com/Phala-Network/sglang/commit/35f00f8cfaaa56fa3f13edb9e81ae0ae1d4504ed).
This is a full version line with shared repairs and conditionally enabled Qwen
behavior. Shared v0.5.20 main runtime code, older versions and ongoing model work
are preserved. Embedded Governor is not selected; external PIG owns admission.

Official source: `297e54fd105e974cdf401e5b4d20253a22c16f90` (v0.5.23).
The [source-owned recipe](../../docker/phala/v0.5.23/README.md) records the fixed
CUDA13 amd64 base, build and final-image gates. The serving-patches repository
maintains the ordered `versions/v0.5.23` source export using its existing verifier.
The exact patches input is
[`bf8d7c92d4046d06129b97d2951c3d0b51a65a3e`](https://github.com/Phala-Network/sglang-serving-patches/commit/bf8d7c92d4046d06129b97d2951c3d0b51a65a3e).
Its fifth ordered step preserves the Qwen XML unquoted null sentinel when the
original argument schema accepts null, retaining quoted and constrained strings.
The exact affected source and tests passed isolated CPU red 7 failed/18 passed,
then green 25 passed. The generic coercion utility and reasoning parser are unchanged.

This is the source/candidate stage: no image has been published and production
has not been updated. Earlier image and performance evidence retain their
predecessor identities; they do not qualify this source successor. The final
receipt will bind the real complete engine
version-line commit, patches main commit, recipe main commit, immutable source
tag/Release and image digest after final selection. This main documentation
commit is not the complete v0.5.23 engine identity. Source/CPU/GPU evidence,
image publication and production acceptance remain separate.
