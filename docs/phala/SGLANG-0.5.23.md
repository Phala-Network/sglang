# Official v0.5.23 version maintenance

The complete v0.5.23 engine is prepared on a fixed version-maintenance branch.
This is a full version line with shared repairs and conditionally enabled Qwen
behavior. Shared v0.5.20 main runtime code, older versions and ongoing model work
are preserved. Embedded Governor is not selected; external PIG owns admission.

Official source: `297e54fd105e974cdf401e5b4d20253a22c16f90` (v0.5.23).
The [source-owned recipe](../../docker/phala/v0.5.23/README.md) records the fixed
CUDA13 amd64 base, build and final-image gates. The serving-patches repository
maintains the ordered `versions/v0.5.23` source export using its existing verifier.

Preparation is unpublished. The final receipt will bind the real complete engine
version-line commit, patches main commit, recipe main commit, immutable source
tag/Release and image digest after final selection. This main documentation
commit is not the complete v0.5.23 engine identity. Source/CPU/GPU evidence,
image publication and production acceptance remain separate.
