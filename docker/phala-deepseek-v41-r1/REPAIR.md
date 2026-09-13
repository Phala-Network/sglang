# DeepSeek V4.1 source-integrated compatibility

Base: SGLang `da64c5cbb8cf6bfd39be19da43573fdfd484c43a`, preview image
`lmsysorg/sglang:dev-dsv41@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860`.

This packaging converts the existing no-TDX experiment's seven adapters into
ordinary modules under `sglang.srt.phala_compat`, with explicit calls in the
affected source modules. It removes the experiment's `sitecustomize`, global
import hooks, runtime source mounts and `PYTHONPATH` injection. The opt-in
adaptive prefill and SWA controls retain their existing environment gates.
All replaced upstream files and final source files are SHA-256 guarded.
Dependencies are inherited without installation from the immutable base.

Includes the narrowly scoped usage-accounting repair from upstream PR
https://github.com/sgl-project/sglang/pull/37450 (head
`e27d2d464151baed376e2c4d2ef60939737c13ac`, open/unmerged at inspection).
DSpark can accept a final run beyond the output budget. Emitted token IDs are
already clipped; the previous reasoning counter included the clipped suffix.
The repair caps that counter at the emitted finishing prefix, without changing
generation, cache ownership, speculative acceptance or the decode hot loop.
Both failing-budget and within-budget regressions accompany the repair.

Source, CPU tests, builder-local image, immutable registry publication and
GPU/protocol acceptance remain separate release stages. See task evidence for
actual results; this file does not declare any unexecuted stage successful.
