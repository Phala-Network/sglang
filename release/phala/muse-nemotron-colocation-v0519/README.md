# Muse-Glimmer-30B and Nemotron 3.5 Lightning SGLang v0.5.19 runtime

This release profile rebuilds the complete `python/sglang` runtime tree from a
single public `Phala-Network/sglang` source commit on the immutable official
SGLang v0.5.19 CUDA 13.0 base image. It preserves the Nemotron latent-MoE and
Gumbel sampling optimizations from the v0.5.19 r2 lineage and adds the audited
Muse protocol, structured-output, reasoning-usage, tool-call, and pinned
llguidance vocab-mask repairs used by this colocation image. It also carries
the upstream Mamba radix-cache SSM-index correction, complete Mamba admission
accounting, and speculative-decode deferred-metadata cleanup required by the
Nemotron hybrid Mamba plus EAGLE path.

The r2 packaging builds and installs a regular wheel, including all four Rust
extensions and matching distribution metadata. Build-time wheel validation
compares every included source Python file against the frozen source tree.
The exact two upstream `.claude` developer-tool scripts are excluded and
reported explicitly; other missing files, including hidden files, fail the gate.
XGrammar is hash-pinned to 0.2.6; SGLang's declared dependency, imported library,
and package version must agree. The established runtime path is retained as
an in-image symlink to the installed package, not an editable source overlay.
Installation isolates pip from the base checkout's PYTHONPATH and removes its
stale `sglang.egg-info`. The final build requires exactly one SGLang distribution;
retaining an old dist-info or source egg-info cannot pass as an aligned install.
New dependency installs use `--no-compile` to avoid timestamp-bearing bytecode;
clean-build comparison is still required, not inferred from this setting.

The Muse chat template is stored beside this Dockerfile and copied into the
flat runtime image at `/opt/muse-glimmer-chat-template.jinja`. No model code,
package installation, Rust compilation, or downloader self-update runs when
the released image starts. Model artifacts are downloaded by the image's
pinned `hf download` command with full Hugging Face revisions supplied by the
Compose manifest.

Publication requires OCI index annotations, an SPDX SBOM attestation, SLSA
provenance in max mode, a Phala build-manifest referrer, registry readback, and
two clean builds whose runnable platform manifest, config, and layer
descriptors are identical after normalizing attestation-only differences.
