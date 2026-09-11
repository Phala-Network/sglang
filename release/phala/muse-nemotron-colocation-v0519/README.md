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

The packaging builds and installs a regular wheel, including all four Rust
extensions and matching distribution metadata. Build-time wheel validation
compares every included source Python file against the frozen source tree.
The exact two upstream `.claude` developer-tool scripts are excluded and
reported explicitly; other missing files, including hidden files, fail the gate.
The r3 XGrammar dependency is rebuilt from the complete, hash-pinned 0.2.6
source distribution with the bounded exact-object-order patch and version
`0.2.6+phala.strictorder1`. See `xgrammar-source.lock.json` and
`STRICT_OBJECT_ORDER_AND_TOOLS.md` for source identity, changes and limits.
SGLang's public-version requirement accepts that local-version build. The
established runtime path is retained as
an in-image symlink to the installed package, not an editable source overlay.
Installation isolates pip from the base checkout's PYTHONPATH and removes its
stale `sglang.egg-info`. The final build requires exactly one SGLang distribution;
retaining an old dist-info or source egg-info cannot pass as an aligned install.
New dependency installs use `--no-compile` to avoid timestamp-bearing bytecode;
clean-build comparison is still required, not inferred from this setting.

XGrammar's build backend emits nondeterministically ordered ZIP members and
RECORD rows even with `PYTHONHASHSEED=0`. The build-only
`canonicalize_wheel.py` verifies every original RECORD hash/size and membership,
then deterministically orders the archive and RECORD. It rejects signed wheels,
links, ambiguous paths, missing or duplicate entries, and tampered payloads.
Every non-RECORD payload and all semantic member metadata must remain identical;
timestamp, permission, native-library and source differences are not normalized
away. The script proves payload preservation and idempotence before replacing
the build output. It is not present in the final runtime or invoked at startup.
Tests: `python3 -m unittest discover -s release/phala/muse-nemotron-colocation-v0519/tests -p test_canonicalize_wheel.py -v`.

Both model templates are stored beside this Dockerfile and baked into the
runtime. Muse uses `/opt/muse-glimmer-chat-template.jinja`; Nemotron uses
`/opt/phala/nemotron-lightning-chat-template.jinja`. No model code,
package installation, Rust compilation, or downloader self-update runs when
the released image starts. Model artifacts are downloaded by the image's
pinned `hf download` command with full Hugging Face revisions supplied by the
Compose manifest.

Publication requires OCI index annotations, an SPDX SBOM attestation, SLSA
provenance in max mode, a Phala build-manifest referrer, registry readback, and
two clean builds whose runnable platform manifest, config, and layer
descriptors are identical after normalizing attestation-only differences.
