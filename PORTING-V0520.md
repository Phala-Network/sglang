# DeepSeek-V4.1 compatibility candidate on SGLang v0.5.20

This is an actual v0.5.20 source branch, rooted at `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`, with selected upstream model commits and reviewed adaptations. It is not the r7 tree relabeled by adding a release ancestor. Source preparation date: 2026-09-19. No remote build, GPU job, serving load, routing or publication was performed by this source-preparation task.

## Frozen inputs

- Official release: v0.5.20, independently rechecked as latest during this work. Annotated tag object `d158602ff1d2cb953196c95158c488d503d2470c`; peeled source above.
- Base image supplied and inspected by the parent task: `lmsysorg/sglang:v0.5.20@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f`; linux/amd64 platform manifest `sha256:b27fce60bc5494c118c4910702812bcfa8cee67abcdd1ff8b0902f21647552f4`.
- Base runtime: Python 3.12.3, torch 2.13.0+cu130, sglang-kernel 0.4.7, sgl-deep-gemm 0.2.0, FlashInfer python/cubin 0.6.18 and jit-cache 0.6.18+cu130, mooncake-transfer-engine-cuda13 0.3.13, huggingface_hub 1.32.0. Do not re-resolve these dependencies while installing the candidate.
- `python/pyproject.toml`, `rust/Cargo.lock`, and `rust/sglang-radix-tree/Cargo.lock` remain the release versions. Rust toolchain in `rust/rust-toolchain.toml` is 1.92.
- Workspace Cargo.lock SHA256: `81b4b5496d3d4eba8ce4f6a1f21fa7d66feec10a0af8587bca9b6ecdb041d5f8`.
- Radix-tree Cargo.lock SHA256: `dfdf5bf35be4c9865a6f42562b64c4ab667a1d7cc7ac35fa0c03956c34c5a2f5`.

The final candidate commit and archive/tree hashes belong in the parent's build-input manifest after this file is committed. They are deliberately not self-referenced here.

Post-freeze review correction: source `2d625cff` omitted the donor's `_pad_intermediate_size(layer)` call in MXFP4 weight postprocessing. A follow-up commit restores it after the MegaMoE bypass and before gate/up reorder or byte shuffling. The former source archive/build is superseded. The actual-method CPU-stub regression reproduced one failure before the correction and all three cases pass afterward; the padding helper, create/load/scale registration and routed quantization functions are AST-identical to donor `a6cf0581`. Real torch CPU fixtures additionally check 576→640 zero/one padding and aligned/no-op behavior; run those in the image because local Windows has no torch.

## Donor closure and adaptations

| Donor commit | Upstream change | Candidate role |
| --- | --- | --- |
| `2f7d04da` | #39677 | Rust V4.1 image preprocessing, C1/C2 KV pool names, PD bootstrap metadata |
| `91f691c4` | #39646 | Standalone V4.1/Engram kernels and Python wrappers |
| `faaff1ec` | #39648 | Candidate/top-k kernels |
| `7d5696b3` | #39653 | NVLink and finalize communication kernels/wrappers |
| `13d593b6` | #39652 | Low-ratio compression, KV I/O and metadata |
| `35b7589e` | #39671 | Candidate indexer API |
| `c2443458` | #39656 | RoPE and FP4 packing |
| `46ae84df` | #39657 | FP8 shape dispatch and associated Hopper configurations |
| `408d2334` | #39664 | mHC computation and compensated projections |
| `c89c63fa` | #39668 | Vision tower/image processing |
| `464fffbe` | #39665 | Native V4.1 chat encoding and tool/reasoning parser selection |
| `3401b752` | #39666 | Engram model module and request history |
| `1f0c73e9` | #39921 | Generalized compression-ratio metadata and pools required by ratios 1/2 |
| `1b200ffa` | #40039 | 32-wide-K UE8M0 dense FP8 weights through MXFP8 GEMMs |
| `a6cf0581` | #38798 | V4.1 config/model/DSpark/HiCache and runtime integration |
| `7061256028c67c0cb8dab77e9cc00eecb2d4b8dc` + `6cec021d06fd47e3a54aef323817ec6e1af13bb1` | OPEN #40217 | Separate dense-prefill memory repair and strengthened consumer tests |
| local `1cfea70c` | r6/r7 repair | Completed write-through chunks use existing backup/ACK path |
| local `52210a6` / `aa54beb` | qualified Phala contract | Reasoning effort/input aliases, role behavior, tool-none/media/usage, async conversion |
| local `7766836c` | r7 narrow #39704 adaptation | Preserve target-only medium finalize namespace/kill switch |

Official donors retain `cherry picked from` provenance in candidate commits. Conflicts were resolved by preserving stable unrelated behavior:

- Kept stable scheduler, graph-pool runtime state and SWA component byte-for-byte. Kept stable load-back quota `kv_tokens + result.delta > mem_quota` and the `rotation_tail_declined` early return. The local chunk-backup call runs only after that early return.
- Did not import unrelated Ling parser registration, NPU-specific CP/WO-A optimizations, AMD breakable-prefill support, general EP/MoE-TP reduction refactoring or an unrelated admission-rejection test. Their context appeared in donor conflict hunks but their implementation dependencies were not part of this model closure.
- Preserved stable NPU plain-FP8 fallback while adapting the MXFP8 dispatch. Retained existing TP all-reduce semantics, with consume-once tracking for the DSV4-specific fusion.
- Added the missing non-padded token-count plumbing to the V2 MoE reused by V4.1; retained explicit imports from their stable owning modules. Dynamic and full-graph execution must still be exercised in the actual image.
- Did not carry r7's optional adaptive-prefill/SWA-retention wrappers. The parent confirmed no corresponding live env overrides. They depend on the old component API and must not replace stable SWA segment-lock behavior.
- Medium finalize keeps two communicator namespaces. The new upstream small-row mHC consumer continues on `default`; medium target verification uses the plain, un-normalized consumer on `dsv41_medium`, leaving mHC post outside that kernel. The upstream weight-dtype-aware JIT signature is retained; the copied GPU precompile fixture was adapted to pass BF16 dtype.

## B OOM repair: scope and remaining proof

The observed B failure is CUDA allocation OOM during `_publish_or_consume_candidates`, not GLM's token-slot allocator failure. The unmodified official integration still builds the full dense FP4 logits matrix and concatenates per-tile position masks. Stable #39182 only provides a generic prefill-buffer ceiling registration hook; it does not bound this DSV4.1 allocation.

#40217 replaces that path with 2 GiB score-row tiles and compact candidate block IDs, consumed on subsequent indexer layers and sliced for tail replay. It preserves the full context contract. It is a separate commit group, not folded into the model-compatibility commit.

**2 GiB is a score-tile budget, not a total HBM bound.** Padded block reductions, masks, queries, keys and allocator behavior add memory. For adversarial non-block-aligned widths, temporary allocation may approach roughly 4.25 GiB plus Q/K/V and other live tensors. Validate the production-shaped 8192-row and long-prefix cases; do not claim the constant alone guarantees no OOM.

The upstream PR is open and its reported results/CI are not this candidate's acceptance. Exact candidate block-ID agreement between old and tiled paths, including score ties at block boundaries, must be checked independently. Comparing a new consumer only against an oracle using the same new IDs is insufficient. Ragged multi-request, zero-length, partial last block, ratio-1/ratio-2, producer→consumer and tail replay remain required tests.

The adjacent GLM fallback patch `43368cd3` is not included. In this stable DSV4 hybrid-SWA path, exhausted continuation budget already returns the parked request, so the exact non-hybrid full-chunk fallback differs; scheduler handling of a parked continuation and physical FULL/SWA page capacity should be added as a focused regression before considering a separate patch. Do not conflate it with this malloc OOM repair.

## Native extensions and build handoff

Required changed native extension interfaces:

1. `sglang.srt.rust_extensions._multimodal`: `_multimodal.dsv41.resize_patchify` from `rust/sglang-mm`.
2. `sglang.srt.mem_cache.rust_tree_core.mem_cache`: updated C1/C2 pool-name serialization from `rust/sglang-radix-tree`.

Build these through **the main `python/setup.py`**, not `pip install rust/sglang-radix-tree` (that crate has no standalone Python pyproject). The setup discovers the independent radix manifest, its `torch_2_13_compat.h`, and passes Cargo `--locked`. Explicitly set `RUSTUP_TOOLCHAIN=1.92.0` for metadata and build invocations: Cargo launched from the `python/` cwd does not necessarily discover the sibling `rust/rust-toolchain.toml`. Record `rustc --version` and `cargo --version` under that override.

Parent packaging design uses a fresh measured `/opt/phala-source`, explicitly copies unchanged `_server`/`_grpc` extensions from the immutable base, builds `SGLANG_BUILD_RUST_EXTS=multimodal,mem_cache`, installs editable with `--no-deps --no-build-isolation`, and binds import resolution to that source tree. Verify all four extension files after installation; a partial wheel reinstall must not silently uninstall the two inherited binaries. Alternatively build all extensions when publishing a full replacement wheel.

Build-only pins chosen by parent: setuptools-rust 1.11.1, setuptools-scm 8.3.1, semantic-version 2.10.0, with wheel SHA256 `--require-hashes`; base setuptools 84.0 and wheel 0.48 are retained. These satisfy source requirements. Fail closed on `import setuptools_rust`: setup.py otherwise permits an installation with no native rebuild. Do not replace these reviewed pins with latest dependencies. Record compiler version, both Cargo lock hashes and installed build-tool versions in build inputs.

Source-archive install outline (commands for the parent builder, not executed here):

```sh
python3 -c 'import setuptools_rust, setuptools_scm, torch; assert torch.__version__.startswith("2.13.0")'
RUSTUP_TOOLCHAIN=1.92.0 SGLANG_BUILD_RUST_EXTS=multimodal,mem_cache python3 -m pip install --no-deps --no-build-isolation -e /opt/phala-source/python
python3 -c 'from sglang.srt.rust_extensions import _multimodal, _grpc, _server; assert callable(_multimodal.dsv41.resize_patchify); import sglang.srt.mem_cache.rust_tree_core.mem_cache'
```

The image must retain model/downloader agreement, direct `hf --help`/`hf download --help`, pinned source path/compiled extensions, installed-file hashes, immutable provenance and rollback. This document does not assert those image gates passed.

## Source checks and image test commands

Local Windows Python 3.13 has no torch/pytest/transformers; no fake claim of the real runtime CPU suite is made. Completed here:

- `python test/phala_deepseek_v41/audit_v0520_source.py`: parses changed Python sources, checks internal import/env references (including namespace packages), and asserts protected stable files are unchanged plus quota/rotation/chunk-backup guards. One pre-existing TYPE_CHECKING-only `FlashinferCombineInput` export is explicitly excluded from import-symbol checking.
- Ruff `F821` over all existing changed Python files: passed after correcting two transplant imports.
- `python test/phala_deepseek_v41/test_cpu_contracts_v0520.py -v`: 8 tests passed, including original-object preservation, reasoning aliases/budgets/exclusion, tool-none carriers, continuous usage and actual Req emitted-prefix reasoning cap.
- `git diff --check`: passed. These are structural/finite CPU checks, not GPU correctness or startup proof.

Run in the source-bound/final-image CPU environment (separate final-image tests must not mount serving code):

```sh
python3 test/phala_deepseek_v41/audit_v0520_source.py
python3 test/phala_deepseek_v41/test_cpu_contracts_v0520.py -v
python3 test/phala_deepseek_v41/test_mxfp4_weight_load_contract.py -v
python3 test/phala_deepseek_v41/selftest.py
python3 -m pytest -q test/registered/unit/entrypoints/openai/test_async_dsv41_conversion.py test/registered/unit/layers/test_dsv41_candidate_blocks.py test/registered/unit/models/test_dsv41_medium_finalize_selection.py
python3 -m pytest -q test/registered/unit/layers/quantization/test_mxfp4_trtllm_padding.py test/registered/unit/model_executor/test_pool_configurator.py test/registered/unit/mem_cache/test_unified_radix_hicache_dispatch.py
python3 -m pytest -q test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py -k 'TestChunkedWriteThroughBackupDecision or scheduler_hicache_load_back_fallback_keeps_old_anchor'
```

The structural Git-based audit needs the original checkout or equivalent source identity metadata; a stripped archive image can run its AST/protected-file checks after providing baseline hashes, or run it before archive creation. Do not mistake a missing `.git` failure for runtime failure. Actual imports of V4.1 config/model/DSpark/backend, extension symbols and server-arg resolution are mandatory before GPU startup. GPU suites: `test/registered/kernels/ops/attention/test_dense_prefill_indexer.py`, `test/registered/kernels/ops/communication/test_dsv41_medium_finalize_all_reduce.py`, and the relevant V4.1 KV/HiCache restore tests. Preserve simulated-versus-real acceptance and test only authorized hardware.

## Design ablation and limitations

The chosen closure is much larger than a cache fix because v0.5.20 lacks the native V4.1 model. Official split donors keep it attributable, while stable generic cache/graph/scheduler repairs remain intact. Removed unrelated conflict-context features and old runtime-adapter knobs instead of importing their dependency stacks. Kept separate commits for model support, #40217, Phala protocol, write-through backup and medium finalize because each has a distinct regression/rollback boundary.

Not yet proven by this task: Rust compilation against the final base, full Python-runtime import/tests, FlashInfer deferred/native payload parity, full-graph and breakable-prefill replay, long-context memory peak, numerical/tie equivalence, protocol service acceptance, HiCache/Mooncake ACK/reload semantics, A watchdog root cause, B OOM elimination, throughput or publication. Parent owns those stages. Freeze and test this candidate before further upstream rebases; do not relabel previous r7 or donor measurements as v0.5.20 results.
