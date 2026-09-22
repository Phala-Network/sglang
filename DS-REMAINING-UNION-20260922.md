# DeepSeek remaining union migration (2026-09-22)

Independent worktree: `tmp/model-union-ds-remaining-20260922`.
Immutable starting union commit: `8a51d28ba2ad953593ed033612d19205e1388cb9`.
No edits to the integrator worktree, patch exports, `serving_chat.py`, or
`xgrammar_backend.py`. No push, PR, image build, GPU traffic, or deployment.

## Source migration

Each donor was applied as its own functional commit, preserving attribution.
DS0015 required three conflict resolutions: retained the union's descriptive
prefill CLI text and its token-budget validation, adding only the DS encoder
SWA request restrictions; reconciled the graph config comment. DS0018 replaced
the existing equivalent inline completed-chunk backup with the donor's tested
helper, rather than performing two backups.

| Manifest entry | Donor source | Local adaptation |
| --- | --- | --- |
| 0012 Engram/history | `22d54075f9` | `fdfd602b93` |
| 0013 compression ratios | `5648185cee` | `cdac0e7332` |
| 0014 UE8M0/MXFP8 | `e53312611f` | `2b8fb6bf41` |
| 0015 native integration | `9af234a489` | `cfd8926f40` |
| 0016 dense prefill | `5645663683` | `2529c66e48` |
| 0017 dense consumer tests | `3c0eb7c73d` | `88852a5a6d` |
| 0018 completed backup | `f94b624c3e` | `efd2a24245` |
| 0021 medium finalize | `ecd9f93317` | `948188cc37` |
| 0022 transplant imports | `7a23a97622` | `66b5a12fa5` |
| 0023 MXFP4 padding | `81b67d8f17` | `e6f51cb222` |
| 0025 routed ABI | `6a899fd829` | `2620c870a1` |
| 0026 bitmap extent | `b6a5ec4af0` | `8c7a719a60` |

0019/0020 are semantic adaptations, not wholesale cherry-picks: the historical
`phala_compat` monkey-patch package is deliberately absent. Native additions
are `AsyncDynamicbatchTokenizer.run_sync` (existing serialized worker,
ContextVars preserved), the base serving async extension point, client media
exception classification, DS encoder effort tiers and role normalization.
Existing union reasoning-token cap, usage details and output-exclusion logic
are not duplicated. 0024's current `parse_dsv41_reasoning_effort` integer and
fractional boundaries are exercised by the rewritten native CPU test, without
restoring the historical wrapper-dependent selftest.

## Required shared-serving adaptation by the integrator

The following is **not implemented in this worktree** because the shared
serving owner must integrate it with the other models:

1. Async chat opt-in: take the `_convert_to_internal_request_async` method from
   `git show 437b982abfdfa1b563122e061ead1a1cd1211c29 -- python/sglang/srt/entrypoints/openai/serving_chat.py`.
   Only `chat_encoding_spec == "dsv41"` with no caller-supplied `input_ids`
   uses the existing batcher's `run_sync`; absent batcher/other encoders use
   the synchronous path. The base hook and worker method now exist.
2. Media modality/URI/inline-size validation: donor semantic implementation is
   `5ad27e1db0d14d17d920e9040599f20655fbe9df:python/sglang/srt/phala_compat/dsv41_media_hardening.py`.
   Union `_validate_media_content` currently rejects all media for text-only
   models, but returns immediately for multimodal models; individual
   image/audio/video capability, scheme and inline-size checks remain missing.
   Integrate direct helpers, not monkey-patches or broad fail-open exception
   wrappers. The original donor's size estimator is not a hardened URL parser.
3. Integer 1–100 top-level DS reasoning budgets: donor
   `437b982abfdfa1b563122e061ead1a1cd1211c29:python/sglang/srt/phala_compat/dsv41_protocol_compat.py`
   routes integers into `chat_template_kwargs`. Union's public
   `ReasoningEffortType` remains tiers or float [0, .99], so integer 75 is not
   currently accepted there. The DS encoder and parser accept integer budgets
   via template kwargs. Do not widen every model's numeric effort semantics or
   transplant the historical global validator. Preserve current Qwen/Muse
   token-budget semantics, flat/nested precedence and exclusion behavior.
4. Recheck both top-level and message-level tools for explicit `tool_choice=none`
   through the existing union filtering path. The donor wrapper source is
   `5ad27e1db0d14d17d920e9040599f20655fbe9df:python/sglang/srt/phala_compat/dsv41_tool_choice_none.py`.
   Do not overwrite allowed-tools/named-tool/schema work or force historical
   streaming usage options globally.

These outstanding shared integrations mean this handoff is not full protocol
acceptance for DS0019/0020. No XGrammar changes are needed by these donors.

## Consumer closure re-evaluated after DS0015

The earlier "consumer absent" observation applied to the 0011-era union, not
this resulting source. The following are real connected implementations,
not evidence based merely on filenames:

- **Vision:** processor `DeepseekV41ImageProcessor.process_mm_data_async`
  expands image token spans and emits dimensions/optional GPU plan;
  `DeepseekV4ForCausalLM.__init__` constructs ViT/Aligner/image parameters when
  `model_type == deepseek_v41 && vision_n_layers > 0`, rejecting CP/PP/MoE A2A.
  `forward` calls `_prepare_mm_embeddings` → shared `embed_mm_inputs` →
  `get_image_feature` → reconstruct/materialize → ViT → Aligner → start,
  feature, newline and end spans. Prompt image hashes are remapped for routing.
  `remap_weight_name_to_dpsk_hf_format` maps vision QKV/output names and retains
  aligner/image names; `load_weights` only skips them when vision is absent.
  `DeepseekV2MoE` dispatches vision routing to `vision_topk`, selecting the
  checkpoint's image-specific correction bias.
- **Engram:** `DeepseekV4Model.__init__` constructs layout/hasher;
  decoder layers construct their selected Engram modules.
  `_forward_with_pre_mix` computes hash IDs (CP, breakable graph or normal
  path), injects each selected layer's embedding/gate and preserves image
  residuals. Scheduler `get_next_batch_to_run` calls manager history
  preparation, ForwardBatch carries it, and the hasher commits decode/extend
  history. DSpark worker invokes `update_after_verify` using accepted lengths.
  `.engram.wkv.scale` maps to its FP8 loader; `EngramEmbedding._load_rows` owns
  shard loading and post-load calls finalize host mappings.
- **Python C1/C2 HiCache:** source layers are collected by ratio, low-ratio KV
  and index pools are allocated, and consumers map to their preceding source.
  `_dsv4_low_ratio_entries` creates KV/index/scale host pool entries grouped in
  full-page units (packed index formats omit the separate scale entry).
  `build_deepseek_v4_hicache_stack` adds them to the host group/controller;
  `_DeepSeekV4Strategy.build` produces KV-derived ALL_PAGES sidecars;
  `_apply_stack_result` registers them with `UnifiedRadixCache`.
  `_build_sidecar_transfers` participates in backup/load/storage/prefetch,
  inheriting FULL keys/index source. Ratio-2's partial request ring is rebuilt
  at even prefix page boundaries rather than independently cached.

Source paths are therefore connected. Remaining qualification includes actual
model weight coverage, GPU/numerical execution, host mapping lifecycle, native
multimodal/radix extension builds, and real HiCache/Mooncake ACK/reload.
The CPU vision test executes span assembly with fake ViT/Aligner boundaries;
the HiCache tests execute pool/sidecar construction with fake host allocations.
Neither is a full model or device-transfer test.

## Guards and verification boundary

- MXFP8 view requires explicit FlashInfer CUTLASS/CuTe-DSL, CUDA Blackwell,
  FlashInfer availability, K-block 32 and UE8M0 scales; plain-weight readers
  stay unshuffled. A new finite-scale check rejects +inf, which the donor
  otherwise encoded as UE8M0 byte 255 (NaN).
- Hopper FP8 admits BF16, aligned N/K and 1 <= M < 32; empty/large batches
  fall back. Shared MNNVL allreduce rejects float4-misaligned widths and wrong
  group identity even with the same world size.
- Medium finalize has an independent communicator namespace, explicit opt-in,
  target-verify-only token range 97–384, no draft activation, and retains
  model mHC/Blackwell/TP4 admission. Deferred routed weights keep their declared
  BF16 or FP32 values; mismatches are errors, never reinterpretation.
- Bitmap host dispatch uses actual page-table width including scratch, with
  separate 1M/2M token extents, 16-byte vector transactions, and explicit bounds.
  CPU test is source/host-arithmetic simulation only. GPU bitmap tests are
  carried but were not run.
- No PTX compilation was performed. UE8M0 exponent conversion is bit arithmetic,
  while low-ratio FP4 rope uses CUDA `__nv_cvt_float2_to_fp4x2` intrinsics.
  Their generated PTX/toolkit fallback behavior on each architecture remains
  unverified here. Architecture-specific decode/communication kernels still
  need the native/GPU gate; no cross-hardware acceptance claim.

Local commands (Python runtime path is machine-specific):

```
python -B test/phala_deepseek_v41/test_cpu_contracts_v0520.py -v
python -B test/phala_deepseek_v41/test_union_remaining_cpu.py -v
python -B test/phala_deepseek_v41/test_mxfp4_weight_load_contract.py -v
python -B test/phala_deepseek_v41/audit_v0520_source.py
```

The first is stdlib-only; the second uses installed torch 2.13.0+cpu and NumPy,
executing actual production function bodies with only kernel/infrastructure
edges substituted. The third checks actual load-method ordering with stubs.
The source audit checks changed imports/env references from the immutable
union base and both ownership-excluded files.

Executed results: 8 native encoder/async tests, 16 CPU source-method tests,
and 3 MXFP4 load-order tests passed (27 total). Source audit parsed 133 changed
Python files with zero missing imports/symbols/envs and zero ownership-excluded
file changes. Ruff F821 passed on the changed Python set; F401/I passed on the
new/adapted regression files; `git diff --check` passed. A Windows GBK default
read in the ablation fixture was corrected to explicit UTF-8 before final pass.
No test process remains live.

Design ablation: removed historical monkey-patch indirection in favor of
native functions. The finite-scale guard is ablated in memory from the actual
function AST: +inf then produces byte 255; restoring the guard raises. The
bitmap arithmetic control demonstrates why a single 1M specialization cannot
cover a context-plus-speculative-scratch page. No serving performance claim.

Historical accepted text image
`35b7d3019eb9a1ac5c2b3ee8cb00e12a8fabc5c59ee6dc0da769baae27a96f6d`
does not accept this new union.
