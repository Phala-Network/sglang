# Gemma and Qwen historical semantic reconciliation

Scope: source audit against unified SGLang
`6b3ca7eddd1f0a1eed2774bdfd461626c5e0e780`, tree
`b6730c9fe47e14319bb92dc61b8463ae7dcfedfd`. Historical records come from
sglang-serving-patches archive `083fe0f9728b05a68481ecace7681c2315031ab2`,
`profiles/v0.5.19/`. This is not image, model, GPU or production acceptance.
No change to the separately frozen e02dfa/3ae4601 PIG candidate is proposed.

## Actual artifact identities, not directory-name inference

The committed Compose snapshot examined was phala-models-compose
`e500363aee98b8146d89e6ae65193ce47bda8dfb`; no live fleet query was made.

| Snapshot/model | Immutable model/config identity | Runtime provenance and applicability |
| --- | --- | --- |
| Qwen2.5 7B, use1-4c | `RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic` at `a4f1d5442ea284bc8e5a3b3e1a4d528f7d6caacb`; parser `qwen` | Image `v0.5.19-qwen38-gemma4-r10`, digest `52f44fbf198f1748db7e559a180ff5c84ef63b29a3c62c1964fecc08bfe1de45`; original source `82da9278e229402abcb50b46ee5a57e801bb651c`, extracted `d0b3e70cbc5ed3cc757d22e79ffa1fff28f58571`. Qwen25Detector, not Qwen3CoderDetector; not GGUF. |
| Qwen3.8 uncensored, use1-3b | `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF` at `993a5971fda8f30dd1b7eb2654792ba4415c7460`; Q8_K_P file and BF16 mmproj; tokenizer `Qwen/Qwen3.8-27B-FP8` at `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | Same r10 image and original/extracted source as above. Explicit GGUF loader and `qwen3_coder` parser make the missing loader/parser behavior below relevant. Q8_K_P filename is not proof of every matrix's GGUF type. |
| Gemma4 uncensored, use1-3b | `cloud19/gemma-4-26B-A4B-it-heretic-FP8-Static` at `0e75c5c353f6df6eae211788ec9563c4f65676da`; both parsers `gemma4` | Same r10 image/source footprint, but FP8 Gemma4 and Gemma4Detector. The archived profile lists inherited Qwen GGUF patches; it contains no independent Gemma-specific source change. This does not mean Gemma itself was qualified on v0.5.20. |
| Qwen3.6 27B, use1-db | `Qwen/Qwen3.6-27B-FP8` at `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`; `qwen3` reasoning and `qwen3_coder` tools | Image `qwen38-27b-sglang-v0.5.19-phala-r5-1a56fbb0dc-20260914`, digest `aafe6c8dd56ea9469d15423c5742655112099d54a01e0a5b7398e2ed2575b559`; original `1a56fbb0dc48ec3fb2b4b629fc4e74a3b57c639a`, extracted `711978779936d1918d038de8515e33f968cb5193`. Image tag and directory names alone do not identify the served model. |

The archive explicitly describes a *complete deployed-image footprint*, not a
minimal model patch selection. Its `d0b3e70` tree is
`5da72f4fcacb3889d4bfa9a2e35a1debac903aa4`; `7119787` tree is
`d209baf0019b73d3126b85401154de8d0fa124fd`. Packaging/external dependency changes
were excluded from the source projection and still need separate qualification.

## Nine-patch shared r10 lineage

Paths below are relative to `python/sglang/srt/` unless identified as tests.
Original -> extracted commit mappings are preserved by the archive.

| Original -> extracted | Current source and semantic evidence | Disposition |
| --- | --- | --- |
| `5404f7600755` -> `81881bd7c1` | `model_loader/gguf_name_maps.py` only registers Muse; `loader.py:GGUFModelLoader` lacks Qwen text-config/transform path. `models/qwen3_5_text.py` still only accepts `lm_head.weight`, dropping quantized head names. Packed `qweight/qweight_type` and scalar GGUF type handling are absent. | **Missing correctness increment**, guard by Qwen architecture plus GGUF load format. Do not apply to FP8 Gemma/Qwen2.5. |
| `d67e821bd5ea` -> `39eedc17eb` | `GGUFModelLoader.load_model` discards the loaded-parameter set and performs no historical shared-parameter identity completeness audit; `get_missing_gguf_parameters` is absent. | **Missing fail-closed weight completeness audit**, dependent on the loader port. |
| `ea31b4a3509f` -> `55c45e7499` | `function_call/utils.py:get_argument_schema/coerce_argument_to_schema` and `qwen3_coder_detector.py:_convert_param_value` replace the old schema-coercion module; union commits `c19e475e`, `514fcaf8`, `7884e393` cover schema-driven types/no network lookup. However `detect_and_parse` falls back to incomplete text and lacks unknown-name filtering; streaming emits a name before `</tool_call>`. | **Partial**: type coercion represented, historical atomic/truncation/unknown-name contract missing; three failures reproduced by the probe below. |
| `79e215cde60b` -> `97882c5842` | `constrained/llguidance_backend.py` has restored wrapper callbacks; shared commit `87ace709c8` also contains guarded pinned-host masks. | Shared source coverage; native llguidance/CUDA acceptance not rerun. |
| `e26b9f4dae32` -> `6ecc100728` | `function_call/utils.py:get_json_schema_constraint` wraps a named tool in an array but omitted root `$defs`/`definitions`. Required-choice `$defs` aggregation did not fix named choice. | **Reproduced and repaired here** by restoring the selected tool's two reference-root keys. Five source-method regressions pass after three red errors. |
| `f80677f937c5` -> `5a9b0690dd` | `entrypoints/openai/mode_sampling_defaults.py:apply_mode_sampling_defaults` plus the actual call in `serving_chat.py`; shared `1db913dd7b` retains opt-in checkpoint modes and explicit-field precedence. | Shared source coverage; no per-model duplicate defaults needed. |
| `2b2584575aa5` -> `788a898375` | `model_loader/gguf_vision.py` is absent; current `GGUFModelLoader.__init__` rejects all extra config, so the historical audited projector load cannot be represented by the current loader. | **Missing Qwen GGUF multimodal increment**; not permission to silently downgrade to text-only. |
| `c4afa9828243` -> `be0e34807d` | `layers/quantization/gguf.py:fused_mul_mat_gguf` has no `_use_bf16_gguf_prefill`; Q8_0 remains on MMQ. Historical guard was CUDA + BF16 + Q8_0 + batch >=128. | **Absent optimization**, separate from correctness. Reassess on the final pinned kernel/actual tensor types before selecting; no speed claim. |
| `82da9278e229` -> `d0b3e70cbc` | Upstream v0.5.20 contains `f478b2bb2d` (#35255), with scheduler/tokenizer dispatched cancellation handling; shared `0bf1ed94` further preserves ownership cleanup. | Upstream/shared coverage, not a Gemma-only patch. |

## Qwen3.6/FP8 26-record lineage

These are extracted source commit IDs; the corresponding original IDs are in
the archive profile and SOURCE_REVIEWS index. No row is inferred from its directory.

| Historical commits | Current equivalent and boundary |
| --- | --- |
| `28238ec400` | `83b69e75` schema rejection, `abdd4d76` allowed-tool validation, `c19e475e` coercion, `514fcaf8` Qwen structural tags. `constrained/xgrammar_schema.py` is byte-identical between the extracted historical head and union. Native grammar tests remain necessary. |
| `354b7c6c3f`, `c30384652e` | `348b3066`: `_fold_qwen35_system_messages` and `_expose_qwen35_reasoning_tool_history` in serving_chat. Executed source-method tests prove folding/no input mutation and preserved assistant tool reasoning. |
| `ea92ab4c5f`, `538e30d613`, `7a6812c460`, `7846a276b0`, `cd196db843`, `d70dbf79e7` | `348b3066` plus `ffcf7799`: explicit medium `(128,4096)`, low `(32,64)`, high/max -> xhigh `(384,8192)`, default8192, request cap and strict-thinking opt-out. Executed bounded source-method tests; actual decode budget enforcement remains a native/runtime gate. |
| `cd31831d54`, `5e3f909d22` | `ffcf7799`, protocol nested disabled/exclude normalization and typed request thinking bounds. `grammar_manager.py:_get_request_thinking_bounds` feeds reasoner grammar. Source inspected; not a new end-to-end protocol pass. |
| `b85619fd04` | Shared request-bound grammar path in grammar_manager avoids unnecessary unbounded thinking compilation while keeping explicit bounds. Full native compiler timing/performance not rerun. |
| `9b1b0671ab` | `f3418d51`, serving_chat `_process_tool_call_stream` drops orphan argument deltas before a name. This **does not** repair Qwen complete-call/truncation behavior: a prematurely emitted valid name is not an orphan. |
| `b2d182e4d6` | `348b3066` completed-tool-result guidance retained for medium/xhigh. Source-method regression passes; this is guidance, not proof every real model answers without a repeated call. |
| `96e883d1e7` | Upstream `f478b2bb2d` (#35255) + shared ownership cleanup, as above. |
| `ae13fff533` | Upstream `07199fa220` (#36267) exists in the fixed official v0.5.20 ancestry. Qwen GDN projection layout is not a missing Phala patch. Hardware/numerical performance qualification not rerun. |
| `42cac46e56`, `6c6cdbc75c` | Shared `fd778061` watchdog recovery independent of diagnostics, `cc703a7a` excessive text budget rejection before media decode, `6a673233` request-local malformed-image handling preserving OS failures. |
| `4c25322c80`, `82c5597d41`, `a1c281180b`, `cdea3b3c96` | Historical fixture/config-bag corrections, not independent runtime features. Current shared source owns v0.5.20 fixtures; copying old version-specific test setup is unnecessary. |
| `bdec50dcf6` | `6efe42ec`, `function_call/utils.py:normalize_json_schema_types` handles null optional properties/required. |
| `774c19539b`, `772a63931f`, `7119787799` | `514fcaf8` includes `test_qwen_xml_empty_required_schema.py` and detector typed-additionalProperties handling/required repetition. Source represented; installed XGrammar tests were not run in this Windows audit. The CPU-wheel title on `772a639` does not imply a runtime source change. |

## Executed evidence and remaining work

- `test/registered/unit/function_call/test_named_tool_reference_scope.py`:
  before the four-line repair, three errors (`$defs`, `definitions`, selected
  root isolation); afterwards five tests pass.
- `test/registered/unit/entrypoints/openai/test_qwen_historical_source_semantics.py`:
  five tests pass, including Gemma4/Gemma4-text/Qwen2 negative guards. These
  execute the actual pure source methods, not a complete imported runtime.
- `python docs/validation/qwen_history_boundary_probe.py` exits **1**:
  truncated stream, truncated nonstream and unknown-function rejection all
  fail; complete empty-call control passes. This failure is retained, not
  converted into an expected-success flag.
- Source inspection establishes the missing GGUF implementation; it is not a
  substitute for loader fixtures or a tensor/GPU test. Port only the actual
  guarded GGUF path, preserving weight alias auditing and projector capability,
  then run the historical loader/vision/dispatch regressions against v0.5.20.
- The native XGrammar, full parser import, final FP8/GGUF model, multimodal,
  tool/SSE and service-level gates remain unperformed. No all-model closure is
  claimed. Gemma and Qwen2.5 do not need inherited Qwen-specific patches merely
  to make their archive lists nonempty; their own runtime/configuration remains
  separately unqualified on this union.

Design ablation: the named-reference fix copies two existing definition-root
fields at the wrapping boundary. Reusing that historical four-line behavior
avoids a new resolver, schema rewriting layer or network-fetch fallback. Removing
it reproduces the three red errors. No additional profile or CI is introduced.
