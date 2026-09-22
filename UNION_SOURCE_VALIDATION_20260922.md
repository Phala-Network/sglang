# Unified source integration checks, 2026-09-22

This is source and CPU evidence, not a runtime image, GPU, ABI, HTTP/SSE,
provider, performance or production acceptance. The paired serving-patches
selector binds the final complete engine commit/tree and each ordered delta.
Existing PRs 121 (engine) and 6 (patches) are reused.

## Installed native dependency

The single version is `xgrammar==0.2.6+phala.union1`. Its deterministic public
upstream-plus-delta recipe is documented in `docker/phala-xgrammar/README.md`.
Serving-patches dependency commit `345f57ee3ff447679a9f4d426b6d61ada95c0e12`
pins native source `8830951a797886cb09568dd0ae2f9b77b9aab552`, tree
`b8dcb87b50b99cc61dbab46e7dcfbd34f78a2f45`, delta SHA256
`c6a4defdfe0967056e144351ff2027132c3fa4cb79026c862bff0306e946e26b`.
The native candidate commit is local provenance, not a claimed public ref.
There is no public Phala wheel claim.

Existing 655 installed native CPU passes and the same-base ablation
(22 semantic failures, 21 passes) are retained. The wheel was built from
`8b5d3b860de06a8f8016ec1c7b131c58e2055229`; its successor adds tests only.
Wheel SHA256 `d62d888568626676837ff64709cd4a3dccae4bbf43401bf7d9c0e4422dcafe2e`,
installed library SHA256
`778e9ac15a67a2bf731057de0834e71874e490f9fa777a627f0b4527f76ee7c4`.

The engine adapter added 14 passing tests, zero skips, using that installed
library in the existing CPU builder, runc, network none and no GPU. Candidate
tar SHA256 `c246bfb9981325967ba84f75fc06471ea889cb8fefedded82c5d6e0fe2c1c316`;
complete log SHA256
`b95e0ac94316fd3914eb4c05b5f5ae7691b48fae6dee44e67675d996093519f1`.
Files are in the custodian's `tmp/xgrammar-audit-20260922/`.
The adapter source thereafter changed only through formatting.

Cases cover direct and ANY JSON bounds, per-format limit preservation,
schema-position-only fail-closed checks, legacy and nested JSON/XML structural
formats, masks/rollback/EOS and invalid backend/compact/tokenizer settings.
The logged fallback-to-none belongs to an unset-limit compatibility control;
an explicit positive bound with that tokenizer raises instead.

The recipe itself was run with `--verify-only` in a fresh source directory:
public immutable fetch, delta application, complete tree, license, DLPack
URL and gitlink all passed. A first exact URL comparison exposed the missing
`.git` suffix in the manifest and was corrected before the passing run.
No wheel rebuild, image build/push or service change occurred.

## Integrated CPU checks

- Muse: 30 passed, zero skips with historical template SHA256
  `900db3effc316e33295ec3d7dfa2df83ea2735228cba73adba8fecc2e83343f7`.
  This includes the actual process/Jinja/render call chain, inner rejection
  state consistency and 100,000 final tokens retaining only header snapshots.
- Nemotron: 19 passed with the pinned real tokenizer, plus 10 focused
  termination/budget cases. Cuts cannot turn unfinished thinking/tool examples
  into content; real closers retain legitimate final calls. No-token-context
  normal EOF compatibility remains. Structured JSON/tool requests reserve
  answer space while explicit budgets, disabled reasoning and continuations
  remain unchanged. Missing token filtering aborts an explicit budget.
- XML const: shared complete-call executor suite now has 17 passes, including
  numeric/boolean/object/array/null/string const intersections and unchanged
  generated values. Atomic withholding and other detector consumers remain.
- DeepSeek: 8 encoder/async, 16 production-function CPU, 3 MXFP4 load-order,
  10 chat and 11 shared-protocol tests passed on the integrated tree.
  Vision, Engram and Python C1/C2 consumers are connected source paths.
  DS-only async conversion, modality/URI/inline-size validation, strict
  top-level integer effort with model-aware rejection and both tool carriers
  for explicit none are integrated. Historical monkey-patches are not restored.
- Nearby Qwen history 5, Marlin/BREAKABLE 6 and HiCache source 13 passed;
  HiCache real Gloo is explicitly skipped in this source-only run.
  Existing unchanged GGUF20/Q8-prefill11 CPU evidence is retained, not GPU proof.

Python 3.12 with the existing isolated Torch 2.13.0 CPU environment was used for
DeepSeek tensor methods. The unrelated system Python 3.10/Torch 1.12 trial
could not execute two modern torch contracts and was not accepted.
Dependency-light CI intentionally skips artifact/native-only classes when
their pinned inputs are unavailable. Local artifact/native results above
are separate executed evidence, not silently converted CI skips.

## Simplification and boundaries

The termination guard ablation disables only `_set_finish_reason` on the
actual parser: the truncated tool example then enters normal content.
Restoring it yields reasoning only. Model-local DS integer rejection and
finite UE8M0-scale guard have separate negative controls. Existing native
extension points and shared validators replace historical wrapper layers.
Muse final-output snapshots were eliminated in favor of its existing counter;
this is CPU bookkeeping evidence, not measured serving throughput.

Source integration does not prove loaded weights, native extension compilation,
device transfers, Mooncake ACK/reload, model generation, new image acceptance
or any Governor combination. Muse template packaging/license remains a
separate gate. See the historical semantic reconciliation before claiming
unreviewed Nemotron fusion/Mamba behaviors are complete.
