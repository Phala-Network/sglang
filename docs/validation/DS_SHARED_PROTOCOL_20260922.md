# DeepSeek shared protocol closure (2026-09-22)

Independent branch `codex/ds-protocol-union-20260922` starts at
`8b9ae03f3d9818fbe69486cf1c8ecf444daefab9`. This is a source/CPU completion,
not an image, GPU, native serving import, HTTP/SSE or provider acceptance.
No integrator worktree, XGrammar, schema compiler, server arguments, CI,
patch exports, image, GPU, CVM or route was changed.

## Historical mapping and adaptation

| Contract | Exact donor and path | Native union implementation |
| --- | --- | --- |
| DS-only asynchronous conversion | `437b982abfdfa1b563122e061ead1a1cd1211c29:python/sglang/srt/entrypoints/openai/serving_chat.py` | `_convert_to_internal_request_async` uses the existing serialized worker only for `dsv41`, absent caller input IDs and an enabled batcher. |
| Modality, URI, inline size | `5ad27e1db0d14d17d920e9040599f20655fbe9df:python/sglang/srt/phala_compat/dsv41_media_hardening.py` | Extend the existing shared media validator. Image/audio capability is the actual ModelConfig flag; video requires a processor video token or token ID, including ID zero. Reuse common URL/domain checks, parse URI structure, bound inline decoded size and reject malformed base64. No fail-open wrapper or broad OSError conversion. |
| Integer effort | `437b982abfdfa1b563122e061ead1a1cd1211c29:python/sglang/srt/phala_compat/dsv41_protocol_compat.py` | Chat-only strict integer transport `[1,100]`, then actual server encoder admission in `_validate_request`; non-DS models reject it before conversion. The shared `ReasoningEffortType` is unchanged. No model-name heuristic or global Pydantic monkey-patch. |
| Explicit tool none | `5ad27e1db0d14d17d920e9040599f20655fbe9df:python/sglang/srt/phala_compat/dsv41_tool_choice_none.py` | Existing top-level and message-level prompt filters suppress definitions without mutating the typed request. Allowed/named choices keep existing behavior. |

The one required shared file beyond `serving_chat.py` is `protocol.py`: its
Chat field must transport an integer before model-aware server validation can
run. Booleans, integer-valued floats such as `75.0`, numeric strings such as
`"75"`, values outside the range, NaN and invalid styles are rejected.
Existing fractional inputs `[0,.99]` are unchanged, including legacy integer
zero coercion to fractional zero. Nested reasoning retains the union's
existing precedence and fractional contract; a nested integer 75 remains
invalid. Template kwargs retain their existing explicit override semantics.
Output exclusion does not disable generation. Qwen/Muse token budgets are not
reinterpreted as DS effort budgets.

`chat_encoding.py` has a documentation-only correction; its native parser is
unchanged. The older DS migration fixture's `medium` rejection assertions were
stale after the prior native DS donor migration and now verify accepted medium.

## Executed CPU checks

| Test file | Result |
| --- | --- |
| `test/manual/phala/test_dsv41_shared_protocol_cpu.py` | 11 passed |
| `test/manual/phala/test_dsv41_chat_migration_cpu.py` | 10 passed |
| `test/phala_deepseek_v41/test_cpu_contracts_v0520.py` | 8 passed |
| `test/registered/unit/function_call/test_qwen_complete_call_cpu.py` | 16 passed |
| `test/manual/phala/test_muse_migration_cpu.py` | 28 passed, no skips with pinned template |

Total: 73 tests. Ruff F821 passed for changed production source; F401/I and
format checks passed for the new regression; `git diff --check` passed.

The new suite imports the entire actual protocol module with real Pydantic
2.13.5/OpenAI 3.17.0 types; only the unrelated `sglang.utils` schema serializer
import is substituted. It executes unmodified production methods extracted
from the serving class and common media helper, the native DS encoder, and
the real serialized async executor. Infrastructure/model/tokenizer boundaries
are doubles. Actual `_validate_request` plus `_apply_jinja_template` produces
DS prompt bytes containing effort 75 but neither tool carrier's definitions.
This does not establish checkpoint tokenization, full serving startup, native
grammar, image loading, HTTP/SSE, model weights or GPU execution.

Local reproducible dependencies are at
`tmp/ds-protocol-deps-20260922` relative to the Compose workspace, with Jinja
from `tmp/union-integration-deps-20260922`. Python is
`C:/Users/zozyo/AppData/Local/Programs/Python/Python313/python.exe`.
Run the five files above with `python -B <test-file> -v`.

For Muse, `MUSE_TEMPLATE` must point to the file extracted under
`tmp/ds-protocol-muse-fixture-20260922/` from Compose commit
`2e16dfd8ab1b433e549fb1dc7cdd200dbd326bbd`, path
`release/sglang/v0.5.19/historical-inputs/muse/release/phala/muse-nemotron-colocation-v0519/muse-glimmer-chat-template.jinja`.
The test verified SHA256
`900db3effc316e33295ec3d7dfa2df83ea2735228cba73adba8fecc2e83343f7`.
An earlier local candidate failed that hash check and was not accepted; the
immutable artifact then passed all three template-call-chain tests. No fixture
template was added to this engine repository.

## Ablation and limits

No historical wrapper/import-hook framework was restored: the existing native
extension point and common filters suffice. The executed ablation removes only
the integer model guard from the actual AST; Qwen then wrongly accepts effort
75. With the guard retained, every tested non-DS encoder rejects it. The media
test injects OSError through the common validator and verifies it propagates,
rather than restoring the donor's overly broad client-error classification.

The previous connected DS vision, Engram and Python C1/C2 consumers remain
untouched. Their source connectivity and earlier CPU evidence do not constitute
GPU or device-transfer qualification. This patch does not add DS audio/video
support, global streaming usage defaults, or global effort semantics.
