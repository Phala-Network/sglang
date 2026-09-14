# Muse-Glimmer-30B and Nemotron 3.5 Lightning SGLang v0.5.19 runtime

The r6 Marlin numerical repair is documented in
[`NVFP4_DETERMINISTIC_REDUCTION.md`](NVFP4_DETERMINISTIC_REDUCTION.md). It makes
expert routing stable and uses FP32 reduction for NVFP4 MoE and dense layers.
Runtime/protocol qualification remains separate from the kernel evidence.

The r5 request-cancellation repair is documented in
[`CANCELLATION_LIFECYCLE.md`](CANCELLATION_LIFECYCLE.md). It preserves the r4
protocol behavior and prevents disconnected requests from continuing to decode
after their tokenizer state is removed. Deployment qualification is separate
from the source and image release.

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
The r4 XGrammar dependency is rebuilt from the complete, hash-pinned 0.2.6
source distribution with the bounded exact-object-order patch and version
`0.2.6+phala.strictorder2`. The second patch backports upstream PR #880 at
`78ed389a31499a4a4fc20e275536774a465b5120`: length-bounded strings accept legal
JSON escapes, reject unescaped control characters, and count a Unicode
surrogate pair as one decoded character. PR #880 was open at review time.
All patch hashes and the final changed-file hashes are verified in the build.
A tests-only third patch aligns two inherited exact-order converter snapshots;
their behavior assertions are retained. The installed-package tests also cover
decoded string lengths and JSON escape validity at both root and nested positions.
See `xgrammar-source.lock.json` and
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

The SGLang r4 source also preserves literal reasoning markers inside Nemotron
answer content, and rejects JSON schemas that combine `pattern` with
`minLength` or `maxLength` instead of silently dropping the length constraints.
This rejection adapts SGLang PR #37726; it does not claim full support for the
intersection of arbitrary regular expressions and string-length constraints.
Request budget exhaustion remains truncation, not a valid JSON completion.

Nemotron assistant-history normalization strips only an opening reasoning
prefix and its first closing delimiter. Literal think tags in final JSON or
tool-associated assistant content remain intact across subsequent user turns.
The installed-template regression covers separate and combined reasoning,
recent and truncated older history, and content with and without tool calls.
The opt-in Nemotron template also distinguishes literal caller data from
reasoning/tool framing during tokenization. A shadow render locates `<think>`,
`</think>`, `<tool_call>` and `</tool_call>` in message content, separate
reasoning fields, historical tool-call data, and the response schema/tool
metadata. Only those data spans use ordinary BPE pieces, preserving the
tokenizer's original vocabulary and merges. Template-generated reasoning and
tool controls are unchanged. Rendered text and decoded prompt equality are
checked explicitly; requests, model artifacts, and the shared tokenizer are
not modified. Other model parsers, non-opt-in templates, native input IDs,
and continuation paths keep their existing behavior.

This addresses a reproduced NVFP4/EAGLE failure where literal added-token IDs
in message data caused JSON strings to repeat escapes until the token budget
was exhausted. It does not guarantee arbitrary model semantics or rewrite a
malformed generated response. The regression includes user/system/tool/assistant
content, Unicode offsets, separate/combined reasoning history, opt-in routing,
and fail-closed render/token-offset checks. Production weight, KV, SSM-state,
context, and speculative-decoding settings are not changed by this repair.

Nemotron response parsing additionally uses the existing internal generated
token IDs to distinguish a true reasoning closer from the same characters
quoted as ordinary BPE text inside reasoning. Previously, the first textual
`</think>` could prematurely end reasoning and leak the remaining analysis
into JSON content even when the model's final JSON was valid. The parser
decodes the prefix once when the real closing control token arrives, tracks
its exact character offset, and handles both cumulative and incremental IDs.
Stream/non-stream paths and each sampled choice preserve their own boundary
state. Leading literal opening tags remain data; malformed/truncated output
does not become a successful completion. Other models, legacy callers without
output IDs, and assistant continuation preserve their existing behavior.

Deterministic regressions exercise literal closers/openers, chunk boundaries,
buffered and emitted reasoning, truncation, and cumulative/incremental token
delivery. A separately recorded real-token replay verifies the original
failure without regenerating or rewriting the model output. SGLang PR #37365
was reviewed but not applied: it exposes public stream token IDs and does not
fix the reasoning/content separation; the internal token IDs already exist.

For Nemotron JSON and tool requests with an explicit completion limit and no
explicit thinking budget, r4 preserves reasoning enablement and effort while reserving final-answer
space: `max(128, min(4096, completion_limit // 2))` tokens. A request-scoped
thinking budget activates the grammar's existing token filter; global strict
thinking need not be enabled. Client limits, explicit thinking budgets,
ordinary chat, other model families and genuine length finishes are preserved.
This is a bounded default, not a guarantee that every requested JSON value
fits in the client's completion budget.

Nemotron's measured template also receives the actual response_format: JSON
mode is made explicit, and json_schema requests render their original schema
into the model's system context. Previously the schema existed only in the
grammar, so reasoning could plan an incompatible answer and stall in allowed
whitespace during generation. Existing messages and reasoning controls are
preserved. Grammar enforcement remains enabled; this is not output rewriting
or a substitute for validation. Ordinary chat, other models, preencoded input
and assistant continuation keep the previous template path.

An asynchronous grammar error in the first streaming event becomes a real
HTTP 4xx/5xx before headers are committed. An error after output starts remains
an SSE event. No false success usage is emitted after an initial error.
Parallel-sampling cleanup removes undispatched parent placeholders after fresh
sample IDs are dispatched; actual sample and unrelated in-flight state remains
intact. This prevents streamed `n>1` requests from leaving stale shutdown state.

Native XML tool parameter conversion also resolves local JSON pointers against
the original full tool schema. Referenced objects, arrays and scalars retain
their JSON types; references are not fetched over the network, and cyclic
references terminate without mutating the caller's schema. Upstream PR #31692
only covers the separate named-choice constraint hoisting path, not this parser.
Type lookup also preserves const-only integers, booleans, arrays and objects.
A non-null string declaration keeps literal `null` text as a string; explicitly
nullable parameters retain the established bare-null conversion. The related
open PR #36835 addresses empty tags and non-string `None`, but does not repair
this non-null-string or const-only type loss, so it is not applied here.
