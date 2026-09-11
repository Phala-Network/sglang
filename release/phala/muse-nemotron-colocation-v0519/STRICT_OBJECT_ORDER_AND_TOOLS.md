# r3 protocol repair scope

This candidate preserves context, KV types, memory allocations and speculation.
It does not use XGrammar's approximate `any_order=True` option.

## Small-object property order

XGrammar issue [#831](https://github.com/mlc-ai/xgrammar/issues/831) permits only
declared-order properties, then additional properties. In model tests that can
force a declared key into a different key spelling or leave unfinished JSON.
The local dependency patch uses an exact subset-state grammar for ordinary JSON
objects with 1 to 8 declared properties, no property-count bounds and no combined
patternProperties/propertyNames override. A total 1024-state budget bounds the
construction across the schema. Larger or unsupported shapes retain the existing
strict fixed-order path, not a relaxed approximation. This is not universal
arbitrary-order support for every JSON Schema.

For the repaired shapes, each declared key is emitted at most once, required keys
must actually occur, original value schemas are retained and additional keys may
occur before/between/after declared keys. Additional keys cannot impersonate a
declared key through escaped spellings. While the key could still match a declared
name, canonical UTF-8 or short-escape spelling is required (unnecessary `\u`
aliases at that prefix are not emitted); after the first different decoded
character the normal JSON-string suffix is allowed. This restriction concerns key
encoding, not string values. Arbitrary duplicate *additional* keys and unrelated
upstream Schema limitations are not newly solved by this patch. XML converters
retain their existing path.

The dependency is rebuilt from the hash-pinned complete upstream source
distribution, with the public upstream commit and local patch recorded in
`xgrammar-source.lock.json`. Patched-file hashes are checked during the build.
The final image installs the complete dependency wheel; it does not overlay
individual dependency files or install packages at service startup.
The 99 main C++/header/Python/CMake/license/project files in the source
distribution were compared to the declared upstream commit: file contents
match after CRLF-to-LF normalization. The sdist's raw hash, not a normalized
tree hash, identifies the actual build input, including its bundled dependencies.

## Native tools and Nemotron template

The Qwen3-Coder native structural-tag path now honors `parallel_tool_calls` for
required, named, strict-auto and non-strict-auto paths. Required/named replies
use one or more whole calls and bounded whitespace, without free prose after
the calls. Named choice can repeat the named function when parallel calls are
enabled. Auto mode can still return normal text and history-based final answers.
For strict and non-strict automatic choice, the native constraint starts at
`<tool_call>`, not the longer `<tool_call>\n<function=` prefix. Malformed
function headers can no longer evade constraints by being treated as plain
text. Ordinary text before a tool marker remains allowed. After the first auto
invocation, the response stays in one final call phase: additional complete
calls, bounded whitespace or end of turn. It cannot resume arbitrary prose
between or after calls. Identical repeated invocations remain structurally
legal; the grammar does not deduplicate or infer a desired count.
The existing native
streaming parser also recognizes a bare `<function=...>` invocation without the
opening outer wrapper. Non-streaming now recognizes the same spelling, including
mixtures with wrapped calls, and auto grammar constrains both spellings with the
same tool-name and argument schemas. A bare function may retain or omit the
trailing outer wrapper. Required/named generation retains canonical wrapped XML.
As in upstream XGrammar, an explicitly non-strict function preserves loose
arguments; it does not acquire a strict-schema guarantee. The protocol tests
still validate actual arguments and semantic values against the supplied schema.
These constraints do not infer the number of desired calls from the user's text
and cannot guarantee arbitrary model-level semantic correctness.

`nemotron-lightning-chat-template.jinja` derives from
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` revision
`cc84af2fe71647d87f4486c064f320e1e7535243`. Its tool instructions separate planning
in reasoning from actual XML invocations, describe multiple independent calls,
and remove conflicting/example-heavy format discussion.
The rest of the official template, history, tool schema rendering and reasoning
toggle are preserved. The immutable runtime selects the baked template path
`/opt/phala/nemotron-lightning-chat-template.jinja` explicitly in Compose.

## Reasoning/tool stream boundary

Upstream SGLang [#30533](https://github.com/sgl-project/sglang/pull/30533)
(merged `0ee236ebdff36fe413d7ca2e7fc99875d5554c89`) added Nemotron's
`<tool_call>` interruption fallback for missing `</think>`. In a real reasoning
stream that quotes a tool example before a later canonical closer, that fallback
prematurely exposes reasoning as tool invocations. Non-streaming can already see
the closer and classifies the same text differently.

The Nemotron detector now defers this ambiguous suffix until `</think>` or EOF.
Text before the ambiguous marker can still stream as reasoning. A later closer
keeps quoted examples in reasoning; a missing closer at EOF retains the existing
fallback on a normal stop. The adapter passes the native finish type to parser
finalization: a length/abort cut inside unclosed reasoning does not promote
quoted tool examples (or force-nonempty reasoning) into executable final
content. Calls after an actual reasoning closer are unaffected. Non-reasoning
tool replies are not deferred. Other model detectors retain their existing
default behavior. Chunk-boundary, implicit/explicit reasoning, hidden reasoning,
budget-cut and missing-closer controls cover this repair.

Source tests, diagnostic-image experiments, final-image qualification, registry
publication and production rollout remain separate gates. Failed original runs
must remain in release evidence; a prompt A/B pass alone is not qualification.

## Emitted reasoning-token accounting

The candidate also carries the source change from upstream SGLang
[#37450](https://github.com/sgl-project/sglang/pull/37450), inspected at head
`e27d2d464151baed376e2c4d2ef60939737c13ac` (open, unmerged on 2026-09-12).
Speculative acceptance counts the complete accepted token run before stop/length
handling determines the emitted prefix. The reasoning count must be bounded by
that emitted prefix when a request finishes, including an EOS past the budget.
This is an O(1) accounting correction and does not change generated tokens,
sampling or the requested output budget. Regression cases cover length crossing,
EOS on either side of the cap, a reasoning closer on either side of the cap,
ordinary in-budget output and a multi-token closer spanning decode steps.

The XGrammar patch is a byte-addressed release input. Its scoped Git attribute
preserves the qualified raw bytes rather than applying workstation line-ending
normalization; the build continues to require its exact SHA-256 and a zero-fuzz
application with matching patched-file hashes.

## Deterministic dependency packaging

The XGrammar builder fixes `PYTHONHASHSEED=0` as well as `SOURCE_DATE_EPOCH`,
but a second independent build proved those settings alone insufficient.
Both builds still contained identical installed runtime code but different
wheel member and RECORD ordering. That changed the wheel digest and pip's
installed direct_url/RECORD metadata, so runtime image digests did not match.
Neither original reproducibility failure is waived or rewritten as a pass.

The build-only `canonicalize_wheel.py` now verifies every original RECORD
hash/size and complete membership, canonically orders the ZIP and RECORD,
then verifies payload/metadata preservation and idempotence. It rejects signed
wheels, non-regular or unsafe paths, duplicate or missing entries, weak hashes
and content/size mismatches. Twelve build-only tests cover ordering, successful
atomic replacement, hash/size corruption, path/signature/link rejection,
preservation on validation failure and failure to hide real payload, timestamp
or permission differences. The preserved two XGrammar wheels yield identical
canonical bytes without changing their non-RECORD payloads.

This step changes no model sampling, runtime hash seed, parser, template or
library payload, and the utility does not enter the final runtime. Only a fresh
two-clean-build comparison can qualify the resulting full-image packaging.
