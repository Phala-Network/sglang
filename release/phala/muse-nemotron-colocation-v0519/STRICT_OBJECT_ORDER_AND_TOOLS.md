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
text. Ordinary text before a tool marker remains allowed. The existing native
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
fallback. Non-reasoning tool replies are not deferred. Other model detectors
retain their existing default behavior. Chunk-boundary, implicit/explicit
reasoning, hidden reasoning and missing-closer controls cover this repair.

Source tests, diagnostic-image experiments, final-image qualification, registry
publication and production rollout remain separate gates. Failed original runs
must remain in release evidence; a prompt A/B pass alone is not qualification.
