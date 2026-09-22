# Muse source migration, 2026-09-22

This is a CPU/source stage, not image, GPU, native grammar or model acceptance.
It preserves the frozen engine428 and the separate frozen PIG candidate.

## Historical semantic inputs

| Historical source | Current treatment |
| --- | --- |
| `405fa1086fdb40d97959ccde22c404e3d3b6748f` | Reuse existing Muse ATEM/channel detector and JSON fallback; do not duplicate it |
| `c5af5ab92184f5d29e4c12ac7d8e8cf81fba46a5` | Adapt token-level channel-header wait and actual grammar acceptance accounting; preserve current min-think and request-specific terminator APIs |
| `889dac086485d3b29d119e8957ce75de718e199f` | Constrained-output capability, constructor/call-site propagation, history normalization and assistant reasoning alias |
| `7aad27962602cc89876ca2c3b536a95aa15a46cd` | Muse-only JSON direct-final default; explicit reasoning wins |
| `c5052c199e05166199b3cb01e4e2adc2d4ed3bba` | Named choice remains constrained independently of native-required capability |
| `7070885b6c511f61d8ae2c4787d7d2dc38fd3e9b` | Required choice has a schema/cardinality constraint; remove duplicate `True` overrides |
| `2224d5c899711426210957da0f3ba341f6b6c110` | Muse-scoped strength mapping; retain current shared schema roots, mask callbacks and visibility/exclusion precedence rather than replacing them with older code |
| `69743bca633b582d9df2bbb41c3a0f3d09926d41` | Exact chosen JSON schema/response format reaches opt-in baked template kwargs |
| `ed4266b4513fd6b23023aed93c00a65eb3fdc434` | Pass the selected constraint through actual `_process_messages` → `_apply_jinja_template` → `_render_and_encode_chat_template` |

The channel matcher uses the full ` to=self<|message|>` token sequence, not
just a recipient prefix. Non-channel models retain the previous compact
reasoning-match history and generation counter instead of new per-token state
snapshots. Chat and Responses both route Muse constrained output through the
native detector; schema enforcement and native parsing are separate capabilities.

## External template identity

The supplied historical Compose source is
`2e16dfd8ab1b433e549fb1dc7cdd200dbd326bbd`, path
`release/sglang/v0.5.19/historical-inputs/muse/release/phala/muse-nemotron-colocation-v0519/muse-glimmer-chat-template.jinja`.
Its bytes are SHA256
`900db3effc316e33295ec3d7dfa2df83ea2735228cba73adba8fecc2e83343f7`.
The historical image path was `/opt/muse-glimmer-chat-template.jinja`.
The source tests require that exact hash through `MUSE_TEMPLATE`; the template
is not silently substituted for a checkpoint template and is not copied into
this engine repository.

The existing template contains the constrained JSON instructions and direct-final
generation prefix. This stage fixes its engine call sites; it does not build,
install or verify an image containing it. A future recipe must explicitly bind
the template artifact above. No standalone template license was found in the
supplied historical artifact; retain that provenance and resolve redistribution
terms before packaging it elsewhere. No template redistribution is performed here.

## CPU evidence and limitations

Run `test/manual/phala/test_muse_migration_cpu.py`. Its source-method tests
execute real detector, schema builder, request normalizer and reasoner code.
Pydantic validates and serializes the real assistant message class. The optional
template tests execute the three real serving call sites against the pinned
template in an immutable Jinja sandbox. Protocol envelopes, unrelated media
helpers, tokenizer encoding and the inner grammar are test doubles: this does
not establish native XGrammar compilation, real tokenization or SSE behavior.
Without `MUSE_TEMPLATE`, the three artifact-dependent tests are explicitly skipped.

The isolated local test packages are Jinja2 3.1.6 (BSD-3-Clause), MarkupSafe 3.0.3
(BSD-3-Clause), jsonschema 4.26.0 (MIT) and Pydantic 2.13.5 (MIT); bundled package
license files remain with the task-local dependencies. These test dependencies
are not changes to the engine/image dependency pins.

Historical XGrammar 0.2.1 source
`5b4e9ce9e72524037ae24ecd831b9b6604d2eb48` versus 0.2.6 source
`bc09a30ec10ba30a6c1ab0c79eaeba3ca518d11f` still needs native API/semantic
reconciliation. No version bump or co-installation is used as evidence.
Cancellation, actual tokenizer/native grammar integration, Linux full runtime
imports and model/GPU acceptance remain separately scoped.
