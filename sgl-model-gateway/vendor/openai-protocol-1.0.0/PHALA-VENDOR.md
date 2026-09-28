# openai-protocol 1.0.0 vendor

Upstream archive: https://static.crates.io/crates/openai-protocol/openai-protocol-1.0.0.crate
Archive SHA256: b365334e1e57a6f57ed932951878b31c1fe9087a90149db6f5d071c3504f6dff
Upstream repository: https://github.com/lightseekorg/smg
Archive VCS commit: 7455eaee1da5212ae5e0fd33d80a903029d2f981 (protocols/).
License: Apache-2.0, declared by the original crate manifest; the standard license text is included as LICENSE. Original authors and .cargo_vcs_info.json are retained.

Only source change: ChatCompletionRequest gains Option<bool> include_reasoning with skip_serializing_if=Option::is_none. Explicit false is serialized. Upstream unknown-field behavior otherwise stays unchanged. The gateway workspace patch selects this deterministic in-tree crate; do not patch Cargo cache files.

The upstream archive contains a Cargo.lock; retained for provenance, not used as the gateway workspace lock. The committed gateway Cargo.lock freezes release resolution; tests/include_reasoning_forwarding/Cargo.lock freezes the focused CPU harness. Use --locked in both locations.
