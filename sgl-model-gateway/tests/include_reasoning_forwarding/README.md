# include_reasoning forwarding CPU regression

The gateway pins openai-protocol 1.0.0, whose original ChatCompletionRequest
silently discards the unknown include_reasoning field. Both HTTP PD and regular
OpenAI routes serialize that typed request. The in-tree crate adds only
Option<bool> with skip_serializing_if=Option::is_none: false/true survive,
and absence preserves worker defaults. The gateway workspace patch binds the
same local protocol crate to all its consumers. No worker suppression or usage
rewriting is introduced.

Run `cargo test --locked --manifest-path Cargo.toml --test forwarding` here.
This separate workspace avoids compiling the unrelated gateway dependency
graph. Tests use the real protocol Deserialize/Serialize implementation,
Arc-to-JSON boundary and reqwest JSON transport to two loopback mock workers.
They cover false/true/absent crossed with stream/nonstream, n=2 and thinking
retention, unchanged response/usage bytes, and rejection of nonboolean values.
Both worker bodies preserve the requested stream value. They do not instantiate
the entire gateway, run models, or prove GPU behavior.

CPU development on .201 used the already-installed Rust/Cargo 1.98.1 toolchain
from immutable image config4551818943918f3f22e2b90de24abb97b08e99d91abf5c8b390d9c3b8b4fa335.
All development/cache data is under
`/opt/dstack/models/ds101-include-reasoning-source`. Original crate: 2 passed,
5 failed (exit101); fixed crate: 7 passed (exit0). Fixtures are identical across
RED/GREEN except subsequent rustfmt formatting. Source baseline is engine
9ac976f783ced6e2fb5c2ca37942bca4c4229981; existing n=2/bootstrap and NVLink
implementation files remain untouched.

Cargo.lock here freezes the focused test graph. The gateway root Cargo.lock
separately freezes release resolution (the published source previously tracked
no gateway lock). Use --locked for release builds and retain the workspace patch.
Dependency resolution and this CPU harness are verified; the full router build
and installed fixed wheel/binary HTTP acceptance are later release gates.

Design ablation: declaring this single typed field avoids generic unknown-field
flattening or a duplicate request wrapper. The vendored crate retains original
metadata, authors, Apache-2.0 license and archive SHA in PHALA-VENDOR.md.
