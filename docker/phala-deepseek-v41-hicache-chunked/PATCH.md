# DeepSeek V4.1 HiCache chunked write-through repair

This overlay preserves completed chunked-prefill nodes through SGLang's
existing `BackupKV` action and ACK path. It does not count a request's own
chunks as cache hits and does not add a second D2H copy or TP collective.

The path is limited to ordinary `write_through` with HiCache enabled and the
Python Unified tree. It excludes selective write-through, write-back, root or
zero-page results, already-backed nodes, Rust/custom trees, buffer mode, and
external linkers.

DeepSeek patch commit: `1cfea70cea7ded5447fd06ced313e0145a50cf5c`

Reference implementation reviewed from the GLM task:

- Source baseline: `f876e4b593bec9d4878c2a2c0741091a8fae4c8c`
- Patch commit: `cf439b00cda90fdc05869f78fb6cc044ce3756b5`
- Patch SHA-256: `953d3b662397832f61d58f41fc2e540e23c8cf223ab6b7cb0de0f5dd17aa26bd`

The GLM canary image is not reused. DeepSeek requires its own MLA, SWA,
DSpark, and TP4 runtime acceptance.
