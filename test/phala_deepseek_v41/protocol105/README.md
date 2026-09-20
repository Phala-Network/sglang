# Finite DeepSeek V4.1 protocol fixture

The runner preserves JSON insertion order on the wire and separately records canonical semantic hashes and exact UTF-8 body hashes. Schema property order affects the actual XGrammar 0.2.1 grammar: sorting the `city,country,population,notable` properties changes the enforced generation order. Hash canonicalization must not silently become request serialization.

This fixes a test-tool fidelity defect, not a serving/runtime defect. Old sorted-wire manifests and responses must remain separately identified. On the frozen full-context repair image, the original-order matrix still failed11/35 strict cases; preserving order is not a claim that model output now satisfies the gate. Do not add maxItems, increase the2000-token budget, repair returnedJSON or relabel truncation as success.

The54 requests retain the same semantic values and prompts as the preceding fixture;35 are the exact reasoning-off/conflicting-prompt matrix,15 fixed and20 seeded random cases. Per-body wire hashes and a regression against order-only drift make the new replay identity explicit. A separate test captures the actual `HTTPConnection.request` bytes, so changing only a hash helper cannot falsely pass serialization validation.

Run offline checks with `python test_assets.py -v`. These tests make no real network requests. `run_protocol.py --static-check` additionally validates schema structure. Live execution is explicit, serial and no-retry; it requires a supplied runtime identity and the authorized rootTOKEN through stdin, and never obtains credentials from another source. Its bound endpoint is the authorized test fixture's localhost18301; source tests or this file do not grant inference authority elsewhere.

The fixture proves only its finite direct endpoint cases. Backend cancellation cleanup, native masked-logit correctness, throughput, provider-chain behavior and model semantic quality need their own evidence. The cancel row intentionally remains pending external backend evidence.
