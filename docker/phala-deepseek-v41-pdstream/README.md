# DeepSeek V4.1 P/D streaming gateway candidate

This candidate changes the SGL Model Gateway, not model weights, SGLang model
kernels, the context limit, precision, PIG admission or public routing.

The observed 8-B300 P/D trace produces token bursts after long TTFT. The current
gateway waits for the prefill HTTP response even when decode already streams.
Backport the narrowly scoped upstream PR
<https://github.com/sgl-project/sglang/pull/37961>, frozen at
`baf2b10b1802a8eca4ac7a6b90d1fe850a0b21b3` (open/unmerged at inspection).
The change commits ordinary streaming responses on decode's 2xx and watches the
prefill leg asynchronously, preserving body drain, failure reporting, paired
decode cancellation, and the existing nonstream/logprob paths.

Upstream's controlled delayed-prefill regression is included unchanged. Release
qualification also requires related routing/cancellation/error-path tests and
actual-image streaming probes. An open upstream PR is not itself acceptance.

Local additions mark early-stream breaker outcomes as relay-owned, preventing
an eager success followed by a duplicate result; a deterministic regression
covers pending and completed prefill counts. Transport diagnostics now retain
the underlying reqwest error chain in server logs, without changing retry or
error-response behavior. This logging change is not a fix for the observed 502s.
Late prefill failures emit `event: error` together with the JSON error envelope;
the installed AIPerf 0.12.0 transport recognizes named errors but ignores a
data-only error packet. The actual-image regression requires that event and
no false `[DONE]`, so failure cannot be presented as a clean completion.

Build-time support for SOURCE_DATE_EPOCH in
`sgl-model-gateway/build.rs`, to avoid embedding wall-clock build times into
otherwise identical runtime artifacts. Invalid explicitly supplied epochs fail.
The digest-pinned base provides the installed `stable` toolchain; the build
asserts its exact Rust/Cargo 1.98.1 versions without updating the toolchain.
Cargo.lock is resolved and frozen for this repair because no original lock was
available; the old binary's Rust dependency closure is not claimed identical.
The routing suite's Redis helper uses the redis-server executable from the
digest-pinned official Redis 7.2.8 Bookworm amd64 image, only in the build stage
and with network disabled. Redis is not copied into the final runtime or
installed on the host. Compile and execute tests in separate layers so a
test-harness failure preserves the compile artifacts for attributable retries.

The serving entrypoint loads the Python Rust extension, not the native binary.
Any image must therefore contain the newly built Python extension; replacing
only the `sgl-model-gateway` executable would not change the running code path.
All copied compiled artifacts must be tested from the image without source
mounts over installed packages. Model backends must not be silently restarted
or described as updated when only the gateway image is changed.

Status: source candidate only. No build, publication or runtime acceptance is
implied by this file; those stages require the separate release evidence.
