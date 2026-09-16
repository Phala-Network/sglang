# Cancellation and scheduler ownership

The r5 repair adapts upstream SGLang PR [#35255](https://github.com/sgl-project/sglang/pull/35255),
merged as `f478b2bb2d582c09e7f1b4e49f0c2d039da8747a`, to this runtime's
synchronous feature-dispatch API. It retains the r4 protocol/template and
XGrammar behavior. Related PRs #35936 and #36418 were closed without merge;
their titles are not treated as released fixes.

A cancelled generator previously removed TokenizerManager state before its
delayed abort ran. The missing-state guard then suppressed AbortReq while
the scheduler continued decoding. This was reproduced on the unchanged r4
H200 dev using a bounded native streaming request and a real TCP disconnect.
Running requests remained nonzero for the full 16-second observation after
disconnect; the synthetic request eventually stopped at its token limit.

The repair records scheduler dispatch, aborts dispatched requests on handler
failure, and retains state until the scheduler responds. It tracks actual
parallel-sampling IDs and separately discards undispatched parent placeholders.
Abort dispatch is deduplicated; failed sends remain retryable. Deferred
chunked-prefill aborts are retried if the request has moved to another queue.
Unknown/stale public request IDs remain guarded because scheduler matching
is prefix-based. This change does not add a public wildcard recovery API.

Regressions cover cancellation, GeneratorExit, generated choice ownership,
pre-dispatch failures, delayed/duplicate aborts, dispatch failures, parent
placeholders and the chunked-prefill transition. Production acceptance must
also prove actual scheduler termination and memory release, preserving normal
and concurrent survivor requests. Persistent orphan-output logging is a
stability failure even when the process and /v1/models remain healthy.
