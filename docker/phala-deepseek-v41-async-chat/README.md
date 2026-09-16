# DeepSeek-V4.1 opt-in asynchronous Chat conversion

This overlay changes three Python modules on the immutable Phala r3 image.
The existing r3 P/D gateway repair is retained. Model weights, CUDA kernels,
KV precision, context limits, parsers, admission policy, and credentials are
not changed by this overlay. No dependency is installed at service startup.

## Scope and activation

DeepSeek-V4.1's native Chat encoder creates `input_ids` synchronously before
TokenizerManager receives the request. This bypasses the existing async
single-string tokenization path and can hold the HTTP event loop while other
requests are streaming.

When `--enable-dynamic-batch-tokenizer` initializes its async tokenizer,
V4.1 Chat conversion now uses that tokenizer's single shared executor. The
operation is unchanged and is not additionally batched. Context variables,
arguments, exceptions, and return values are preserved. Queued cancellation
does not submit an inference request; already running CPU work may finish.

The default remains unchanged. Other encoders, disabled async tokenization,
and explicitly pre-tokenized Chat input continue on their original path.
The synchronous conversion method remains available to existing callers.

## Tests and known boundaries

The focused CPU suite covers opt-in routing, request context, one serialized
worker shared with text encoding, loop responsiveness, queued cancellation,
cancellation before inference submission, error status, and default paths.

Neighboring source tests also cover Chat, completions, and the existing
dynamic batch tokenizer. Two Chat-suite assertions already fail on the
unmodified r1 runtime: the empty-delta logprobs assertion and abort stream
chunk-count assertion. Those baseline failures are retained and are not
claimed fixed here. The offline GPT-2 integration fixture may be skipped;
that skip must be reported, not counted passed.

Source tests, built-image checks, GPU protocol acceptance, AgentX stability,
registry publication, and a production rollout are separate stages. This
source change alone does not prove improved serving performance or resolve
all strict-schema/model-quality failures.

## Reproducible image procedure

Build from a clean source commit with the pinned base, `SOURCE_DATE_EPOCH`,
`linux/amd64`, immutable source guards, BuildKit SBOM and maximal provenance.
Freeze build inputs before building; keep final image and result digests in
post-build artifacts rather than creating self-referential image metadata.

Verify the installed module hashes in the actual final image without a
serving-source overlay. Run the focused and neighboring CPU checks with
separately identified test assets, exercise the inherited r3 gateway probe,
and execute the final image's `hf --help` and `hf download --help` through
the direct downloader entrypoint. A second clean build must confirm the
runtime layer/config identity. Registry read-back must bind metadata,
SBOM, provenance, source tag and human version tag to the tested digest.

Use the same verified runtime image for the downloader and model backend.
The test-only pytest plugin is not installed in this runtime image.
