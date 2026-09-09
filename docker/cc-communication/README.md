# CC IPC all-reduce fusion candidate

This is a **candidate**, not a GPU-validated release. Base SGLang is
`c1ff6d9b34a3c5257c8eaf5222e8a0b18c017517`, with the exact dependency versions
and base image in `overlay-manifest.json`.

SGLang's CC dispatch follows the communication part of upstream PR
https://github.com/sgl-project/sglang/pull/31447 (head
`64121c62a8dca7f5d1fdd39c8155ac3b8fc9da70`). Its scheduler/D2H changes are
deliberately excluded. FlashInfer 0.6.18 already contains the multicast-free
IPC implementation from https://github.com/flashinfer-ai/flashinfer/pull/3993.

The detector also incorporates the PPCIE insight from upstream PR
https://github.com/sgl-project/sglang/pull/36810, reviewed at head
`3be871e94a2430c38a0fa64e54c925a3e07f83b7` (open, not merged). Both SGLang and
FlashInfer query the NVML Settings API first: protected multi-GPU PCIe mode
selects CC dispatch even if the legacy `ccFeature` is zero. Missing/unsupported
Settings APIs fall back to the State API; other query errors are not claimed
as successful CC detection. The exact nvidia-ml-py binding version is pinned
in the manifest.

## Independent D2H candidate

PR #36810's D2H worker is backported as a separate change, disabled unless
`SGLANG_CC_ASYNC_D2H=1`. This opt-in also requires CC/PPCIE detection and the
scheduler's existing overlap path. Keep `0` for the communication-only control.
The normal and delayed-sampling generation paths use the same helper; existing
pinned CPU destinations and `record_stream` lifetime protection are unchanged.
The result-completion handle is published before enqueue. Source-ready events,
a private copy stream and exception-carrying completion handles follow upstream.

Deterministic Linux thread tests reproduced two upstream lifecycle edge cases:
submitting after shutdown accepted work with no worker, and an idle worker held
the last completed callback (and its captured batch). This backport atomically
closes submissions on shutdown and releases completed work before waiting for
the next item. Bounded shutdown returns its state so graceful GPU teardown can
be skipped if the worker is still active. These are bounded lifecycle fixes,
not evidence of a GPU memory leak in the live deployment.

The additional FlashInfer patch is paired with SGLang's CPU-only allgather:
after an actual local symmetric-memory allocation, successful tensors are
held while all ranks vote. A local allocation failure makes all ranks raise
before any enters rendezvous. This avoids the probe/free/reallocate race.
The SGLang dispatcher fails closed if this matching guard is absent, software
CC modes disagree, ranks disagree, or a CPU group is missing. Non-CC hardware
keeps the existing backend selection and memory preflight.

These checks protect the local allocation boundary. They do not claim to
recover from GPU/driver failures inside rendezvous or fused kernels. GPU
failures, graph replay and workspace lifetime require separate hardware gates.
Software CC overrides only affect library dispatch, never GPU security mode.
Actual NVML CC state must be verified on the target before promotion.

## Build context

Use this directory's Dockerfile, patch, installer and manifest, plus these
files copied into the context root using the committed LF line endings:

- `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- `python/sglang/srt/utils/confidential_compute.py`
- `python/sglang/srt/managers/async_d2h_copy_worker.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/utils.py` (as `managers_utils.py`)
- `test/registered/backends/test_flashinfer_cc_contract.py`
- `test/registered/backends/test_flashinfer_cc_gloo.py`
- `test/registered/backends/test_async_d2h_cc_contract.py`

The installer checks the exact original and patched SHA-256 of all changed
runtime files. No package version upgrades, startup downloads or dependency
installs are introduced. Build as linux/amd64, with the commit timestamp as
SOURCE_DATE_EPOCH and BuildKit SBOM/provenance enabled. Local images, OCI
artifacts, registry publication and live service acceptance are separate gates.

## Tests and deployment boundary

The CPU contract suite checks dispatch, detection, rank agreement and actual
allocation-before-rendezvous ordering. The Gloo suite uses eight real local
processes, with CUDA allocation mocked, and injects failures at rank 0 and 7.
Neither is an eight-GPU numerical or performance test.
The D2H CPU suite uses real threads with simulated device events/streams; it
does not establish CUDA correctness, CC overlap speed or graph-buffer lifetime.

Before promotion, require eight-rank NVML CC agreement, multicast-free IPC
workspace, all-reduce/residual-RMSNorm numerical parity, oneshot/twoshot,
CUDA graph replay, cleanup/recreation, model protocol gates and same-workload
performance/stability evidence. Preserve all existing model, KV, context,
EAGLE, batching and PIG parameters. Drain the exact target deployment before
any change to a live model instance.

`gpu_preflight.py` is an executable eight-B200 gate that reads hidden size
from the pinned local checkpoint. It tests real NVML agreement, injected
rank-local allocation failure before rendezvous, 1/8/32/48/288/384-token
all-reduce and residual RMSNorm in oneshot/twoshot modes, graph capture/replay,
three workspace lifecycles and pinned D2H parity. CPU tests cannot run this
gate. `run_verified_server.py` runs it only with `PHALA_CC_GPU_PREFLIGHT=1`,
with a bounded process-group timeout, then execs the original server command.
Any failure refuses model launch. Use only after routing is down and drained.

## Target-GPU findings and follow-up fixes

The first real eight-B200 run exposed two errors in the preflight itself:
two-shot requires token_count > world_size (small shapes retain one-shot
coverage), and the numerical oracle must accumulate in FP32 rather than reuse
NCCL's BF16 intermediate rounding. The original tolerances are unchanged;
both references and individual discrepant coordinates are logged for audit.
This does not change the model's FP8 E4M3 KV cache or model compute dtype.

The next run passed every legal numeric/graph shape but found that FlashInfer
0.6.18's TRTLLM workspace destroy method did not release its IPC allocator
registry entry and retained memory through its internal creation tuple.
The bounded fix calls the existing low-level release API and clears that
tuple. Five deterministic regressions fail on the original destroy method
and pass on the repaired method: IPC references, non-CC tuple references,
idempotence, failed-release retry, and repeated creation/cleanup. Real GPU
lifecycle and end-to-end model acceptance remain independent requirements.

On this pinned CUDA 13 / driver 595.91.07 CC runtime, a direct `pin_memory=True`
allocation returns `is_pinned() == False`: CUDA reports memory type Managed (3),
but `cudaHostGetFlags` succeeds with flags 3. Pageable controls return type 0
and invalid-value from that query. The preflight therefore validates the CPU
destination through both native host-allocation queries, allowing Managed
reporting only after real CC detection. The CUDA 13 pointer-attribute ABI
includes its eight-long reserved tail. No PyTorch memory behavior is patched.

The repaired preflight passed on all eight B200 ranks on 2026-09-09: 14 legal
numeric/graph cases per rank, 16 graph replays per case, three complete workspace
lifecycles, and 64 exact D2H copies per rank. Source contract suites total 56
passing cases. This is hardware correctness evidence, not a throughput or
production-readiness claim; model and real ingress validation are separate.
