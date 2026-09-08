# CC IPC all-reduce fusion candidate

This is a **candidate**, not a GPU-validated release. Base SGLang is
`c1ff6d9b34a3c5257c8eaf5222e8a0b18c017517`, with the exact dependency versions
and base image in `overlay-manifest.json`.

SGLang's CC dispatch follows the communication part of upstream PR
https://github.com/sgl-project/sglang/pull/31447 (head
`64121c62a8dca7f5d1fdd39c8155ac3b8fc9da70`). Its scheduler/D2H changes are
deliberately excluded. FlashInfer 0.6.18 already contains the multicast-free
IPC implementation from https://github.com/flashinfer-ai/flashinfer/pull/3993.

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
files copied into the context root without changing their contents:

- `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- `python/sglang/srt/utils/confidential_compute.py`
- `test/registered/backends/test_flashinfer_cc_contract.py`
- `test/registered/backends/test_flashinfer_cc_gloo.py`

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

Before promotion, require eight-rank NVML CC agreement, multicast-free IPC
workspace, all-reduce/residual-RMSNorm numerical parity, oneshot/twoshot,
CUDA graph replay, cleanup/recreation, model protocol gates and same-workload
performance/stability evidence. Preserve all existing model, KV, context,
EAGLE, batching and PIG parameters. Drain the exact target deployment before
any change to a live model instance.
