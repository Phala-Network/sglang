# Official Kimi K3.1 compatibility maintenance

This is the approved maintenance line for the official K3.1 developer image and
its native dependency stack. Other models' main runtime remains independently
maintained.

- Official image source: `afcd7d6b487a9cbe73e25941971b4cc107e6f9ae`, recovered
  from its image checkout; public parent `db0dfc72286e6322fb79762d4e2d568b438af66c`.
- Complete source: [`308921ab7f196ce41e866b878aa6c8565a982033`](https://github.com/Phala-Network/sglang/commit/308921ab7f196ce41e866b878aa6c8565a982033),
  on `codex/kimi-k31-pro-20261105`.
- Source Release: [`phala-dev-kimi-k31-4`](https://github.com/Phala-Network/sglang/releases/tag/phala-dev-kimi-k31-4).
- [Build and final-image qualification](https://github.com/Phala-Network/sglang/tree/308921ab7f196ce41e866b878aa6c8565a982033/docker/phala/k31/video-20261107).
- [Eleven-step source-derived export](https://github.com/Phala-Network/sglang-serving-patches/tree/6187fb710d3d6a8aeb336f9774777c629bee14cf/versions/dev-kimi-k31)
  actually replays to `c001d76ae187bdcb92be1dfd697edbbc1d0ad109`, matching every
  intermediate tree and preserving the previous eight patches byte for byte.

Published image:
`ghcr.io/phala-network/sglang:phala-dev-kimi-k31-4@sha256:1b56364d12871d614c429af1fe1f6f8a1428af730d90fc13996fbc43e76494ea`.

Image4 adds timestamped video input, bounded CPU staging and owner-side vision
groups, temporal position device placement and owned graph outputs. It also
fixes hybrid L3 partial allocation, chunks Mamba temporal copies, and corrects
cached usage after excluding K3's generation stub. It retains image3's hybrid
SSM/COW retry and replay-aware HRRN, packaged HF implementations, complete-byte
Mooncake reads and the fixed native rail client. Governor is not added;
external PIG remains separately versioned.

The single final image passes 475 CPU tests and 314 subtests, with two explicit
inherited baseline-only comparisons skipped. All five packaged-HF tests, an
additional unchanged vision CPU suite, native Mooncake and fastsafetensors APIs,
SGLang serve, and direct HF executable checks pass without source mounts or
runtime installation. All 82 inherited image3 layers, runtime configuration,
package versions and pip-check results are preserved. Source marker and OCI
revision identify the complete source above. Registry tag and immutable digest
read back to that qualified image.

GPU video semantics and rollout are separate deployment receipts. CPU checks
do not prove throughput gains, broad codec support or full-concurrency capacity.
Video consumes no audio track. The 4 GiB budget estimates per-request CPU
staging; other codec/upload/embedding/KV memory is extra. Native decoder calls
are not preemptible; cancellation is cooperative at chunk boundaries.

The [image3 Release](https://github.com/Phala-Network/sglang/releases/tag/phala-dev-kimi-k31-3)
and [image2 Release](https://github.com/Phala-Network/sglang/releases/tag/phala-dev-kimi-k31-2)
retain their immutable source, image and qualification records. The native
client remains [`467533b2e80a101dfabd4789ed5aa8ab251c24a6`](https://github.com/Phala-Network/Mooncake/commit/467533b2e80a101dfabd4789ed5aa8ab251c24a6),
tag [`phala-v0.3.13.post1-k31-client-1`](https://github.com/Phala-Network/Mooncake/releases/tag/phala-v0.3.13.post1-k31-client-1).
