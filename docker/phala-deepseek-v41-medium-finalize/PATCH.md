# DeepSeek V4.1 medium target finalize plane

This overlay is a narrow adaptation of SGLang PR #39704 at donor head
`46a204a5eca58ec5aba316f9c538a871ddf5be69` and donor base
`72407c7b5513dd512a96b51013f90bf738fecc0f`.

It does not merge or cherry-pick the whole PR. It retains the existing C++
FP32 Top-K accumulation, routed BF16 rounding, shared BF16 add, and rank-sum
ordering. It adds:

- a separately named TP4 push-only communicator with 4 MiB slots and 512 row
  counters;
- an opt-in main-model target-verify gate for 97 through 384 rows;
- eager communicator initialization before CUDA graph capture;
- the medium FlashInfer autotune output allocation used by the donor path.

The existing 96-row plane remains the default for small batches and the draft
model. Prefill, NEXTN/draft, batch-invariant mode, unsupported topologies,
rows above 384, and insufficient workspace all fall back. The feature switch
`SGLANG_DSV41_MEDIUM_FUSED_FINALIZE_ALL_REDUCE` defaults to false and is a
startup/re-capture switch, not a hot runtime switch.

The complete PR's mHC, prefill, indexer, router-PDL, speculative metadata, and
argmax changes are excluded. HiCache/Mooncake code and cache formats are not
changed.

Release baseline:

- execution-time latest official stable: SGLang `v0.5.19`, commit
  `0bcd822377da7b5718e674eaf9c870d349424dd1`;
- stable anchor merge: `e04227fa6763885cb98af01aff1f643e3c4bfed9`;
- tested DSV4.1/HiCache parent: `b85aa875dd9118270f7c82efde296064807e321f`;
- performance patch: `7766836cbd42a520eee76be14ff6d3ed5071ee6d`.

The stable-anchor merge has the same Git tree as its DSV4.1 parent, because
the release fixes were already present under their upstream commits. No older
release implementation replaced the model-specific runtime.
