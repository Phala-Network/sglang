# Official Kimi K3.1 developer-image compatibility line

The user approved a scoped maintenance line for the official K3.1 developer
image and its incompatible native dependency stack. Main provides this entry
to its complete source and releases.

- Official image source: `afcd7d6b487a9cbe73e25941971b4cc107e6f9ae`, recovered
  from its shallow image checkout; public parent `db0dfc72286e6322fb79762d4e2d568b438af66c`.
- Complete repaired source: [`204565cce1d82988c4f7235c6113fe1883c0f467`](https://github.com/Phala-Network/sglang/commit/204565cce1d82988c4f7235c6113fe1883c0f467),
  on `codex/kimi-k31-pro-20261105`.
- Immutable source release: [`phala-dev-kimi-k31-3`](https://github.com/Phala-Network/sglang/releases/tag/phala-dev-kimi-k31-3).
- [Incremental build and qualification](https://github.com/Phala-Network/sglang/tree/204565cce1d82988c4f7235c6113fe1883c0f467/docker/phala/k31/repair-20261106).
- [Eight-step source-derived patch export](https://github.com/Phala-Network/sglang-serving-patches/tree/b6ec6e621af2c875f6b5ee63c9646b32c9e70662/versions/dev-kimi-k31)
  actually replays to tree `91b79deceac376c186949758889e8cba1004a3fa`, matching
  every intermediate tree and preserving the four image2 patches byte for byte.
- Native client: [`467533b2e80a101dfabd4789ed5aa8ab251c24a6`](https://github.com/Phala-Network/Mooncake/commit/467533b2e80a101dfabd4789ed5aa8ab251c24a6),
  source release [`phala-v0.3.13.post1-k31-client-1`](https://github.com/Phala-Network/Mooncake/releases/tag/phala-v0.3.13.post1-k31-client-1).

Image3 retries hybrid-SSM admission after capacity returns, guards COW capacity
and includes retracted output in HRRN replay cost. It adapts open upstream PRs
[37612](https://github.com/sgl-project/sglang/pull/37612) and
[44867](https://github.com/sgl-project/sglang/pull/44867); upstream approval or
merge is not claimed. The four changed runtime files are Python only. All 79
image2 layers, runtime configuration, package inventory and pip-check result
are preserved. Governor is not added; external PIG remains separately versioned.

Published image:
`ghcr.io/phala-network/sglang:phala-dev-kimi-k31-3@sha256:08acd79a162875bc9bfded2ff42cd526a8b0b4e1c4f96e32c1997b61246f226b`.
Final-image qualification passes 162 CPU tests and 21 subtests, plus all five
HF cases. Two explicit old-source comparisons are skipped in the full-import
suite and covered separately by 14 source-method before/after cases. Native
CPU APIs, source identity, SGLang serve and direct HF executable checks pass
without source mounts, runtime installs, network access or model downloads.
The installed marker `/sgl-workspace/sglang/.k31-source-commit` and OCI revision
both bind the image to the complete source above.

CPU/image qualification does not establish GPU or production acceptance, a
unique cause of the 2026-11-06 incident, or permanent resolution. Full-budget/HOL
waiting and long-fill retry costs remain limits. Deployment acceptance is a
separate receipt. The source Release records the image and build-helper caveats.

Historical [`image2`](https://github.com/Phala-Network/sglang/releases/tag/phala-dev-kimi-k31-2)
preserves source `254af1d9de5eb6baf68d72e961dcb3116969acfa` and digest
`sha256:edc2bb25b3c7f7d67a902e0b7e3a99f1ef8a1371e8395f950ecbb06cb9f3db62`.
Its inherited repairs reject incomplete Mooncake object reads, package pinned
K3 tokenizer/processor code, and supply opt-in GID rail groups and the pinned
fastsafetensors 0.5.0 loader. The earlier
[`image1`](https://github.com/Phala-Network/sglang/releases/tag/phala-dev-kimi-k31-1)
retains its immutable source and validation record.
