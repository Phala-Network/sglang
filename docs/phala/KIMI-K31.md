# Official Kimi K3.1 developer-image compatibility line

The shared main runtime remains on its existing baseline for other consumers.
The user approved a scoped maintenance-line release for the official K3.1
developer image and its incompatible native dependency stack.

- Official image source: `afcd7d6b487a9cbe73e25941971b4cc107e6f9ae`, recovered
  from its shallow image checkout; public parent `db0dfc72286e6322fb79762d4e2d568b438af66c`.
- Complete patched source: [`254af1d9de5eb6baf68d72e961dcb3116969acfa`](https://github.com/Phala-Network/sglang/commit/254af1d9de5eb6baf68d72e961dcb3116969acfa).
- Immutable source release: [`phala-dev-kimi-k31-2`](https://github.com/Phala-Network/sglang/releases/tag/phala-dev-kimi-k31-2).
- [Source-owned build and qualification](https://github.com/Phala-Network/sglang/tree/254af1d9de5eb6baf68d72e961dcb3116969acfa/docker/phala/k31).
- [Four-step source-derived patch export](https://github.com/Phala-Network/sglang-serving-patches/tree/55818d50266f6f3a1984d1c514fb57d1840404f3/versions/dev-kimi-k31)
  reproduces complete tree `86714f67cd66d0bc8399e485effd2bd12c4ada1e`.
- Native client: [`467533b2e80a101dfabd4789ed5aa8ab251c24a6`](https://github.com/Phala-Network/Mooncake/commit/467533b2e80a101dfabd4789ed5aa8ab251c24a6),
  source release [`phala-v0.3.13.post1-k31-client-1`](https://github.com/Phala-Network/Mooncake/releases/tag/phala-v0.3.13.post1-k31-client-1).

The runtime repair rejects incomplete Mooncake object reads before restoring
hybrid cache pages. K3 tokenizer and processor implementations are packaged in
the engine; checkpoint directories supply metadata and BPE data without
executing checkpoint Python. The native client adds opt-in GID rail groups;
the optional fastsafetensors 0.5.0 loader uses a hash-pinned wheel installed
without dependency resolution. The fixed recipe records the retained official
native modules and qualification gates. External PIG is separately versioned.

Canonical patch bytes and all four intermediate trees match the frozen source.
Development checks pass five packaged-HF cases and 26 native worker-policy CPU
cases. The export preserves the two image1 patch files byte for byte;
[`phala-dev-kimi-k31-1`](https://github.com/Phala-Network/sglang/releases/tag/phala-dev-kimi-k31-1)
retains its immutable source and validation record.

Image2 publication and final-image qualification are pending; its source Release
will record the immutable image digest and installed-image results. GPU/model,
RoCE/shared-KV and production acceptance belong to deployment receipts.
