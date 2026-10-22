# Official 0.5.22 compatibility maintenance

The shared0.5.20 main remains the integration source for its current consumers. Official0.5.22 is a real compatibility exception: complete source is maintained on `codex/sglang-v0.5.22-maintenance`, with a single version-scoped ordered export. This main documentation commit does not contain the0.5.22 engine bytes.

- Official baseline: `baa9befa381511cd4d56f83980951e5a45fffc6d`.
- Complete engine: [`386bc8376cac5121fe93ccdb75954631a7a4e64c`](https://github.com/Phala-Network/sglang/commit/386bc8376cac5121fe93ccdb75954631a7a4e64c), tree `af69b818250468199e696d43b83b092eb0481c43`; [source compare](https://github.com/Phala-Network/sglang/compare/baa9befa381511cd4d56f83980951e5a45fffc6d...386bc8376cac5121fe93ccdb75954631a7a4e64c).
- Ordered [patch export and replay](https://github.com/Phala-Network/sglang-serving-patches/tree/79ac4c5f552a5b435283073b9e9d0d67a0881a29/versions/v0.5.22): patch commit `79ac4c5f552a5b435283073b9e9d0d67a0881a29`, six exact intermediate-tree matches; original four patch bytes preserved.
- Planned immutable source tag/image version: `phala-v0.5.22-xing4-altstream-20261022`. Publication, immutable image digest and final-image/GPU acceptance are pending.

The stack retains necessary common serving contracts and scoped Xing model/native compatibility repairs, then the measured target MoE stream. Its stdlib CPU harness passed3 tests/144 dispatch subcases. The frozen recipe retains BF16 weights, FP8 KV, TP1, native262144, EAGLE1/1/2, chunk8192, graph2048 and HRRN. Governor is not added; external PIG and shared Mooncake remain independently pinned. No kernel-source, graph8192, varlen, DualGemm, global-prefix or interval change is selected.

Build and installed-image checks belong to the fixed engine's [`docker/xing4-0522`](https://github.com/Phala-Network/sglang/tree/386bc8376cac5121fe93ccdb75954631a7a4e64c/docker/xing4-0522), using official amd64 base `docker.io/lmsysorg/sglang@sha256:5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e`. XGrammar0.2.7 retains the existing wheel/native identity. Source replay and CPU evidence do not replace installed-image, real GPU, numerical/content, protocol, context or performance gates; existing failures and41 bounded-escape limitations remain visible in the migration ledger.
