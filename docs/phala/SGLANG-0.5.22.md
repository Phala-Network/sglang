# Official 0.5.22 compatibility maintenance

The shared 0.5.20 main remains the integration source for its consumers. Complete 0.5.22 source is maintained on `codex/sglang-v0.5.22-maintenance`; this main entry does not contain its engine bytes.

- Official baseline: `baa9befa381511cd4d56f83980951e5a45fffc6d`; CUDA13 amd64 base `docker.io/lmsysorg/sglang@sha256:5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e`.
- Complete engine: [`bd1ecb0fc6e056b22e7d41ecbf2c81c40ff19403`](https://github.com/Phala-Network/sglang/commit/bd1ecb0fc6e056b22e7d41ecbf2c81c40ff19403); [official compare](https://github.com/Phala-Network/sglang/compare/baa9befa381511cd4d56f83980951e5a45fffc6d...bd1ecb0fc6e056b22e7d41ecbf2c81c40ff19403).
- Ordered [patch export](https://github.com/Phala-Network/sglang-serving-patches/tree/83246ddd075ef90a8d8e8b6df8cb9c00348cd8c6/versions/v0.5.22): eight steps replay to tree `225466cc00f58fc49e7165dbb92d28bff4550cba`.
- Current [source Release](https://github.com/Phala-Network/sglang/releases/tag/phala-v0.5.22-xing4-or29b-fp8-20261023) records image digest and qualification status. It is a GPU0 FP8 trial candidate; GPU/runtime/production gates are pending.

Xing now uses the supplied clamped, transposed Sinkhorn mixing contract for both small/decode and large-prefill paths. Other models retain their kernel default. The previous target-only MoE stream is removed; existing native BCG, protocol/SO and XGrammar fixes remain. Latest official FP8 target and bundled draft use the checkpoint FP8/block128 configuration. No dependencies or native packages are rebuilt.

Source CPU: 37 passed. Installed CPU: 188 passed, 2 missing-tokenizer-fixture skips; direct HF entrypoints, text launcher and TOKEN auth/redaction pass. CPU results do not establish GPU lowering/graphs, model quality, throughput or production acceptance. Existing numerical/content and XGrammar limits remain visible in prior releases. Build and focused GPU probe belong to the fixed engine's [docker/xing4-0522](https://github.com/Phala-Network/sglang/tree/bd1ecb0fc6e056b22e7d41ecbf2c81c40ff19403/docker/xing4-0522).

The previous [altstream release](https://github.com/Phala-Network/sglang/releases/tag/phala-v0.5.22-xing4-altstream-20261022) retains its immutable source and historical evidence; the live GPU0 deployment was rolled back before this trial.
