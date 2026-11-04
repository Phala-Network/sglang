# Official Kimi K3.1 developer-image compatibility line

The shared main runtime remains on its existing baseline for other consumers.
The user approved a scoped maintenance-line release for the official K3.1
developer image and its incompatible native dependency stack.

- Official image source: `afcd7d6b487a9cbe73e25941971b4cc107e6f9ae`, recovered
  from its shallow image checkout; public parent `db0dfc72286e6322fb79762d4e2d568b438af66c`.
- Complete patched source: [`03cf254955c2606744c8eafc641822a93244c325`](https://github.com/Phala-Network/sglang/commit/03cf254955c2606744c8eafc641822a93244c325).
- Immutable release: [`phala-dev-kimi-k31-1`](https://github.com/Phala-Network/sglang/releases/tag/phala-dev-kimi-k31-1).
- [Source-owned build and qualification](https://github.com/Phala-Network/sglang/tree/03cf254955c2606744c8eafc641822a93244c325/docker/phala/k31).
- [Two-step source-derived patch export](https://github.com/Phala-Network/sglang-serving-patches/tree/main/versions/dev-kimi-k31).

The runtime repair rejects incomplete Mooncake object reads before restoring
hybrid cache pages. Native model optimizations, dependencies, Kimi protocol
handlers and DFlash grammar-mask reset stay upstream. External PIG is separate;
no old strict schema stack or Governor component is injected. The source tree
is not merged into the incompatible main engine tree.

CPU development passes 150 tests and 11 subtests. The Release records final
image identity and installed-image qualification separately. GPU/model,
RoCE/shared-KV and production acceptance belong to deployment receipts;
neither a source release nor a successful CPU suite establishes those gates.
