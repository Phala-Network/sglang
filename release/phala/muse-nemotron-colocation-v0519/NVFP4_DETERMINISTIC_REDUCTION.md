# Marlin routing and NVFP4 reduction

Identical inputs can produce different outputs through two independent Marlin
paths: expert alignment uses cross-CTA atomic slot allocation, and matrix
reduction can accumulate partial products in the FP16/BF16 output buffer.
Stable routing alone does not remove the second source of numerical variation.

This repair adapts the deterministic alignment from SGLang PR
[#33511](https://github.com/sgl-project/sglang/pull/33511), head
`071ee44609d5d00f1dfc28dafcbd117443e787dc`, preserving the existing single-token
sort and deterministic tiny-CUDA path. The alignment option remains disabled
for other callers by default. Expert assignments and padding are unchanged.

It also disables atomic reduction for NVFP4 MoE, as already done for MXFP4.
The calls' existing `use_fp32_reduce=True` can then select actual FP32 scratch
reduction. For NVFP4 dense/shared-expert layers, an explicit FP32-reduction
request likewise takes precedence over the atomic-add performance heuristic.
Callers explicitly choosing `use_fp32_reduce=False` retain that heuristic.
SGLang PR [#26627](https://github.com/sgl-project/sglang/pull/26627), head
`47500a14155b1edc8505ffb707602df0119361c6`, describes the same atomic-reduction
source of nondeterminism for Kimi. Its global deterministic-mode and
DeepSeek-specific stream changes are not imported here. Both PRs were open
and unmerged when reviewed on 2026-09-14.

On H200, a synthetic NVFP4 MoE with Nemotron's actual dimensions (128 experts,
hidden2688, intermediate1856, top-k6) gave 12 distinct results in 12 identical
single-token calls and 10 distinct CUDA-graph replays with atomic reduction.
With stable routes and FP32 reduction, all repeated and graph results matched.
The dense shared-expert shape also reproduced atomic variation at 1, 6, 81,
and 337 tokens; FP32 reduction removed variation in those finite trials and
reduced relative error against the dequantized reference. These measurements
use seeded synthetic weights, not a captured hidden state or an end-to-end
quality guarantee. Kernel latency measurements are not serving throughput.

Regressions cover assignment completeness, invalid routes, stable packing,
repeated MoE/dense execution, CUDA graph replay, finite values, and numerical
agreement with dequantized weights. They must pass on the final installed
release image. Model-level acceptance must separately cover real ingress,
streaming, reasoning, tools, JSON/schema, cancellation and shared-card load.
No template, model artifact, context, KV/state dtype or speculative-decoding
setting is changed by this numerical repair. No generated tool calls are
rewritten or deduplicated.
