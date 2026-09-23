# Shared Hopper/Blackwell KV-transfer extension

This narrow build compiles the fixed sibling `../aot/csrc/kvcacheio/transfer.cu`
and its device-address helper without rebuilding common_ops or FlashMLA.
It registers its 15 APIs under `phala_kvcache_incremental`. The sibling
sgl_kernel wrappers select this backend lazily for CUDA capabilities 9.0,
10.0 and 10.3, using the tensors' actual device (or the explicit pointer
target). Importing a wrapper does not load the extension or initialize CUDA.
HIP, non-CUDA and other capabilities retain the original operator namespace.
The CPU cache-copy operation is not replaced.

Build in the target image's exact PyTorch/CUDA environment:

```sh
python -m pip wheel --no-deps --no-build-isolation \
  -Ccmake.define.PHALA_FIXED_AOT_ROOT=/sgl-workspace/sglang/python/sglang/kernels/aot \
  -w /build/wheels /sgl-workspace/sglang/python/sglang/kernels/kvcache_incremental
```

The single wheel contains `sm_90` and `sm_100f` code, not model-specific wheels.
The selected capabilities and both code targets must be checked together when
changing the build. This is a source/build contract, not GPU qualification.
Use the exact target PyTorch/CUDA ABI; no cross-version ABI compatibility is
implied. Native compilation, code-object inspection and CPU operator loading
do not establish numeric correctness, registration, streams or CUDA Graphs.
`source-lock.json` pins the transfer source in `sources`; `donor_preimages`
retains historical audit hashes, not current wrapper/allocator inputs.
The complete engine commit binds the current wrapper, host-pointer integration
and generator. The generator uses only the transfer-source entry during
compilation. Final-image checks must verify the installed wrapper against that
complete engine, not merely check that the private wheel is present.

The address helper follows the fixed CUDA implementation. Host allocation
performs and checks explicit `cudaHostRegister` calls. An ordinary CPU
allocation returning an address is not itself a registration failure.
Qualification must verify real registered transfers and reject a missing
pointer implementation rather than silently selecting raw addresses. A missing
private backend on a selected CUDA device is an error, not a fallback to older
operators. Run `tests/qualify_registered_transfer.py` plus the registered MHA/MLA
and graph tests on each applicable device family before release acceptance.

The final image must install the AOT wheel and the corresponding wrapper.
There is no startup JIT compilation, package download or source mount.
The surrounding SGLang repository license and source notices apply.
