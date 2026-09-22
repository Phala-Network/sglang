# B300 KV-transfer extension

This narrow build compiles the fixed sibling `../aot/csrc/kvcacheio/transfer.cu`
and its device-address helper without rebuilding common_ops or FlashMLA.
It registers its 15 APIs under `phala_kvcache_incremental`; the sibling
sgl_kernel wrappers and host-pool helper route registered transfers here.

Build in the target image's exact PyTorch/CUDA environment:

```sh
python -m pip wheel --no-deps --no-build-isolation \
  -Ccmake.define.PHALA_FIXED_AOT_ROOT=/sgl-workspace/sglang/python/sglang/kernels/aot \
  -w /build/wheels /sgl-workspace/sglang/python/sglang/kernels/kvcache_incremental
```

The sole CUDA target is `sm_100f`. This is a B300 build variant, not a claim
that the binary supports other GPU architectures or PyTorch ABIs.
`source-lock.json` pins the transfer source; its other hashes record the
preimages used to generate the routing changes now present in this tree.
The generator uses only the transfer-source entry during compilation.

The address helper follows the fixed CUDA implementation. Host allocation
performs and checks explicit `cudaHostRegister` calls. An ordinary CPU
allocation returning an address is not itself a registration failure.
Qualification must verify real registered transfers and reject a missing
private pointer implementation rather than silently selecting raw addresses.

The final image installs the AOT wheel and the corresponding wrapper.
There is no startup JIT compilation, package download or source mount.
The surrounding SGLang repository license and source notices apply.
