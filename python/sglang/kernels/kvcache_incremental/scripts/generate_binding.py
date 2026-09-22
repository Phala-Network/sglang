from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PRIVATE_NAMESPACE = "phala_kvcache_incremental"
CPP_NAMESPACE = "phala_kvcache_incremental_impl"


REGISTRATION = r'''

TORCH_LIBRARY_FRAGMENT(phala_kvcache_incremental, m) {
  m.def("get_device_accessible_ptr(Tensor tensor, int device_index) -> int", &phala_kvcache_incremental_impl::get_device_accessible_ptr);
  m.def("transfer_kv_per_layer(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor dst_indices, int item_size, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer", c10::DispatchKey::CUDA, &phala_kvcache_incremental_impl::transfer_kv_per_layer);
  m.def("transfer_kv_per_layer_pf_lf(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor dst_indices, int layer_id, int item_size, int src_layout_dim, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_pf_lf", c10::DispatchKey::CUDA, &phala_kvcache_incremental_impl::transfer_kv_per_layer_pf_lf);
  m.def("transfer_kv_per_layer_ph_lf(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor dst_indices, int layer_id, int item_size, int src_layout_dim, int page_size, int head_num, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_ph_lf", c10::DispatchKey::CUDA, &phala_kvcache_incremental_impl::transfer_kv_per_layer_ph_lf);
  m.def("transfer_kv_all_layer(Tensor src_k_layers, Tensor dst_k_layers, Tensor src_v_layers, Tensor dst_v_layers, Tensor src_indices, Tensor dst_indices, int item_size, int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer", c10::DispatchKey::CUDA, &phala_kvcache_incremental_impl::transfer_kv_all_layer);
  m.def("transfer_kv_all_layer_lf_pf(Tensor src_k_layers, Tensor dst_k, Tensor src_v_layers, Tensor dst_v, Tensor src_indices, Tensor dst_indices, int item_size, int dst_layout_dim, int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_lf_pf", c10::DispatchKey::CUDA, &phala_kvcache_incremental_impl::transfer_kv_all_layer_lf_pf);
  m.def("transfer_kv_all_layer_lf_ph(Tensor src_k_layers, Tensor dst_k, Tensor src_v_layers, Tensor dst_v, Tensor src_indices, Tensor dst_indices, int item_size, int dst_layout_dim, int num_layers, int page_size, int head_num, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_lf_ph", c10::DispatchKey::CUDA, &phala_kvcache_incremental_impl::transfer_kv_all_layer_lf_ph);
  m.def("transfer_kv_per_layer_mla(Tensor src, Tensor dst, Tensor src_indices, Tensor dst_indices, int item_size, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_mla", c10::DispatchKey::CUDA, &phala_kvcache_incremental_impl::transfer_kv_per_layer_mla);
  m.def("transfer_kv_per_layer_mla_pf_lf(Tensor src, Tensor dst, Tensor src_indices, Tensor dst_indices, int layer_id, int item_size, int src_layout_dim, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_mla_pf_lf", c10::DispatchKey::CUDA, &phala_kvcache_incremental_impl::transfer_kv_per_layer_mla_pf_lf);
  m.def("transfer_kv_all_layer_mla(Tensor src_layers, Tensor dst_layers, Tensor src_indices, Tensor dst_indices, int item_size, int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_mla", c10::DispatchKey::CUDA, &phala_kvcache_incremental_impl::transfer_kv_all_layer_mla);
  m.def("transfer_kv_all_layer_mla_lf_pf(Tensor src_layers, Tensor dst, Tensor src_indices, Tensor dst_indices, int item_size, int dst_layout_dim, int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_mla_lf_pf", c10::DispatchKey::CUDA, &phala_kvcache_incremental_impl::transfer_kv_all_layer_mla_lf_pf);
  m.def("transfer_kv_direct(Tensor[] src_layers, Tensor[] dst_layers, Tensor src_indices, Tensor dst_indices, int page_size) -> ()");
  m.impl("transfer_kv_direct", c10::DispatchKey::CUDA, &phala_kvcache_incremental_impl::transfer_kv_direct);
  m.def("transfer_embedding_ranges_direct(Tensor src, Tensor! dst, int[] src_starts, int[] dst_starts, int[] lengths) -> ()");
  m.impl("transfer_embedding_ranges_direct", c10::DispatchKey::CUDA, &phala_kvcache_incremental_impl::transfer_embedding_ranges_direct);
  m.def("transfer_kv_per_layer_direct_pf_lf(Tensor[] src_ptrs, Tensor[] dst_ptrs, Tensor src_indices, Tensor dst_indices, int layer_id, int page_size) -> ()");
  m.impl("transfer_kv_per_layer_direct_pf_lf", c10::DispatchKey::CUDA, &phala_kvcache_incremental_impl::transfer_kv_per_layer_direct_pf_lf);
  m.def("transfer_kv_all_layer_direct_lf_pf(Tensor[] src_ptrs, Tensor[] dst_ptrs, Tensor src_indices, Tensor dst_indices, int page_size) -> ()");
  m.impl("transfer_kv_all_layer_direct_lf_pf", c10::DispatchKey::CUDA, &phala_kvcache_incremental_impl::transfer_kv_all_layer_direct_lf_pf);
}
'''


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def generate(source: Path, lock_path: Path) -> bytes:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = lock["sources"]["csrc/kvcacheio/transfer.cu"]
    raw = source.read_bytes()
    if digest(raw) != expected:
        raise RuntimeError("transfer.cu hash does not match source-lock.json")
    text = raw.decode("utf-8")
    include = '#include "pytorch_extension_utils.h"\n'
    if text.count(include) != 1:
        raise RuntimeError("unexpected pytorch_extension_utils include shape")
    text = text.replace(
        include,
        "// pytorch_extension_utils.h is unused by this translation unit.\n",
    )
    anchor = "inline void* resolve_device_accessible_ptr"
    offset = text.find(anchor)
    if offset < 0 or text.find(anchor, offset + 1) >= 0:
        raise RuntimeError("unable to locate unique namespace insertion anchor")
    prefix = text[:offset]
    body = text[offset:]
    return (
        prefix
        + "#include <torch/library.h>\n\n"
        + f"namespace {CPP_NAMESPACE} {{\n\n"
        + body
        + f"\n}}  // namespace {CPP_NAMESPACE}\n"
        + REGISTRATION
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generated = generate(args.source, args.lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(generated)
    print(json.dumps({"output": str(args.output), "sha256": digest(generated)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
