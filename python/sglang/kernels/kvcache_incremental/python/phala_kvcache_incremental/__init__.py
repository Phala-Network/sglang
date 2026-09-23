from __future__ import annotations

from pathlib import Path
import threading

import torch


_LOCK = threading.Lock()
_LOADED = False
_NAMESPACE = "phala_kvcache_incremental"
_REQUIRED_OPS = (
    "get_device_accessible_ptr",
    "transfer_kv_per_layer",
    "transfer_kv_per_layer_pf_lf",
    "transfer_kv_per_layer_ph_lf",
    "transfer_kv_all_layer",
    "transfer_kv_all_layer_lf_pf",
    "transfer_kv_all_layer_lf_ph",
    "transfer_kv_per_layer_mla",
    "transfer_kv_per_layer_mla_pf_lf",
    "transfer_kv_all_layer_mla",
    "transfer_kv_all_layer_mla_lf_pf",
    "transfer_kv_direct",
    "transfer_embedding_ranges_direct",
    "transfer_kv_per_layer_direct_pf_lf",
    "transfer_kv_all_layer_direct_lf_pf",
)


def ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    with _LOCK:
        if _LOADED:
            return
        native_dir = Path(__file__).resolve().parent / "_native"
        candidates = sorted(native_dir.glob("phala_kvcache_ops*.so"))
        if len(candidates) != 1:
            raise ImportError(
                f"expected exactly one phala_kvcache_ops shared library, found {candidates}"
            )
        torch.ops.load_library(str(candidates[0]))
        namespace = getattr(torch.ops, _NAMESPACE)
        missing = [name for name in _REQUIRED_OPS if not hasattr(namespace, name)]
        if missing:
            raise ImportError(f"focused KV extension did not register: {missing}")
        _LOADED = True


def op_namespace():
    ensure_loaded()
    return getattr(torch.ops, _NAMESPACE)


__all__ = ["ensure_loaded", "op_namespace"]
