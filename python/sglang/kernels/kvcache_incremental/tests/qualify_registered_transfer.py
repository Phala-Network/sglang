from __future__ import annotations

import json
import os
import subprocess
import sys
from unittest.mock import patch

import torch

from phala_kvcache_incremental import ensure_loaded


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def require_missing_private_pointer_fails_closed() -> None:
    probe = r"""
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.mem_cache.pool_host import common

common._resolve_device_accessible_ptr_fn.cache_clear()
try:
    with patch.object(
        common.torch.ops,
        "phala_kvcache_incremental",
        SimpleNamespace(),
    ):
        try:
            resolve = common._resolve_device_accessible_ptr_fn()
            resolve(torch.empty(1, device="cpu"), 0)
        except (ImportError, AttributeError):
            pass
        else:
            raise RuntimeError("missing private pointer op did not fail closed")
finally:
    common._resolve_device_accessible_ptr_fn.cache_clear()
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "missing-private-pointer subprocess failed: "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )


def main() -> int:
    require(torch.cuda.is_available(), "real CUDA device is required")
    require(
        torch.cuda.get_device_capability(0) in {(9, 0), (10, 0), (10, 3)},
        "A configured Hopper/Blackwell GPU is required",
    )
    ensure_loaded()

    from sgl_kernel import kvcacheio
    from sgl_kernel.kvcacheio import (
        get_device_accessible_ptr,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_per_layer_mla_pf_lf,
    )
    from sglang.srt.mem_cache.pool_host.common import (
        _CUDA_HOST_REGISTERED_RANGES_ATTR,
        HostTensorAllocator,
        _cuda_host_unregister,
        alloc_with_host_register,
        make_kernel_ptr_table,
    )

    namespace = torch.ops.phala_kvcache_incremental
    require(
        hasattr(namespace, "get_device_accessible_ptr"), "private pointer op missing"
    )
    require(
        hasattr(kvcacheio, "_device_transfer_ops"),
        "Installed sgl_kernel wrapper does not implement unified device dispatch",
    )
    require(
        kvcacheio._device_transfer_ops(0) is namespace,
        "Installed wrapper does not select the qualified private backend",
    )

    pageable = torch.empty(4096, dtype=torch.uint8, device="cpu")
    try:
        pageable_alias = get_device_accessible_ptr(pageable, 0)
        pageable_probe = {
            "resolved": True,
            "same_address": pageable_alias == pageable.data_ptr(),
        }
    except RuntimeError as error:
        pageable_probe = {"resolved": False, "error": str(error)}

    try:
        get_device_accessible_ptr(pageable, -1)
    except RuntimeError:
        invalid_device_rejected = True
    else:
        raise RuntimeError("negative target device index unexpectedly passed")

    require_missing_private_pointer_fails_closed()

    pages, layers, width = 4, 3, 64
    dtype = torch.bfloat16
    item_size = width * dtype.itemsize
    page_dim = layers * item_size
    host = alloc_with_host_register(
        (pages, layers, 1, width),
        dtype,
        "cpu",
        True,
        HostTensorAllocator(),
        registration_granularity_bytes=page_dim,
    )
    try:
        registered_ranges = getattr(host, _CUDA_HOST_REGISTERED_RANGES_ATTR)
        expected_start = host.data_ptr()
        expected_end = expected_start + host.numel() * host.element_size()
        cursor = expected_start
        for start, size in registered_ranges:
            require(start == cursor, "registered host ranges are not contiguous")
            require(size > 0, "registered host range is empty")
            cursor += size
        require(
            cursor == expected_end, "registered host ranges do not cover the tensor"
        )

        values = torch.arange(host.numel(), dtype=torch.float32).reshape(host.shape)
        host.copy_(values.to(dtype))
        alias = get_device_accessible_ptr(host, 0)
        table = make_kernel_ptr_table([host], "cuda:0", host_memory_registered=True)
        require(
            table.cpu().tolist() == [alias], "pointer table did not use qualified alias"
        )

        src_indices = torch.tensor([0, 2], dtype=torch.int64, device="cuda:0")
        dst_indices = torch.tensor([1, 3], dtype=torch.int64, device="cuda:0")
        per_layer = torch.zeros((pages, 1, width), dtype=dtype, device="cuda:0")
        packet = namespace.transfer_kv_per_layer_mla_pf_lf
        with patch.object(packet, "default", wraps=packet.default) as per_layer_call:
            transfer_kv_per_layer_mla_pf_lf(
                host,
                per_layer,
                src_indices,
                dst_indices,
                layer_id=1,
                item_size=item_size,
                src_layout_dim=page_dim,
            )
            require(
                per_layer_call.call_count == 1, "Per-layer copy bypassed private op"
            )
        torch.cuda.synchronize()
        torch.testing.assert_close(per_layer[1].cpu(), host[0, 1])
        torch.testing.assert_close(per_layer[3].cpu(), host[2, 1])

        source_layers = [
            torch.full((pages, 1, width), layer + 11, dtype=dtype, device="cuda:0")
            for layer in range(layers)
        ]
        source_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in source_layers],
            dtype=torch.uint64,
            device="cuda:0",
        )
        all_src_indices = torch.tensor([1, 3], dtype=torch.int64, device="cuda:0")
        all_dst_indices = torch.tensor([0, 2], dtype=torch.int64, device="cuda:0")
        host.zero_()
        packet = namespace.transfer_kv_all_layer_mla_lf_pf
        with patch.object(packet, "default", wraps=packet.default) as all_layer_call:
            transfer_kv_all_layer_mla_lf_pf(
                source_ptrs,
                host,
                all_src_indices,
                all_dst_indices,
                item_size=item_size,
                dst_layout_dim=page_dim,
                num_layers=layers,
            )
            require(
                all_layer_call.call_count == 1, "All-layer copy bypassed private op"
            )
        torch.cuda.synchronize()
        for layer in range(layers):
            torch.testing.assert_close(host[0, layer], source_layers[layer][1].cpu())
            torch.testing.assert_close(host[2, layer], source_layers[layer][3].cpu())
    finally:
        _cuda_host_unregister(host)

    print(
        json.dumps(
            {
                "pass": True,
                "device": torch.cuda.get_device_name(0),
                "namespace": "phala_kvcache_incremental",
                "pageable_pointer_probe": pageable_probe,
                "invalid_device_rejected": invalid_device_rejected,
                "missing_private_pointer_failed_closed": True,
                "registered_range_count": len(registered_ranges),
                "registered_ranges_cover_tensor": True,
                "registered_pointer_table": True,
                "per_layer_pf_lf": True,
                "all_layer_lf_pf": True,
                "installed_wrapper_selected_private_backend": True,
                "private_transfer_calls": 2,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
