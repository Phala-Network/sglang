"""Bounded, on-demand host KV tensor accounting for scheduler server_info.

Storage addresses are used only inside one snapshot to union aliases. Neither
addresses nor tensor contents are serialized. This is not process RSS or a
measurement of allocator overhead, metadata, registration, or OS ownership.
"""

import os
import time
from numbers import Integral

import torch

MAX_ENTRIES = 64
MAX_TENSORS = 1024
# These implementations own data only through DATA_FIELDS. New pool kinds
# require an explicit review before this surface can claim complete coverage.
SUPPORTED_POOL_KINDS = {
    "LogicalHostPool",
    "DeepSeekV4PagedHostPool",
    "DeepSeekV4StateHostPool",
    "MHATokenToKVPoolHost",
    "MHATokenToKOnlyPoolHost",
    "AsymmetricMHATokenToKVPoolHost",
    "MLATokenToKVPoolHost",
    "MambaPoolHost",
}
DATA_FIELDS = (
    "kv_buffer",
    "k_buffer",
    "v_buffer",
    "index_k_buffer",
    "index_k_scale_buffer",
    "temporal_buffer",
    "conv_buffer",
)
SIZING_FIELDS = (
    "size_per_token",
    "item_bytes",
    "state_page_bytes",
    "num_host_pages",
    "layer_num",
    "page_num",
    "dcp_size",
)


def _integer(value):
    return (
        int(value)
        if isinstance(value, Integral) and not isinstance(value, bool)
        else None
    )


def _union_bytes(intervals):
    end = total = 0
    for start, stop in sorted(set(intervals)):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def _groups(scheduler):
    """Only known ownership paths; never traverse model/request object graphs."""
    cache = getattr(scheduler, "tree_cache", None)
    controller = getattr(cache, "cache_controller", None)
    primary = getattr(cache, "host_pool_group", None)
    if primary is None:
        primary = getattr(controller, "mem_pool_host", None)
    if primary is None:
        primary = getattr(cache, "token_to_kv_pool_host", None)
    if primary is not None:
        yield "hicache", primary
    offload = getattr(scheduler, "decode_offload_manager", None)
    writeback = getattr(offload, "decode_host_mem_pool", None)
    if writeback is not None:
        yield "decode_writeback", writeback
    sparse = getattr(scheduler, "hisparse_coordinator", None)
    sparse_pool = getattr(sparse, "mem_pool_host", None)
    if sparse_pool is not None:
        yield "hisparse", sparse_pool


def host_pool_observability(scheduler, *, role):
    """Return schema v1, scoped to this returned scheduler worker and snapshot."""
    started = time.time_ns() // 1_000_000
    report = {
        "schema": "sglang.host-pool-observability.v1",
        "scope": "host_pool_data_tensor_storages",
        "identity_scope": "this_worker_snapshot_only",
        "consistency": "capacity_per_pool_lock_backing_best_effort",
        "worker": {
            "pid": os.getpid(),
            "role": role,
            **{
                rank: _integer(getattr(scheduler, rank, None))
                for rank in ("tp_rank", "pp_rank", "dp_rank")
            },
        },
        "capture_started_unix_ms": started,
        "groups": [],
        "backings": [],
        "metadata_bytes": None,
        "metadata_status": "unknown",
        "allocator_overhead_bytes": None,
        "allocator_ownership_status": "unknown",
        "limits": {"entries": MAX_ENTRIES, "tensors": MAX_TENSORS},
        "logical_capacity_sum_semantics": "entry_capacities_including_aliases_not_usable_token_budget",
        "unique_backing_bytes_semantics": "union_of_observed_cpu_storage_address_spans",
    }
    # Keep Tensor/storage references alive until all address-based unions finish.
    references = []
    storages = {}
    all_intervals = []
    entry_count = tensor_count = 0
    complete = True
    for group_name, group in _groups(scheduler):
        group_complete = True
        entries = getattr(group, "entries", None)
        if entries is None:
            entries = [("kv", group)]
        elif isinstance(entries, (list, tuple)):
            if len(entries) > MAX_ENTRIES - entry_count:
                complete = group_complete = False
            entries = [
                (getattr(entry.name, "value", entry.name), entry.host_pool)
                for entry in entries[: MAX_ENTRIES - entry_count]
            ]
        else:
            complete = group_complete = False
            entries = []
        group_result = {"name": group_name, "entries": []}
        group_intervals = []
        for name, pool in entries:
            if entry_count >= MAX_ENTRIES:
                complete = group_complete = False
                break
            entry_count += 1
            result = {
                "name": str(name),
                "kind": type(pool).__name__,
                "layout": getattr(pool, "layout", None),
                "page_size": _integer(getattr(pool, "page_size", None)),
                "capacity_unit": "token_slots",
                "sizing_inputs": {
                    field: _integer(getattr(pool, field, None))
                    for field in SIZING_FIELDS
                },
                "pin_memory_configured": getattr(pool, "pin_memory", None),
                "allocator_class": (
                    type(pool.allocator).__name__
                    if getattr(pool, "allocator", None) is not None
                    else None
                ),
                "allocator_ownership_status": "unknown",
                "metadata_bytes": None,
                "metadata_status": "unknown",
                "tensors": [],
                "issues": [],
            }
            if type(pool).__name__ not in SUPPORTED_POOL_KINDS:
                result["issues"].append("unsupported_pool_kind")
            intervals = []
            capacity = logical = free = None
            lock = getattr(pool, "lock", None)
            acquired = lock is not None and lock.acquire(blocking=False)
            if acquired:
                try:
                    capacity = _integer(getattr(pool, "size", None))
                    logical = _integer(getattr(pool, "logical_size", None))
                    free = _integer(pool.available_size())
                except Exception:
                    result["issues"].append("capacity_unavailable")
                finally:
                    lock.release()
            else:
                result["issues"].append("capacity_lock_unavailable")
            valid = (
                capacity is not None
                and logical is not None
                and free is not None
                and capacity >= 0
                and 0 <= free <= logical
            )
            result["logical_capacity"] = logical
            result["logical_free"] = free if valid else None
            result["logical_used"] = logical - free if valid else None
            result["physical_slot_capacity"] = capacity
            # DCP logical indices can share physical rows. Free-list length
            # alone does not prove which physical rows are occupied.
            physical_known = valid and capacity == logical
            result["physical_slot_free"] = free if physical_known else None
            result["physical_slot_used"] = logical - free if physical_known else None
            result["capacity_status"] = "observed" if physical_known else "unknown"
            if not physical_known:
                result["issues"].append("physical_occupancy_unknown")
            for field in DATA_FIELDS:
                value = getattr(pool, field, None)
                if value is None:
                    continue
                tensors = value if isinstance(value, (list, tuple)) else (value,)
                if len(tensors) > MAX_TENSORS - tensor_count:
                    result["issues"].append("tensor_limit")
                for index, tensor in enumerate(tensors[: MAX_TENSORS - tensor_count]):
                    tensor_count += 1
                    if (
                        not isinstance(tensor, torch.Tensor)
                        or tensor.device.type != "cpu"
                    ):
                        result["issues"].append("non_cpu_or_unknown_tensor")
                        continue
                    try:
                        storage = tensor.untyped_storage()
                        references.extend((tensor, storage))
                        start, size = storage.data_ptr(), storage.nbytes()
                        key = (start, size)
                        if key not in storages:
                            backing_id = f"backing_{len(storages)}"
                            storages[key] = backing_id
                            report["backings"].append(
                                {"id": backing_id, "storage_bytes": size}
                            )
                        offset = tensor.storage_offset() * tensor.element_size()
                        extent = (
                            0
                            if tensor.numel() == 0
                            else (
                                1
                                + sum(
                                    (dim - 1) * stride
                                    for dim, stride in zip(
                                        tensor.shape, tensor.stride()
                                    )
                                )
                            )
                            * tensor.element_size()
                        )
                        if (
                            any(stride < 0 for stride in tensor.stride())
                            or offset + extent > size
                        ):
                            raise ValueError("unsupported span")
                        result["tensors"].append(
                            {
                                "field": field,
                                "index": index,
                                "backing_id": storages[key],
                                "storage_bytes": size,
                                "byte_offset": offset,
                                "byte_span": extent,
                                "tensor_bytes": tensor.numel() * tensor.element_size(),
                                "contiguous": tensor.is_contiguous(),
                                "torch_is_pinned": tensor.is_pinned(),
                            }
                        )
                        intervals.append((start, start + size))
                    except Exception:
                        result["issues"].append("tensor_storage_unavailable")
            # The DSv4 anchor explicitly owns no KV storage. Other missing
            # buffers are unknown, never an invented zero-byte allocation.
            if not result["tensors"] and type(pool).__name__ != "LogicalHostPool":
                result["issues"].append("data_backing_unavailable")
            result["issues"] = sorted(set(result["issues"]))
            result["status"] = "complete" if not result["issues"] else "incomplete"
            result["observed_unique_backing_bytes"] = _union_bytes(intervals)
            result["unique_backing_bytes"] = (
                result["observed_unique_backing_bytes"]
                if not result["issues"]
                else None
            )
            group_complete &= not result["issues"]
            group_intervals.extend(intervals)
            group_result["entries"].append(result)
        group_result["status"] = "complete" if group_complete else "incomplete"
        group_result["observed_unique_backing_bytes"] = _union_bytes(group_intervals)
        group_result["unique_backing_bytes"] = (
            group_result["observed_unique_backing_bytes"] if group_complete else None
        )
        group_result["logical_capacity_sum"] = (
            sum(entry["logical_capacity"] for entry in group_result["entries"])
            if all(
                entry["logical_capacity"] is not None
                for entry in group_result["entries"]
            )
            else None
        )
        all_intervals.extend(group_intervals)
        complete &= group_complete
        report["groups"].append(group_result)
    # Distinct torch storage wrappers may still overlap the same allocation
    # (e.g. frombuffer). Expose relative overlap coordinates, never addresses.
    components = []
    for start, stop in sorted(set(all_intervals)):
        if components and start < components[-1][1]:
            components[-1][1] = max(components[-1][1], stop)
        else:
            components.append([start, stop])
    report["allocations"] = [
        {"id": f"allocation_{index}", "bytes": stop - start}
        for index, (start, stop) in enumerate(components)
    ]
    component_index = 0
    backing_rows = {row["id"]: row for row in report["backings"]}
    for (start, size), backing_id in sorted(storages.items()):
        while (
            component_index + 1 < len(components)
            and start >= components[component_index][1]
        ):
            component_index += 1
        if components:
            backing_rows[backing_id].update(
                allocation_id=f"allocation_{component_index}",
                allocation_byte_offset=start - components[component_index][0],
            )
    report["status"] = (
        "not_present"
        if not report["groups"]
        else "complete"
        if complete
        else "incomplete"
    )
    report["observed_unique_backing_bytes"] = _union_bytes(all_intervals)
    report["unique_backing_bytes"] = (
        report["observed_unique_backing_bytes"] if complete else None
    )
    report["logical_capacity_sum"] = (
        sum(group["logical_capacity_sum"] for group in report["groups"])
        if complete
        and all(group["logical_capacity_sum"] is not None for group in report["groups"])
        else None
    )
    report["capture_finished_unix_ms"] = time.time_ns() // 1_000_000
    return report
