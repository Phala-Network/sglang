"""Bounded, on-demand host KV tensor accounting for scheduler server_info.

Storage addresses are used only inside one snapshot to union aliases. Neither
addresses nor tensor contents are serialized. This is not process RSS or a
measurement of allocator overhead, metadata, registration, or OS ownership.
"""

import hashlib
import json
import os
import re
import time
from numbers import Integral
from types import SimpleNamespace

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


def _capacity_snapshot(pool):
    """One nonblocking capacity read; state sidecars have no free-list API."""
    snapshot = dict(capacity=None, logical=None, free=None, geometry={}, issues=[])
    lock = getattr(pool, "lock", None)
    if lock is None or not lock.acquire(blocking=False):
        snapshot["issues"].append("capacity_lock_unavailable")
        return snapshot
    try:
        snapshot["capacity"] = _integer(getattr(pool, "size", None))
        snapshot["logical"] = _integer(getattr(pool, "logical_size", None))
        snapshot["geometry"] = {
            field: _integer(getattr(pool, field, None))
            for field in ("page_size", "num_host_pages", "swa_page_size")
        }
        if type(pool).__name__ != "DeepSeekV4StateHostPool":
            snapshot["free"] = _integer(pool.available_size())
    except Exception:
        snapshot["issues"].append("capacity_unavailable")
    finally:
        lock.release()
    return snapshot


def _state_index_sources(scheduler, group_name, group):
    """Read the actual transfer declarations, never infer ownership by name."""
    if group_name == "hicache":
        owner = getattr(scheduler, "tree_cache", None)
        same_group = getattr(owner, "host_pool_group", None) is group
        specs = getattr(owner, "sidecar_pool_specs", None)
    elif group_name == "decode_writeback":
        owner = getattr(scheduler, "decode_offload_manager", None)
        same_group = getattr(owner, "decode_host_mem_pool", None) is group
        specs = getattr(owner, "sidecar_specs", None)
    else:
        return {}
    if (
        not same_group
        or not isinstance(specs, (list, tuple))
        or len(specs) > MAX_ENTRIES
    ):
        return {}
    sources = {}
    for spec in specs:
        name = getattr(spec, "pool_name", None)
        name = getattr(name, "value", name)
        source = getattr(spec, "indices_from_pool", None)
        source = getattr(source, "value", source)
        if isinstance(name, str):
            sources[name] = source if name not in sources else None
    return sources


def _state_occupancy(name, snapshot, sources, named_pools, snapshots):
    source = sources.get(name)
    candidates = named_pools.get(source, []) if isinstance(source, str) else []
    if source != "swa" or len(candidates) != 1:
        return None, "occupancy_owner_unavailable"
    owner = candidates[0]
    observed = snapshots[id(owner)]
    if type(owner).__name__ != "DeepSeekV4PagedHostPool" or observed["issues"]:
        return None, "occupancy_owner_unavailable"
    capacity, logical, free = (observed[key] for key in ("capacity", "logical", "free"))
    geometry, own_geometry = observed["geometry"], snapshot["geometry"]
    page, pages = geometry.get("page_size"), geometry.get("num_host_pages")
    if (
        snapshot["issues"]
        or capacity is None
        or logical is None
        or free is None
        or capacity != logical
        or not 0 <= free <= logical
        or page is None
        or page <= 0
        or pages is None
        or pages <= 0
        or capacity != page * pages
        or snapshot["capacity"] != capacity
        or snapshot["logical"] != logical
        or own_geometry.get("page_size") != page
        or own_geometry.get("swa_page_size") != page
        or own_geometry.get("num_host_pages") != pages
        or free % page != 0
    ):
        return None, "occupancy_owner_geometry_mismatch"
    return free, None


def _seed_schema(scheduler):
    """Read retained D startup selectors; never arm capture or manufacture keys."""
    result = {
        "schema": "sglang.shared-cache-seed-schema.v1",
        "status": "not_present",
        "scope": "startup_selectors_not_backup_operation",
        "page_range": None,
        "page_range_status": "operation_dependent",
    }
    manager = getattr(scheduler, "decode_offload_manager", None)
    if manager is None or not getattr(manager, "is_dsv4", False):
        return result
    try:
        cc = manager.cache_controller
        cfg = cc.storage_config
        schema = manager.decode_host_mem_pool.storage_schema
        expected = {
            "version",
            "revision",
            "layout",
            "unified",
            "uniform_fp8",
            "layers",
            "layer_range",
            "topology",
            "cp_rank",
            "pools",
        }
        if type(schema) is not dict or set(schema) != expected:
            raise ValueError("unsupported schema")
        if (
            schema["version"] != 1
            or type(schema["unified"]) is not bool
            or type(schema["uniform_fp8"]) is not bool
        ):
            raise ValueError("invalid schema flags")
        revision = schema["revision"]
        if revision is not None and (
            not isinstance(revision, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}", revision)
            or ".." in revision.split("/")
        ):
            raise ValueError("unsupported revision")
        if not isinstance(schema["layout"], str) or not re.fullmatch(
            r"[A-Za-z0-9._-]{1,64}", schema["layout"]
        ):
            raise ValueError("invalid layout")
        for key, width in (("layer_range", 2), ("topology", 3)):
            value = schema[key]
            if (
                type(value) is not list
                or len(value) != width
                or any(type(x) is not int or not 0 <= x <= 1_000_000 for x in value)
            ):
                raise ValueError("invalid geometry")
        if type(schema["cp_rank"]) is not int or not 0 <= schema["cp_rank"] <= 1024:
            raise ValueError("invalid rank")
        layers = schema["layers"]
        if type(layers) is not list or not 1 <= len(layers) <= 1024:
            raise ValueError("invalid layers")
        for layer in layers:
            if (
                type(layer) is not list
                or len(layer) != 2
                or any(type(x) is not int or not 0 <= x <= 1_000_000 for x in layer)
            ):
                raise ValueError("invalid layer mapping")
        pools = schema["pools"]
        if type(pools) is not list or not 1 <= len(pools) <= MAX_ENTRIES:
            raise ValueError("invalid pools")
        for pool in pools:
            if type(pool) is not list or len(pool) != 7:
                raise ValueError("invalid pool row")
            for index in (0, 2, 4):
                if not isinstance(pool[index], str) or not re.fullmatch(
                    r"[A-Za-z0-9._-]{1,96}", pool[index]
                ):
                    raise ValueError("invalid pool label")
            for index in (1, 3, 5, 6):
                value = pool[index]
                if value is None and index in (5, 6):
                    continue
                if type(value) is not int or not 0 <= value <= 2**63 - 1:
                    raise ValueError("invalid pool geometry")
        raw = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
        if len(raw) > 64 * 1024:
            raise ValueError("schema too large")
        if cfg.tp_size != 1 or cfg.pp_size != 1 or cfg.tp_rank != 0:
            raise ValueError("unsupported seed topology")
        backend_tag = cfg.extra_config["extra_backend_tag"]
        if not isinstance(backend_tag, str) or not re.fullmatch(
            r"dsv4-v1-[0-9a-f]{64}", backend_tag
        ):
            raise ValueError("invalid derived tag")
        store = cc.storage_backend
        names = list(store.registered_pools)
        if not 1 <= len(names) <= MAX_ENTRIES:
            raise ValueError("invalid registered set")
        components = []
        registered = []
        for name in names:
            if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", str(name)):
                raise ValueError("invalid component name")
            transfer = SimpleNamespace(name=name)
            if not cc.should_backup(transfer):
                raise ValueError("registered pool not backed up")
            # Source formatter only reads pool/type metadata and builds suffixes.
            # An empty key list yields no object keys and performs no store I/O.
            keys, multiplier = store._get_hybrid_page_component_keys([], transfer)
            if keys or type(multiplier) is not int or not 1 <= multiplier <= 64:
                raise ValueError("invalid component multiplier")
            components.extend(f"{name}:{index}" for index in range(multiplier))
            registered.append({"pool": str(name), "component_count": multiplier})
        if len(components) > 64 or len(components) != len(set(components)):
            raise ValueError("invalid component set")
        result.update(
            status="complete",
            storage_schema=json.loads(raw),
            kv_schema=hashlib.sha256(raw).hexdigest(),
            model_revision=revision,
            backend_tag=backend_tag,
            registered_components=sorted(registered, key=lambda item: item["pool"]),
            required_components=sorted(components),
            rank=cfg.tp_rank,
        )
    except Exception:
        # Do not include arbitrary values or exception text in a public readback.
        result.update(status="incomplete", error="startup_schema_unavailable")
    return result


def _scheduler_rank(scheduler, rank):
    value = getattr(scheduler, rank, None)
    if value is None:
        process_state = getattr(scheduler, "ps", None)
        value = getattr(process_state, rank, None)
    return _integer(value)


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
                rank: _scheduler_rank(scheduler, rank)
                for rank in ("tp_rank", "pp_rank", "dp_rank")
            },
        },
        "parallel_state": {
            "tp_rank": _scheduler_rank(scheduler, "tp_rank"),
            "pp_rank": _scheduler_rank(scheduler, "pp_rank"),
            "dp_rank": _scheduler_rank(scheduler, "dp_rank"),
            "tp_size": _integer(
                getattr(getattr(scheduler, "ps", None), "tp_size", None)
            ),
            "pp_size": _integer(
                getattr(getattr(scheduler, "ps", None), "pp_size", None)
            ),
            "dp_size": _integer(
                getattr(getattr(scheduler, "ps", None), "dp_size", None)
            ),
            "dp_disabled": _scheduler_rank(scheduler, "dp_rank") is None
            or _integer(getattr(getattr(scheduler, "ps", None), "dp_size", None))
            in (None, 1),
            "source": "scheduler.ps",
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
        # Reuse the same owner's snapshot for its entry and every sidecar,
        # independent of entry order. Never reacquire an owner or nest locks.
        snapshots = {}
        for _, pool in entries:
            if id(pool) not in snapshots:
                snapshots[id(pool)] = _capacity_snapshot(pool)
        named_pools = {}
        for name, pool in entries:
            named_pools.setdefault(name, []).append(pool)
        sources = _state_index_sources(scheduler, group_name, group)
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
            snapshot = snapshots[id(pool)]
            capacity, logical, free = (
                snapshot[key] for key in ("capacity", "logical", "free")
            )
            result["issues"].extend(snapshot["issues"])
            if type(pool).__name__ == "DeepSeekV4StateHostPool":
                result["occupancy_source"] = sources.get(name)
                result["occupancy_semantics"] = "shared_transfer_indices"
                free, issue = _state_occupancy(
                    name, snapshot, sources, named_pools, snapshots
                )
                if issue:
                    result["issues"].append(issue)
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
            backing_issue_start = len(result["issues"])
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
            result["backing_status"] = (
                "observed"
                if len(result["issues"]) == backing_issue_start
                and type(pool).__name__ in SUPPORTED_POOL_KINDS
                else "unknown"
            )
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
    report["seed_schema"] = _seed_schema(scheduler)
    report["capture_finished_unix_ms"] = time.time_ns() // 1_000_000
    return report
