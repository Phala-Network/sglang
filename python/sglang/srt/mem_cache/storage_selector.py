"""Read-only effective HiCache storage selector metadata.

The selector is deliberately derived from the constructed host entries rather
than from requested flags.  This makes startup evidence describe the bytes
that P actually allocated and keeps private draft sidecars out of the shared
target schema.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from sglang.srt.mem_cache.hicache_storage import PoolName, SidecarPoolSpec


def _dtype_name(value: Any) -> str:
    return str(value) if value is not None else "unknown"


def pool_descriptor(name: PoolName, entry: Any) -> dict[str, Any]:
    host = entry.host_pool
    state_page_bytes = int(getattr(host, "state_page_bytes", 0) or 0)
    item_bytes = getattr(host, "item_bytes", None)
    if item_bytes is None:
        item_bytes = state_page_bytes or getattr(host, "size_per_token", 0)
    return {
        "pool_name": str(name),
        "layer_count": int(
            getattr(host, "target_layer_num", getattr(host, "layer_num", 0))
        ),
        "page_size": int(getattr(host, "page_size", 0)),
        "layout": str(getattr(host, "layout", "unknown")),
        "item_bytes": int(item_bytes or 0),
        "dtype": _dtype_name(getattr(host, "dtype", None)),
        "state_page_bytes": state_page_bytes,
        "start_layer": int(getattr(host, "start_layer", 0)),
        "end_layer": int(getattr(host, "end_layer", 0)),
        "layer_mapping": [
            [int(layer), int(mapped)]
            for layer, mapped in sorted(
                (getattr(entry, "layer_mapping", None) or {}).items()
            )
        ],
    }


@dataclass(frozen=True)
class EffectiveStorageSelector:
    """Canonical startup observation for one rank's target and sidecar pools."""

    target_schema: Mapping[str, Any]
    schema_digest: str
    backend: Optional[str]
    rank: Mapping[str, int]
    components: tuple[Mapping[str, Any], ...]
    backend_tag: Optional[str] = None
    backend_namespace: Optional[str] = None
    registered_components: tuple[Mapping[str, Any], ...] = ()
    required_components: tuple[str, ...] = ()
    model_revision: Optional[str] = None
    draft_swa: Optional[Mapping[str, Any]] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_schema": dict(self.target_schema),
            "schema_digest": self.schema_digest,
            "backend": self.backend,
            "backend_tag": self.backend_tag,
            "backend_namespace": self.backend_namespace,
            "rank": dict(self.rank),
            "components": [dict(item) for item in self.components],
            "registered_components": [dict(item) for item in self.registered_components],
            "required_components": list(self.required_components),
            "model_revision": self.model_revision,
            "draft_swa": None if self.draft_swa is None else dict(self.draft_swa),
        }


def build_effective_storage_selector(
    *,
    host_pool_group: Any,
    sidecar_pool_specs: list[SidecarPoolSpec],
    controller: Any = None,
) -> EffectiveStorageSelector:
    """Build target-only schema plus an independent DRAFT_SWA descriptor."""

    draft_spec = next(
        (spec for spec in sidecar_pool_specs if spec.pool_name == PoolName.DRAFT_SWA),
        None,
    )
    if controller is None or not hasattr(controller, "tp_rank") or not hasattr(controller, "pp_rank"):
        raise RuntimeError("effective shared selector requires controller TP/PP rank identity")
    if not hasattr(controller, "get_attn_cp_rank_and_size"):
        raise RuntimeError("effective shared selector requires controller attention-CP identity")
    rank = {
        "tp_rank": int(controller.tp_rank),
        "pp_rank": int(controller.pp_rank),
        "attn_cp_rank": 0,
        "attn_cp_size": 1,
    }
    rank["attn_cp_rank"], rank["attn_cp_size"] = map(
        int, controller.get_attn_cp_rank_and_size()
    )
    backend = getattr(controller, "storage_backend_type", None)
    backend_namespace = None
    storage_config = getattr(controller, "storage_config", None)
    extra_config = getattr(storage_config, "extra_config", None)
    if isinstance(extra_config, Mapping):
        value = extra_config.get("extra_backend_tag")
        if isinstance(value, str) and value:
            backend_namespace = value
    # DSV4 assembly publishes the exact canonical schema.  Never reconstruct a
    # digest from an observation projection: that would permit P/D drift.
    schema = getattr(host_pool_group, "storage_schema", None)
    if schema is None:
        raise RuntimeError(
            "effective shared selector requires the canonical storage_schema "
            "published by the constructed DSV4 host pool"
        )
    components = tuple(
        pool_descriptor(entry.name, entry)
        for entry in host_pool_group.entries
        if entry.name != PoolName.DRAFT_SWA
        and not str(entry.name).startswith("draft")
    )
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    # These identities are derived from the constructed host entries and the
    # canonical schema.  They are never copied from a seed or requested config.
    component_names = sorted(
        str(entry.name)
        for entry in host_pool_group.entries
        if entry.name != PoolName.DRAFT_SWA
        and not str(entry.name).startswith("draft")
        and (str(entry.name) == str(PoolName.SWA)
             or str(entry.name).startswith("deepseek_v4_"))
    )
    registered_components = tuple(
        {"pool": name, "component_count": 1} for name in component_names
    )
    required_components = tuple(f"{name}:0" for name in component_names)
    backend_tag = "dsv4-v1-" + hashlib.sha256(encoded).hexdigest()
    draft_desc = None
    if draft_spec is not None:
        entry = host_pool_group.entry_map.get(PoolName.DRAFT_SWA)
        if entry is not None:
            draft_desc = {
                **pool_descriptor(PoolName.DRAFT_SWA, entry),
                "indices_from_pool": str(draft_spec.indices_from_pool),
                "required_for_restore": bool(draft_spec.required_for_restore),
                "state": "registered",
            }
        else:
            draft_desc = {
                "pool_name": str(PoolName.DRAFT_SWA),
                "indices_from_pool": str(draft_spec.indices_from_pool),
                "required_for_restore": bool(draft_spec.required_for_restore),
                "state": "declared_without_host_entry",
            }
    return EffectiveStorageSelector(
        target_schema=schema,
        schema_digest=hashlib.sha256(encoded).hexdigest(),
        backend=backend,
        rank=rank,
        components=components,
        backend_tag=backend_tag,
        backend_namespace=backend_namespace,
        registered_components=registered_components,
        required_components=required_components,
        model_revision=schema.get("revision"),
        draft_swa=draft_desc,
    )



