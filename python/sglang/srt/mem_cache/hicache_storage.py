from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional, Set

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host import HostKVCache

logger = logging.getLogger(__name__)

# Max pages per batched storage IO call.
STORAGE_BATCH_SIZE = 128


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    attn_cp_rank: int
    attn_cp_size: int
    is_mla_model: bool
    enable_storage_metrics: bool
    is_page_first_layout: bool
    model_name: Optional[str]
    tp_lcm_size: Optional[int] = None
    should_split_heads: bool = False
    extra_config: Optional[dict] = None


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = None
    extra_info: Optional[dict] = None


@dataclass(frozen=True)
class PrefetchTimeoutConfig:
    """Knobs for the linear prefetch-timeout policy used by HiCache."""

    base: float = 2.0  # seconds, fixed overhead unrelated to token count
    per_ki_token: float = 0.1  # seconds per 1024 tokens
    max: float = 30.0  # seconds, upper bound for the linear timeout


class PoolName(str, Enum):
    """Well-known pool names used as PoolTransfer/PoolEntry identifiers."""

    KV = "kv"
    MAMBA = "mamba"
    SWA = "swa"
    INDEXER = "indexer"
    # TODO(hzh0425): Current DeepSeek V4 pool naming is verbose; will be normalized to
    # 'COMPRESSED_KV / COMPRESSED_INDEXER / COMPRESSED_STATE' in the next PR.
    DEEPSEEK_V4_C4 = "deepseek_v4_c4"
    DEEPSEEK_V4_C4_INDEXER = "deepseek_v4_c4_indexer"
    # FP4 indexer splits the indexer cache into separate payload/scale buffers,
    # so it needs a second pool alongside DEEPSEEK_V4_C4_INDEXER.
    DEEPSEEK_V4_C4_INDEXER_SCALE = "deepseek_v4_c4_indexer_scale"
    DEEPSEEK_V4_C128 = "deepseek_v4_c128"
    DEEPSEEK_V4_C4_STATE = "deepseek_v4_c4_state"
    DEEPSEEK_V4_C4_INDEXER_STATE = "deepseek_v4_c4_indexer_state"
    DEEPSEEK_V4_C128_STATE = "deepseek_v4_c128_state"

    # Draft KV pool
    DRAFT = "draft"
    DRAFT_INDEXER = "draft_indexer"
    DRAFT_SWA = "draft_swa"

    def __str__(self) -> str:
        return self.value


class PoolHitPolicy(str, Enum):
    """Hit policy for batch_exists_v2 per-pool prefix matching.

    ALL_PAGES      : every page in [0, kv_hit) must exist (e.g. DSA).
    TRAILING_PAGES : only the last N pages must exist (e.g. Mamba/SWA states).
    """

    ALL_PAGES = "all_pages"
    TRAILING_PAGES = "trailing_pages"


@dataclass
class PoolTransfer:
    """Unified per-pool transfer descriptor for batch v2 interface.

    device<->host path : host_indices + device_indices
    host<->storage path: host_indices + keys
    nodes_to_load      : evicted nodes this transfer covers
    """

    name: PoolName
    host_indices: Optional[torch.Tensor] = None
    device_indices: Optional[torch.Tensor] = None
    keys: Optional[List[str]] = None
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES
    nodes_to_load: Optional[List[Any]] = None
    indices_from_pool: Optional[PoolName] = None


@dataclass(frozen=True)
class SidecarPoolSpec:
    """Pool whose transfer indices are reused from one real source pool."""

    pool_name: PoolName
    indices_from_pool: PoolName
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES


@dataclass
class PoolTransferResult:
    """Tracks how many pages were successfully processed per pool."""

    kv_hit_pages: int
    extra_pool_hit_pages: dict[str, int]

    # Pools with TRAILING_PAGES (SWA, Mamba state) only hold a window that ends on an
    # offloaded node boundary, so 5 can be restorable while 4 and 3 are not.
    # Each rank owns its own shard and may hold a different set, so reducing a
    # per-rank maximum would pick a length that is illegal on another rank; the
    # caller intersects these sets instead.
    restorable_prefix_pages: Optional[List[int]] = None

    @classmethod
    def empty(cls) -> PoolTransferResult:
        return cls(0, {})

    def update_kv_hit_pages(self, kv_hit_pages: int) -> None:
        """Accumulate kv_hit_pages across batches (max = last successful batch)."""
        self.kv_hit_pages = max(self.kv_hit_pages, kv_hit_pages)

    def update_extra_pool_hit_pages(self, results: dict[str, int]) -> None:
        """Record actual load/write success counts per extra pool.

        Every extra pool contributes a prefix that must be contiguous from the
        start, so count the leading run of successes
        """
        self.extra_pool_hit_pages.update(results)


def count_pool_hits(results: dict[str, List[bool]]) -> dict[str, int]:
    return {
        name: (rs.index(False) if False in rs else len(rs))
        for name, rs in results.items()
    }


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, the page size of storage backend does not have to be the same as the same as host memory pool
    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self.mem_pool_host = mem_pool_host

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        if not hasattr(self, "registered_pools"):
            self.registered_pools = {}
        self.registered_pools[host_pool_name] = host_pool

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        """Check which cache pages exist in storage, respecting per-pool hit policies.

        Longest-prefix semantics
        Extra-pool hit policies (``PoolTransfer.hit_policy``)
        ------------------------------------------------------
        Each ``PoolTransfer`` in ``pool_transfers`` describes a secondary
        cache pool (e.g. Mamba SSM states) that must be co-present with the
        KV pages. The usable prefix ends at the greatest stop point that all
        pools can restore. Trailing-page pools may have gaps in their valid
        stop points, so taking the minimum of their maxima is not sufficient.

        - ``"all_pages"`` (default):  every page in [0, kv_hit) must exist
          for this pool.  Used for pools that are required for every token
          in the prefix (e.g. DeepSeek DSA pool).

        - ``"trailing_pages"``:  only the *last* ``len(transfer.keys)`` pages
          of the KV prefix need to exist.  Used for pools whose data covers
          only the tail of a prefix (e.g. Mamba/SWA Pool).

        Returns
        -------
        PoolTransferResult
            ``kv_hit_pages`` = length of the usable KV prefix.
            ``extra_pool_hit_pages`` maps each pool name to the number of pages
            that were found.
        """
        raise NotImplementedError()

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        """Read data from storage into host memory for each PoolTransfer.

        Returns a dict mapping pool name to a per-entry success list.
        """
        raise NotImplementedError()

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        """Write data from host memory to storage for each PoolTransfer.

        Returns a dict mapping pool name to a per-entry success list.
        """
        raise NotImplementedError()

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of booleans indicating success for each key.
        """
        pass

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Store multiple key-value pairs.
        Returns a list of booleans indicating success for each key.
        """
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        pass

    # TODO: Use a finer-grained return type (e.g., List[bool])
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None


class MetadataCache:
    def __init__(self, ttl_seconds: float):
        self.ttl_seconds = ttl_seconds
        # key -> monotonic timestamp
        self.cache: dict[str, float] = {}
        self.lock = threading.Lock()

    def add(self, key: str):
        with self.lock:
            if key not in self.cache:
                self.cache[key] = time.monotonic()

    def remove(self, key: str):
        with self.lock:
            self.cache.pop(key, None)

    def contains(self, key: str) -> bool:
        with self.lock:
            if key not in self.cache:
                return False
            if self.ttl_seconds == -1.0:
                return True
            if time.monotonic() - self.cache[key] > self.ttl_seconds:
                del self.cache[key]
                return False
            return True

    def clear(self):
        with self.lock:
            self.cache.clear()


class HiCacheFile(HiCacheStorage):
    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get() or file_path

        tp_rank, tp_size, pp_rank, pp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.pp_rank,
            storage_config.pp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        attn_cp_rank = storage_config.attn_cp_rank
        attn_cp_size = storage_config.attn_cp_size
        if is_mla_model and (not model_name or not model_name.strip()):
            raise ValueError(
                "MLA HiCacheFile requires a non-empty model_name for namespace ownership"
            )
        model_name = "-".join(model_name.split("/")) if model_name else ""
        enable_pp = pp_size > 1
        self.config_suffix = f"_{model_name}"
        if not is_mla_model:
            self.config_suffix += f"_{tp_rank}_{tp_size}"
        if enable_pp:
            self.config_suffix += f"_{pp_size}_{pp_rank}"
        # Under NSA context parallel each CP rank holds a disjoint slice of every
        # page, so give each rank its own file key to avoid a cross-rank write race.
        if attn_cp_size > 1:
            self.config_suffix += f"_cp{attn_cp_rank}_{attn_cp_size}"

        self._tp_rank = tp_rank
        self._tp_size = tp_size
        self._is_mla_model = is_mla_model
        # MLA KV is replicated and keeps the historical TP-independent suffix.
        # Kimi-K3 Mamba/KDA state is TP-sharded, so it needs a rank-qualified
        # namespace even though both pools use this HiCacheFile instance.
        self._mamba_config_suffix = (
            f"{self.config_suffix}_mamba_tp{tp_rank}_{tp_size}"
            if is_mla_model
            else self.config_suffix
        )

        owned_file_suffixes = []
        if not is_mla_model or tp_rank == 0:
            owned_file_suffixes.append(self.config_suffix)
        if is_mla_model:
            owned_file_suffixes.append(self._mamba_config_suffix)
        self._owned_file_suffixes = tuple(owned_file_suffixes)
        self._owned_temp_prefixes = tuple(
            ".sglang-" + hashlib.sha256(suffix.encode()).hexdigest() + "-"
            for suffix in self._owned_file_suffixes
        )

        # Validate the complete budget contract before creating a directory or
        # inspecting any files. Invalid topology/configuration must have no
        # filesystem side effects.
        from sglang.srt.mem_cache.storage.file.lru_file_evictor import (
            LRUFileEvictor,
            _parse_size_to_bytes,
        )

        extra_config = dict(storage_config.extra_config or {})
        total_max_size = _parse_size_to_bytes(
            extra_config.get("max_size")
            if extra_config.get("max_size") is not None
            else envs.SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE.get()
        )
        mamba_max_size_raw = extra_config.get("mamba_max_size")
        mamba_max_size = _parse_size_to_bytes(mamba_max_size_raw)

        shared_extra_config = dict(extra_config)
        mamba_extra_config = dict(extra_config)
        # With an MLA File backend, max_size is the aggregate budget for one
        # DP1/PP1/CP1 TP-group namespace. mamba_max_size allocates part of it to
        # all TP-sharded Mamba writers; the remainder belongs to replicated KV
        # and legacy files. A deployment with another instance sharing the same
        # directory must allocate a separate outer namespace/budget.
        self._mamba_budget_required = (
            is_mla_model and total_max_size > 0 and mamba_max_size_raw is None
        )
        if mamba_max_size_raw is not None:
            if not is_mla_model:
                raise ValueError("mamba_max_size is only valid for an MLA File backend")
            if pp_size != 1 or attn_cp_size != 1:
                raise ValueError(
                    "MLA File mamba_max_size currently requires PP1 and CP1; "
                    "allocate independent outer budgets for other topologies"
                )
            if total_max_size <= 0:
                raise ValueError("mamba_max_size requires a positive max_size")
            if not 0 < mamba_max_size < total_max_size:
                raise ValueError("mamba_max_size must be positive and smaller than max_size")
            if tp_size <= 0 or mamba_max_size // tp_size == 0:
                raise ValueError("mamba_max_size must allocate at least one byte per TP rank")
            shared_extra_config["max_size"] = total_max_size - mamba_max_size
            mamba_extra_config["max_size"] = mamba_max_size // tp_size
            logger.info(
                "HiCacheFile MLA TP-group budget: aggregate=%s B, "
                "replicated_and_legacy=%s B, mamba_aggregate=%s B, "
                "mamba_per_rank=%s B, tp_size=%s",
                total_max_size,
                shared_extra_config["max_size"],
                mamba_max_size,
                mamba_extra_config["max_size"],
                tp_size,
            )
        else:
            # Never let each Mamba rank inherit the full aggregate max_size.
            # If a cap is configured, set() rejects Mamba writes until an
            # explicit aggregate allocation is supplied.
            mamba_extra_config["max_size"] = 0

        if not os.path.exists(self.file_path) and tp_rank == 0 and attn_cp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

        # A temp file may belong to a live same-rank writer. Without a process
        # identity/lock proving it stale, never unlink it automatically. Refuse
        # startup so a controller can clean it only after stopping all writers.
        self._assert_no_unaccounted_temp_files()

        # Metadata cache positive lookup toggle & TTL
        enable_cache_raw = None
        if storage_config.extra_config:
            enable_cache_raw = storage_config.extra_config.get("enable_metadata_cache")
        if enable_cache_raw is None:
            enable_cache_raw = (
                envs.SGLANG_HICACHE_FILE_BACKEND_ENABLE_METADATA_CACHE.get()
            )

        self.enable_metadata_cache = bool(enable_cache_raw)

        if self.enable_metadata_cache:
            ttl_raw = None
            if storage_config.extra_config:
                ttl_raw = storage_config.extra_config.get("metadata_ttl")
            if ttl_raw is None:
                ttl_raw = envs.SGLANG_HICACHE_FILE_BACKEND_METADATA_TTL.get()

            self.metadata_ttl = float(ttl_raw) if ttl_raw is not None else 5.0
            self.metadata_cache = MetadataCache(self.metadata_ttl)
            self._scan_existing_files_to_metadata_cache()
        else:
            self.metadata_cache = None

        # All LRU / size accounting and disk eviction lives in the evictor so
        # this backend stays a thin raw-bytes store.
        self._evictor = LRUFileEvictor(
            self.file_path,
            self.config_suffix,
            tp_rank=tp_rank,
            is_mla_model=is_mla_model,
            extra_config=shared_extra_config,
            on_evict=(
                self.metadata_cache.remove if self.metadata_cache is not None else None
            ),
        )
        self._mamba_evictor = None
        if is_mla_model:
            self._mamba_evictor = LRUFileEvictor(
                self.file_path,
                self._mamba_config_suffix,
                tp_rank=tp_rank,
                is_mla_model=False,
                extra_config=mamba_extra_config,
                on_evict=(
                    self.metadata_cache.remove
                    if self.metadata_cache is not None
                    else None
                ),
            )

    def _assert_no_unaccounted_temp_files(self) -> None:
        try:
            names = os.listdir(self.file_path)
        except FileNotFoundError:
            return
        unaccounted = []
        for name in names:
            # Upstream's anonymous short names carry no namespace ownership.
            # Do not silently exclude their bytes or unlink a possibly live write.
            if re.fullmatch(r"\.[0-9a-f]{32}\.tmp", name):
                unaccounted.append(name)
                continue
            if name.startswith(self._owned_temp_prefixes) and name.endswith(".tmp"):
                unaccounted.append(name)
                continue
            marker = ".bin.tmp."
            if marker not in name:
                continue
            stem = name.split(marker, 1)[0]
            if stem.endswith(self._owned_file_suffixes):
                unaccounted.append(name)
        if unaccounted:
            raise RuntimeError(
                "HiCacheFile found unaccounted temporary files in this writer "
                "namespace; stop every writer and remove them before restart: "
                + ", ".join(sorted(unaccounted))
            )

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        if host_pool_name == PoolName.MAMBA and self._mamba_budget_required:
            raise ValueError(
                "MLA File Mamba storage requires extra_config.mamba_max_size "
                "when max_size is configured"
            )
        super().register_mem_host_pool_v2(host_pool, host_pool_name)

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def _get_component_key(
        self, key: str, component_name: Optional[str] = None
    ) -> str:
        if component_name is None or component_name in ("__default__", PoolName.KV):
            return self._get_suffixed_key(key)
        suffix = (
            self._mamba_config_suffix
            if component_name == PoolName.MAMBA
            else self.config_suffix
        )
        return f"{key}.{component_name}{suffix}"

    def _get_component_evictor(self, component_name: Optional[str] = None):
        if self._is_mla_model and component_name == PoolName.MAMBA:
            return self._mamba_evictor
        return self._evictor

    def _get_component_path(
        self, key: str, component_name: Optional[str] = None
    ) -> str:
        return os.path.join(
            self.file_path, f"{self._get_component_key(key, component_name)}.bin"
        )

    def _scan_existing_files_to_metadata_cache(self) -> None:
        try:
            names = os.listdir(self.file_path)
        except FileNotFoundError:
            return
        for fn in names:
            if not fn.endswith(".bin"):
                continue
            stem = fn[:-4]
            # Only files belonging to this rank/model.
            if stem.endswith((self.config_suffix, self._mamba_config_suffix)):
                self.metadata_cache.add(stem)

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
        component_name: Optional[str] = None,
    ) -> torch.Tensor | None:
        suffixed = self._get_component_key(key, component_name)
        tensor_path = os.path.join(self.file_path, f"{suffixed}.bin")
        evictor = self._get_component_evictor(component_name)
        try:
            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected:
                    raise IOError(f"Short read for {suffixed}")
            evictor.touch(suffixed, tensor_path)
            if self.metadata_cache is not None:
                self.metadata_cache.add(suffixed)
            return target_location
        except FileNotFoundError:
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
        component_name: Optional[str] = None,
    ) -> bool:
        suffixed = self._get_component_key(key, component_name)
        tensor_path = os.path.join(self.file_path, f"{suffixed}.bin")
        evictor = self._get_component_evictor(component_name)

        # Fast path: same key already on disk. Refresh recency and skip rewrite.
        if self.exists(key, component_name=component_name):
            logger.debug(f"Key {key} already exists. Skipped.")
            evictor.touch(suffixed, tensor_path)
            return True

        if component_name == PoolName.MAMBA and self._mamba_budget_required:
            logger.error(
                "HiCacheFile MLA Mamba writes require extra_config.mamba_max_size "
                "when max_size is configured; refusing an unbounded TP shard."
            )
            return False

        tmp_path = None
        reserved = False
        try:
            value_bytes = value.numel() * value.element_size()
            # Ask the evictor to admit + reserve disk space (evicting if needed).
            if not evictor.reserve(suffixed, value_bytes, key=key):
                return False
            reserved = True

            # Keep the upstream NAME_MAX fix while retaining namespace ownership
            # for fail-closed accounting of abandoned writes after a restart.
            namespace = (
                self._mamba_config_suffix
                if component_name == PoolName.MAMBA
                else self.config_suffix
            )
            owner = hashlib.sha256(namespace.encode()).hexdigest()
            tmp_path = os.path.join(
                self.file_path, f".sglang-{owner}-{uuid.uuid4().hex}.tmp"
            )
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tmp_path)
            os.replace(tmp_path, tensor_path)
            evictor.commit(suffixed)
            if self.metadata_cache is not None:
                self.metadata_cache.add(suffixed)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            # Roll back the reservation and clean up any half-written file.
            if reserved:
                evictor.abort(suffixed)
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str, component_name: Optional[str] = None) -> bool:
        key = self._get_component_key(key, component_name)
        if self.metadata_cache is not None and self.metadata_cache.contains(key):
            return True
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        if os.path.exists(tensor_path):
            if self.metadata_cache is not None:
                self.metadata_cache.add(key)
            return True
        return False

    def _collect_existing_component_keys(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
    ) -> Set[str]:
        target_files = {f"{self._get_component_key(key)}.bin" for key in keys}
        for transfer in pool_transfers or []:
            for key in keys:
                target_files.add(f"{self._get_component_key(key, transfer.name)}.bin")

        if self.metadata_cache is None:
            existing_files = set()
            with os.scandir(self.file_path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name in target_files:
                        existing_files.add(entry.name)
            return existing_files

        existing_files = set()
        for filename in target_files:
            stem = filename[:-4]
            if self.metadata_cache.contains(stem):
                existing_files.add(filename)
            else:
                path = os.path.join(self.file_path, filename)
                if os.path.exists(path):
                    self.metadata_cache.add(stem)
                    existing_files.add(filename)
        return existing_files

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        existing_files = self._collect_existing_component_keys(keys, pool_transfers)

        def has_component(page_idx: int, name: str) -> bool:
            return (
                f"{self._get_component_key(keys[page_idx], name)}.bin" in existing_files
            )

        # Longest contiguous KV prefix present in storage.
        kv_pages = next(
            (
                i
                for i in range(len(keys))
                if f"{self._get_component_key(keys[i])}.bin" not in existing_files
            ),
            len(keys),
        )

        hit_count: dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        # Trailing pools can have holes in their valid stop points: a longer
        # prefix being restorable does not imply that a shorter one is. Keep
        # the common stop points instead of taking the minimum of maxima.
        restorable = list(range(1, kv_pages + 1))

        for transfer in pool_transfers or []:
            if not restorable:
                break
            name = transfer.name
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next(
                    (i for i in range(kv_pages) if not has_component(i, name)), kv_pages
                )
                pool_restorable = list(range(1, boundary + 1))
            else:  # trailing_pages
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                pool_restorable = []
                consecutive = 0
                for prefix_len in range(1, kv_pages + 1):
                    if has_component(prefix_len - 1, name):
                        consecutive += 1
                    else:
                        consecutive = 0
                    if consecutive >= min(trailing, prefix_len):
                        pool_restorable.append(prefix_len)
                boundary = pool_restorable[-1] if pool_restorable else 0
            if boundary:
                hit_count[name] = boundary
            pool_restorable_set = set(pool_restorable)
            restorable = [p for p in restorable if p in pool_restorable_set]

        final_pages = restorable[-1] if restorable else 0
        return PoolTransferResult(final_pages, hit_count, restorable)

    def _read_page(self, pool_name: str, key: str, host_pool, page_offset: int) -> bool:
        """Read one page from storage into host_pool at page_offset."""
        data_page = self.get(
            key,
            host_pool.get_dummy_flat_data_page(),
            component_name=pool_name,
        )
        if data_page is None:
            return False
        host_pool.set_from_flat_data_page(page_offset, data_page)
        return True

    def _write_page(
        self, pool_name: str, key: str, host_pool, page_offset: int
    ) -> bool:
        """Write one page from host_pool at page_offset to storage as raw bytes."""
        data_page = host_pool.get_data_page(page_offset, flat=True)
        return self.set(key, data_page, component_name=pool_name)

    def _batch_io_v2(self, transfers: List[PoolTransfer], op_fn):
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.name]
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            expected = len(keys) * page_size
            host_indices = transfer.host_indices

            if host_indices is None or host_indices.numel() != expected:
                logger.error(
                    "%s indices length mismatch for %s: expected %s, got %s",
                    op_fn.__name__,
                    transfer.name,
                    expected,
                    host_indices.numel() if host_indices is not None else 0,
                )
                results[transfer.name] = [False] * len(keys)
                continue

            results[transfer.name] = [
                op_fn(transfer.name, key, host_pool, host_indices[i * page_size].item())
                for i, key in enumerate(keys)
            ]
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._read_page)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._write_page)

    def clear(self) -> bool:
        try:
            # This is intentionally rank-local. The scheduler request receiver
            # broadcasts the HTTP clear control request to every TP rank, so a
            # group reset calls each namespace owner without cross-rank deletes.
            for filename in os.listdir(self.file_path):
                if not filename.endswith(".bin"):
                    continue
                stem = filename[:-4]
                if not stem.endswith(self._owned_file_suffixes):
                    continue
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            self._evictor.clear()
            if self._mamba_evictor is not None:
                self._mamba_evictor.clear()
            if self.metadata_cache is not None:
                self.metadata_cache.clear()
            logger.info("Cleared all entries in HiCacheFile storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
            return False
