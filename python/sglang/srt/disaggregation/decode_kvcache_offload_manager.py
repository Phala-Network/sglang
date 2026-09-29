from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary as WeakKeyDict

import torch

from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheWriteSubmissionError,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_deepseek_v4_hicache_stack,
    build_kv_host_pool,
    deepseek_v4_sidecar_specs,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.shared_cache_diagnostics import shared_cache_diagnostics
from sglang.srt.mem_cache.storage_backend_config import (
    load_storage_backend_extra_config,
)
from sglang.srt.mem_cache.utils import storage_namespace_seed
from sglang.srt.runtime_context import (
    get_memory,
    get_schedule,
    get_serving,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class DecodeKVCacheOffloadManager:
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        tree_cache: BasePrefixCache,
        attn_cp_group=None,
        attn_tp_group=None,
        pp_group=None,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = get_schedule().page_size
        self.request_counter = 0
        self.tree_cache = tree_cache
        env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
        if env_stride is None or env_stride <= 0:
            self.offload_stride = self.page_size
        else:
            self.offload_stride = max(
                self.page_size, (env_stride // self.page_size) * self.page_size
            )
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        self.is_dsv4 = isinstance(kv_cache, DeepSeekV4TokenToKVPool)
        self.sidecar_specs = []
        if not isinstance(
            kv_cache, (MHATokenToKVPool, MLATokenToKVPool, DeepSeekV4TokenToKVPool)
        ):
            raise ValueError("Unsupported KV cache type for decode offload")
        use_mla = isinstance(kv_cache, MLATokenToKVPool)
        self.decode_host_mem_pool = (
            None
            if self.is_dsv4
            else build_kv_host_pool(
                kv_pool=kv_cache,
                page_size=self.page_size,
                use_mla=use_mla,
                # Host rows must have the device pool's row geometry; a packed DSA
                # row is wider than the width the MLA host pool assumes without
                # the override.
                override_kv_cache_dim=kv_cache.kv_cache_dim if use_mla else None,
            )
        )

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        hicache_storage_backend_extra_config = load_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )

        controller_kwargs = dict(
            load_cache_event=threading.Event(),
            storage_backend=get_memory().hicache_storage_backend,
            model_name=get_serving().served_model_name,
            storage_backend_extra_config=hicache_storage_backend_extra_config,
        )
        if self.is_dsv4:
            # The existing P-side storage contract uses one key per global page.
            # Independent NPU C128 coordinates require a different descriptor.
            if self.page_size % 128 or kv_cache.swa_page_size != self.page_size:
                raise ValueError(
                    "DSV4 decode offload requires matching full/SWA page sizes "
                    "and complete 128-token compression groups"
                )
            if (
                getattr(token_to_kv_pool_allocator, "c128_attn_allocator", None)
                is not None
            ):
                raise ValueError(
                    "DSV4 decode offload does not support NPU independent C128 allocation"
                )
            window = kv_cache.sliding_window
            if window is not None:
                # SWA storage is page addressed. A model window smaller than one
                # page still occupies one page in the non-unified DSV4 pool.
                window = max(
                    self.page_size,
                    ((window + self.page_size - 1) // self.page_size) * self.page_size,
                )
            if not getattr(kv_cache, "_unified_kv", False) and (
                window is None or self.offload_stride > window
            ):
                raise ValueError(
                    "DSV4 decode offload stride must fit the live SWA window"
                )
            params = CacheInitParams(
                disable=True,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                page_size=self.page_size,
                tp_cache_group=tp_group,
                attn_cp_cache_group=attn_cp_group,
                attn_tp_cache_group=attn_tp_group,
                pp_cache_group=pp_group,
            )
            self.decode_host_mem_pool, self.cache_controller = (
                build_deepseek_v4_hicache_stack(
                    params=params, kvcache=kv_cache, **controller_kwargs
                )
            )
            self.sidecar_specs = deepseek_v4_sidecar_specs(self.decode_host_mem_pool)
        else:
            self.cache_controller = HiCacheController(
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                mem_pool_host=self.decode_host_mem_pool,
                page_size=self.page_size,
                tp_group=tp_group,
                io_backend=get_memory().hicache_io_backend,
                **controller_kwargs,
            )

        self.ongoing_offload = {}
        self.ongoing_backup = {}
        self.offload_extra_pools = {}
        self.backup_extra_pools = {}
        # Keyed by Req identity (rids can be reused while a D2H copy is still
        # in flight); weak keys so a dropped Req is never pinned here.
        self.offloaded_state: WeakKeyDict[Req, OffloadedState] = WeakKeyDict()
        self.offload_inflight: WeakKeyDict[Req, int] = WeakKeyDict()
        logger.info("Enable offload kv cache for decode side")

    def release_host_resources(self) -> None:
        if getattr(self, "_host_resources_released", False):
            return
        if self.is_dsv4:
            # Shutdown must not unregister buffers under a D2H or backend IO.
            self.cache_controller.l2_transfer_engine.device_to_host_stream.synchronize()
            self.cache_controller.detach_storage_backend()
        self.decode_host_mem_pool.destroy()
        self._host_resources_released = True

    def _mark_offload_started(self, req: Req):
        self.offload_inflight[req] = self.offload_inflight.get(req, 0) + 1

    def _mark_offload_finished(self, req: Req):
        count = self.offload_inflight.get(req, 0)
        if count <= 1:
            self.offload_inflight.pop(req, None)
        else:
            self.offload_inflight[req] = count - 1

    def _has_inflight_offload(self, req: Req):
        return self.offload_inflight.get(req, 0) > 0

    def _prefill_offloaded_len(self, req: Req) -> int:
        # Page-aligned prompt length; the prefill instance offloaded this part.
        return len(req.origin_input_ids) // self.page_size * self.page_size

    def offload_kv_cache(self, req) -> bool:
        """Offload incremental KV cache for decode side."""

        if self.cache_controller is None or self.decode_host_mem_pool is None:
            return False

        if req.kv.req_pool_idx in (None, -1) or len(req.output_ids) == 0:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.kv.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        # Prefill side offloads page-aligned origin_input_ids, decode side offloads the incremental part
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        prefill_offloaded_len = self._prefill_offloaded_len(req)
        state = self.offloaded_state.get(req)
        if state is None:
            prefill_hashes = self._compute_prefix_hash(
                req, req.origin_input_ids[:prefill_offloaded_len]
            )
            last_prefill_hash = (
                prefill_hashes[-1] if prefill_offloaded_len > 0 else None
            )
            state = OffloadedState(last_hash=last_prefill_hash)
            self.offloaded_state[req] = state
        incremental_total = len(all_tokens) - prefill_offloaded_len
        incremental_new = incremental_total - state.inc_len
        incremental_aligned_len = (
            incremental_new // self.offload_stride * self.offload_stride
        )

        if incremental_aligned_len == 0:
            return False

        # Extract incremental tokens and indices for the newly available chunk
        start = prefill_offloaded_len + state.inc_len
        end = start + incremental_aligned_len
        incremental_tokens = all_tokens[start:end]
        incremental_indices = token_indices[start:end]
        extra_pools = (
            self._dsv4_device_transfers(incremental_indices) if self.is_dsv4 else None
        )
        if self.is_dsv4 and not self._all_ranks_ready(extra_pools is not None):
            # An evicted/unmapped SWA page must never be advertised as a full
            # DSV4 hit. Leave the hash/length frontier unchanged for a retry.
            return False

        # Prefill-aligned GPU slots are freed at request finish in
        # _release_finished_req, NOT here. The decoding request
        # continues to attend to those slots via req_to_token; freeing
        # them mid-decode races with concurrent admission, which can
        # reuse the slots and produce cross-pollinated KV reads.

        # Asynchronously offload incremental KV cache from device to host
        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = None
        try:
            host_indices = self.cache_controller.write(
                device_indices=incremental_indices.long(),
                node_id=ack_id,
                **({"extra_pools": extra_pools} if self.is_dsv4 else {}),
            )
        except CacheWriteSubmissionError:
            if not self.is_dsv4:
                raise
            logger.warning("DSV4 D2H submission failed; host allocations rolled back")
        if self.is_dsv4 and not self._all_ranks_ready(host_indices is not None):
            if host_indices is not None:
                # A peer could not allocate/submit. Cancel this rank's completed
                # snapshot so all ranks keep the same submitted/hash frontier.
                ack = self.cache_controller.ack_write_queue.pop()
                assert ack.node_ids == [ack_id]
                ack.finish_event.synchronize()
                self.decode_host_mem_pool.release_transfers(extra_pools)
                self.decode_host_mem_pool.free(host_indices)
            return False
        if host_indices is None:
            logger.error("Not enough host memory for request <redacted>")
            return False

        self._mark_offload_started(req)
        if extra_pools:
            self.offload_extra_pools[ack_id] = extra_pools
        self.ongoing_offload[ack_id] = (
            req,
            host_indices,
            incremental_tokens,
            time.time(),
        )
        state.inc_len += incremental_aligned_len
        if self.is_dsv4:
            # Chunk-cache SWA eviction is independent of this manager. Complete
            # the snapshot before returning control to the scheduler, which may
            # recycle SWA pages on its next batch. Storage remains asynchronous.
            self.cache_controller.ack_write_queue[-1].finish_event.synchronize()
        return True

    def _all_ranks_ready(self, ready):
        if self.tp_world_size == 1:
            return ready
        status = torch.tensor(int(ready), dtype=torch.int)
        torch.distributed.all_reduce(
            status, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
        )
        return bool(status.item())

    def _dsv4_device_transfers(self, full_indices):
        transfers = []
        if PoolName.SWA in self.decode_host_mem_pool.entry_map:
            kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
            swa_indices = kv_cache.translate_loc_from_full_to_swa(full_indices).long()
            rows = swa_indices.reshape(-1, self.page_size)
            offsets = torch.arange(self.page_size, device=swa_indices.device)
            if (
                not bool((swa_indices > 0).all())
                or swa_indices.unique().numel() != swa_indices.numel()
                or bool((rows[:, 0] % self.page_size != 0).any())
                or not torch.equal(rows, rows[:, :1] + offsets)
            ):
                return None
            transfers.append(
                PoolTransfer(
                    name=PoolName.SWA,
                    device_indices=swa_indices,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            )
        transfers.extend(
            PoolTransfer(
                name=spec.pool_name,
                indices_from_pool=spec.indices_from_pool,
                hit_policy=spec.hit_policy,
            )
            for spec in self.sidecar_specs
        )
        return transfers

    def check_offload_progress(self):
        """Check the progress of offload from device to host and backup from host to storage."""
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                len(cc.ack_write_queue),
                cc.ack_backup_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_write, n_backup = map(int, qsizes.tolist())
        self._check_offload_progress(n_write)
        self._check_backup_progress(n_backup)

    def _check_offload_progress(self, finish_count):
        """Check the progress of offload from device to host."""
        while finish_count > 0:
            ack = self.cache_controller.ack_write_queue.pop(0)
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                pending = self.ongoing_offload.pop(ack_id, None)
                if pending is None:
                    continue  # Duplicate/stale completion owns no allocations.
                (
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                ) = pending

                self._mark_offload_finished(req)
                prior_hash = (
                    self.offloaded_state[req].last_hash
                    if req in self.offloaded_state
                    else None
                )
                last_hash = self._trigger_backup(
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    prior_hash,
                    self.offload_extra_pools.pop(ack_id, None),
                )
                if req in self.offloaded_state:
                    self.offloaded_state[req].last_hash = last_hash

                if req.finished() and not self._has_inflight_offload(req):
                    self._release_finished_req(req)
            finish_count -= 1

    def _release_finished_req(self, req: Req):
        # Defensive guard: ReqToTokenPool.free sets req_pool_idx to None,
        # so a previously-released request must be skipped here to avoid
        # non-idempotent side effects (e.g. tree_cache.protected_size_
        # double-decrement, host pool double-free).
        if req.kv.req_pool_idx is None or req.kv.req_pool_idx == -1:
            return

        # Released only at request finish; a mid-decode free races with
        # concurrent admission over live slots.
        self.tree_cache.free_kv_row(req.kv, [(0, req.kv.kv_allocated_len)])

        self.req_to_token_pool.free(req)
        req.kv.mark_kv_released()
        self.tree_cache.protected_size_ -= len(req.prefix_indices)
        self.offloaded_state.pop(req, None)

    def _check_backup_progress(self, finish_count):
        """Check the progress of backup from host to storage."""
        for _ in range(finish_count):
            storage_operation = self.cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            pending = self.ongoing_backup.pop(ack_id, None)
            if pending is None:
                continue
            req_id, host_indices, start_time = pending
            extra_pools = self.backup_extra_pools.pop(ack_id, None)
            dsv4_complete = (
                self._dsv4_backup_complete(storage_operation) if self.is_dsv4 else None
            )
            if self.is_dsv4:
                shared_cache_diagnostics.record_backup(
                    phase="ack",
                    request_id=req_id,
                    operation_id=ack_id,
                    complete=dsv4_complete,
                    tokens=storage_operation.completed_tokens,
                    tenant_id=getattr(
                        getattr(
                            getattr(self.cache_controller, "storage_backend", None),
                            "config",
                            None,
                        ),
                        "tenant_id",
                        "default",
                    ),
                )
            if self.is_dsv4 and not dsv4_complete:
                # Cache writes are best effort. Complete-pool lookup on P stops
                # at the missing page; later chunks cannot bridge this gap.
                logger.warning(
                    "Incomplete DSV4 storage backup; releasing failed host snapshot"
                )

            # Release host memory
            self.decode_host_mem_pool.free(host_indices)
            if extra_pools:
                self.decode_host_mem_pool.release_transfers(extra_pools)

            logger.debug(
                "Finished backup request <redacted>, free host memory, len:<redacted>, cost time:<redacted> seconds."
            )

    def _dsv4_backup_complete(self, operation):
        cc = self.cache_controller
        complete = not getattr(operation, "backup_failed", False) and (
            cc.backup_skip or operation.completed_tokens == len(operation.token_ids)
        )
        hits = operation.pool_storage_result.extra_pool_hit_pages
        for transfer in operation.pool_transfers or []:
            if cc.should_backup(transfer):
                complete = complete and hits.get(transfer.name, 0) == len(
                    operation.hash_value
                )
        return complete

    def _trigger_backup(
        self,
        req,
        host_indices,
        incremental_tokens,
        start_time,
        prior_hash,
        extra_pools=None,
    ):
        """Trigger async backup from host to storage."""
        page_hashes = self._compute_prefix_hash(req, incremental_tokens, prior_hash)
        for transfer in extra_pools or []:
            transfer.keys = page_hashes
        try:
            ack_id = self.cache_controller.write_storage(
                host_indices,
                incremental_tokens,
                hash_value=page_hashes,
                **({"extra_pools": extra_pools} if extra_pools else {}),
            )
        except Exception:
            if not self.is_dsv4:
                raise
            self.decode_host_mem_pool.release_transfers(extra_pools)
            self.decode_host_mem_pool.free(host_indices)
            logger.warning("DSV4 storage enqueue failed; releasing host snapshot")
            return page_hashes[-1] if page_hashes else prior_hash
        if extra_pools:
            self.backup_extra_pools[ack_id] = extra_pools
        self.ongoing_backup[ack_id] = (req.rid, host_indices, start_time)
        if self.is_dsv4:
            shared_cache_diagnostics.record_backup(
                phase="submitted",
                request_id=req.rid,
                operation_id=ack_id,
                complete=False,
                tokens=len(incremental_tokens),
                tenant_id=getattr(
                    getattr(
                        getattr(self.cache_controller, "storage_backend", None),
                        "config",
                        None,
                    ),
                    "tenant_id",
                    "default",
                ),
            )
        return page_hashes[-1] if len(page_hashes) > 0 else prior_hash

    def _compute_prefix_hash(self, req: Req, tokens, prior_hash=""):
        """Match prefill storage hashes."""
        page_hashes = []
        last_hash = prior_hash or storage_namespace_seed(req.extra_key, req.cache_salt)
        for offset in range(0, len(tokens), self.page_size):
            page_tokens = tokens[offset : offset + self.page_size]
            last_hash = self.cache_controller.get_hash_str(page_tokens, last_hash)
            page_hashes.append(last_hash)
        return page_hashes

    def finalize_release_on_finish(self, req: Req):
        """Free any remaining tail KV that was not offloaded due to non-aligned length."""
        if self._has_inflight_offload(req):
            return
        self._release_finished_req(req)
