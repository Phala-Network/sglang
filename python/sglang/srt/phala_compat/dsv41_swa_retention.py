"""Source-integrated adapter, converted from dsv41_swa_retention.py.
No import hooks or external runtime code paths.
"""

import os


import sys


ENV_ENABLE = "DSV41_SWA_RETENTION"


ENV_PARTS = "DSV41_SWA_RETENTION_PARTS"


_UPSTREAM_GATE = "SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS"


_TARGET = "sglang.srt.mem_cache.unified_cache.components.swa_component"


_MARK_FINISH = "_dsv41_swa_retention_finish"


_MARK_EVICT = "_dsv41_swa_retention_evict"


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "").strip().lower() in ("1", "true", "yes", "on")


def parts() -> set:
    raw = os.environ.get(ENV_PARTS, "finish,evict")
    return {p.strip() for p in raw.split(",") if p.strip()}


def _upstream_gate_on() -> bool:
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get())
    except Exception:
        raw = os.environ.get(_UPSTREAM_GATE)
        return True if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


def finished_frontier(token_ids_len: int, is_bigram: bool) -> int:
    """``len(radix_key) - 1`` for the key ``cache_finished_req`` inserts."""
    return token_ids_len - (1 if is_bigram else 0) - 1


def _apply_finish(cls) -> None:
    original = cls.prepare_for_caching_req
    if getattr(original, _MARK_FINISH, False):
        return
    if not hasattr(cls, "free_out_of_window_slots"):
        raise AttributeError("SWAComponent has no free_out_of_window_slots")

    def prepare_for_caching_req(self, req, insert_params, token_ids_len, is_finished):
        result = original(self, req, insert_params, token_ids_len, is_finished)
        if is_finished and result is None and _upstream_gate_on():
            pre_len = finished_frontier(
                token_ids_len, bool(getattr(self.tree_core, "is_eagle", False))
            )
            if pre_len > 0:
                self.free_out_of_window_slots(req, pre_len, insert_params)
        return result

    setattr(prepare_for_caching_req, _MARK_FINISH, True)
    prepare_for_caching_req.__name__ = "prepare_for_caching_req"
    prepare_for_caching_req.__wrapped__ = original
    cls.prepare_for_caching_req = prepare_for_caching_req


def tombstone_is_safe(node, component_type, sliding_window_size: int) -> bool:
    """False iff some child still holds live SWA and is shorter than a window."""
    for child in node.children.values():
        if (
            child.component_data[component_type].value is not None
            and len(child.key) < sliding_window_size
        ):
            return False
    return True


def _apply_evict(cls) -> None:
    original = cls._evict_device_next_node
    if getattr(original, _MARK_EVICT, False):
        return
    original_start = cls._evict_device_start

    def _evict_device_start(self, request_cnt):
        original_start(self, request_cnt)
        self._dsv41_deferred = False    # a load-bearing internal node was skipped this walk
        self._dsv41_retry_pass = False  # second pass from the LRU tail, upstream's rule

    def _evict_device_next_node(self, tracker, device_frees, host_frees):
        tree_core = self.tree_core
        if tree_core.enable_session_radix_cache:
            return original(self, tracker, device_frees, host_frees)
        ct = self.component_type
        lru = tree_core.lru_lists[ct]
        window = self.sliding_window_size
        while True:
            cursor = self._evict_device_cursor
            if cursor is not None and not lru.in_list(cursor):
                cursor = self._evict_device_cursor = lru.get_lru_no_lock()
            if tracker[ct] >= self._evict_device_request_cnt:
                return original(self, tracker, device_frees, host_frees)
            if cursor is None or not lru.in_list(cursor):
                if getattr(self, "_dsv41_deferred", False) and not getattr(
                    self, "_dsv41_retry_pass", False
                ):
                    # Every other candidate is exhausted and the target is not met:
                    # walk again from the LRU tail with upstream's rule. Deferral is a
                    # priority, never a reservation.
                    self._dsv41_retry_pass = True
                    self._evict_device_cursor = lru.get_lru_no_lock()
                    continue
                return original(self, tracker, device_frees, host_frees)
            if (
                getattr(self, "_dsv41_retry_pass", False)
                or cursor in tree_core.evictable_device_leaves
                or tombstone_is_safe(cursor, ct, window)
            ):
                return original(self, tracker, device_frees, host_frees)
            # Load-bearing internal node: defer it and keep walking in LRU order.
            self._dsv41_deferred = True
            self._evict_device_cursor = lru.get_prev_no_lock(cursor)

    setattr(_evict_device_next_node, _MARK_EVICT, True)
    _evict_device_next_node.__name__ = "_evict_device_next_node"
    _evict_device_next_node.__wrapped__ = original
    _evict_device_start.__name__ = "_evict_device_start"
    _evict_device_start.__wrapped__ = original_start
    cls._evict_device_start = _evict_device_start
    cls._evict_device_next_node = _evict_device_next_node


def apply(module) -> None:
    cls = getattr(module, "SWAComponent", None)
    if cls is None:
        raise AttributeError(f"{_TARGET} has no SWAComponent")
    wanted = parts()
    if "finish" in wanted:
        _apply_finish(cls)
    if "evict" in wanted:
        _apply_evict(cls)
