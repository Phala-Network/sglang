"""Tests against installed source-integrated adapters; no runtime package mounts."""
import os
import sys
from array import array

sys.path.insert(0, "/sgl-workspace/sglang/test/registered/unit/mem_cache")

import test_unified_radix_cache_unittest as T  # noqa: E402

# The fixture resolves its device through sglang's platform probe, which has no
# accelerator in an offline container. Every pool and allocator it builds takes
# that device as an argument, so binding the fixture's own name to CPU keeps the
# whole cache on CPU without touching a GPU.
T.get_device = lambda *args, **kwargs: "cpu"
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType  # noqa: E402
import sglang.srt.mem_cache.unified_cache.components.swa_component as swa_mod  # noqa: E402

from sglang.srt.phala_compat import dsv41_swa_retention as R  # noqa: E402

FAIL = 0


def check(label, got, want=True):
    global FAIL
    ok = got == want
    FAIL += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + ("" if ok else f": {got!r} != {want!r}"))


def cfg_for(page, window, kv, is_eagle=False):
    return T.CacheConfig(page_size=page, components=(ComponentType.FULL, ComponentType.SWA),
                         sliding_window_size=window, is_eagle=is_eagle, kv_size=kv,
                         max_context_len=kv, head_num=1, head_dim=8)


def finish_one(cfg, n_tokens):
    """cache_finished_req for one request, the way #39159's unit test drives it."""
    cache, allocator, pool = T.build_fixture(cfg)
    suite = T.UnifiedRadixCacheSuite()
    suite.cfg = cfg
    req = suite._make_req(pool)
    tokens = array("q", range(1, n_tokens + 1))
    req.origin_input_ids = tokens
    req.output_ids = array("q", [90000])
    loc = suite._alloc(allocator, len(tokens))
    pool.write((req.kv.req_pool_idx, slice(0, len(tokens))), loc)
    req.kv.kv_committed_len = len(tokens)
    req.last_node = cache.root_node_handle()
    req.full_untruncated_fill_ids = tokens
    req.set_extend_range(0, len(tokens))
    key = RadixKey(tokens, is_bigram=cfg.is_eagle).page_aligned(cfg.page_size)
    with envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(True):
        cache.cache_finished_req(req, kv_len_to_handle=len(tokens))
    retained = cfg.kv_size - allocator.swa_available_size()
    match = cache.match_prefix(T.MatchPrefixParams(key=key))
    cache.sanity_check()
    return retained, len(match.device_indices), len(key)


def strand_scenario():
    """page 64 < window 128: node P(128) with a live child C(64), then evict 64 SWA tokens.

    Returns (C still matchable?, P still matchable?, SWA tokens the eviction freed,
             SWA held by nodes no match can reach).
    """
    cfg = cfg_for(page=64, window=128, kv=4096)
    cache, allocator, pool = T.build_fixture(cfg)
    suite = T.UnifiedRadixCacheSuite()
    suite.cfg = cfg
    P = list(range(1, 129))            # 2 pages == one window, fits the tail cap unsplit
    C = P + list(range(5000, 5064))    # extends P by one 64-token page (< window)
    suite._insert(cache, allocator, pool, P)
    suite._insert(cache, allocator, pool, C)
    before = allocator.swa_available_size()
    cache.evict(EvictParams(num_tokens=0, swa_num_tokens=64))
    freed = allocator.swa_available_size() - before
    mc = len(cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", C)))).device_indices)
    mp = len(cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", P)))).device_indices)
    held = cfg.kv_size - allocator.swa_available_size()
    cache.sanity_check()
    return mc, mp, freed, held


def main() -> int:
    upstream_finish = swa_mod.SWAComponent.prepare_for_caching_req
    upstream_evict = swa_mod.SWAComponent._evict_device_next_node

    print("UPSTREAM (unpatched)")
    u3072 = finish_one(cfg_for(256, 128, 16384), 3072)
    u8192e = finish_one(cfg_for(256, 128, 16384, is_eagle=True), 8192)
    print(f"  finished 3,072: retained SWA {u3072[0]}, match {u3072[1]}/{u3072[2]}")
    print(f"  finished 8,192 eagle: retained SWA {u8192e[0]}, match {u8192e[1]}/{u8192e[2]}")
    check("upstream keeps the whole finished prefix as live SWA", u3072[0], 3072)
    us = strand_scenario()
    print(f"  strand scenario: match(C)={us[0]} match(P)={us[1]} evict freed={us[2]} swa held={us[3]}")
    check("upstream strands: C holds live SWA but no longer matches", us[0] < 192 and us[3] >= 64)

    os.environ[R.ENV_PARTS] = "finish,evict"
    R.apply(swa_mod)
    check("patch installed on the real class",
          getattr(swa_mod.SWAComponent.prepare_for_caching_req, R._MARK_FINISH, False)
          and getattr(swa_mod.SWAComponent._evict_device_next_node, R._MARK_EVICT, False))
    check("originals reachable via __wrapped__",
          swa_mod.SWAComponent.prepare_for_caching_req.__wrapped__ is upstream_finish
          and swa_mod.SWAComponent._evict_device_next_node.__wrapped__ is upstream_evict)

    print("PATCHED")
    p3072 = finish_one(cfg_for(256, 128, 16384), 3072)
    p8192e = finish_one(cfg_for(256, 128, 16384, is_eagle=True), 8192)
    p3073 = finish_one(cfg_for(256, 128, 16384), 3073)
    print(f"  finished 3,072: retained SWA {p3072[0]}, match {p3072[1]}/{p3072[2]}")
    print(f"  finished 3,073: retained SWA {p3073[0]}, match {p3073[1]}/{p3073[2]}")
    print(f"  finished 8,192 eagle: retained SWA {p8192e[0]}, match {p8192e[1]}/{p8192e[2]}")
    check("3,072-token finished prefix keeps 512 SWA tokens (one window + one page)", p3072[0], 512)
    check("...and still matches its whole key", p3072[1], p3072[2])
    check("unaligned 3,073 keeps >= one window and matches its whole key",
          p3073[0] >= 128 and p3073[1] == p3073[2])
    check("bigram (eagle) key keeps a window and matches its whole key",
          128 <= p8192e[0] <= 512 and p8192e[1] == p8192e[2])

    ps = strand_scenario()
    print(f"  strand scenario: match(C)={ps[0]} match(P)={ps[1]} evict freed={ps[2]} swa held={ps[3]}")
    check("no stranding: C evicted whole, so what remains is exactly P", (ps[0], ps[1]), (128, 128))
    check("eviction still met its 64-token SWA target", ps[2] >= 64)
    check("no SWA held by unreachable nodes (only P's 128 remain)", ps[3], 128)

    print("\nall checks passed" if not FAIL else f"\nFAILED: {FAIL} check(s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
