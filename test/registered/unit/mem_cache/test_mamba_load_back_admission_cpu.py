"""CPU admission/load-back regression using repository production method bodies.

Only device execution/transport and the single-node tree ledger are fixtures.
All slot allocation, admission, load-back preparation/rollback and controller
allocation methods under test are AST-extracted from repository source. This
keeps the test runnable with CPU torch without importing GPU-only dependencies.
Run this file directly, through unittest/pytest, or with --baseline-root pointing
to an unpatched checkout to reproduce the original failure as a control.
"""

import ast, contextlib, dataclasses, json, logging
import argparse, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"
BASELINE_ROOT = None
VARIANT_ROOTS = {}
g = globals()
logger = logging.getLogger(__name__)
contextmanager = contextlib.contextmanager
cast = lambda typ, value: value


class CT:
    FULL = "full"
    MAMBA = "mamba"
    SWA = "swa"
    C128 = "c128"


ComponentType = CT


class PN:
    KV = "kv"
    MAMBA = "mamba"


PoolName = PN


class Phase:
    LOAD_BACK = "load_back"
    BACKUP_HOST = "backup"
    BACKUP_STORAGE = "storage"
    PREFETCH = "prefetch"


CacheTransferPhase = Phase


class Verdict:
    CONTINUE = "continue"
    OTHER = "other"
    NO_TOKEN = "no_token"


AddReqResult = Verdict
TreeComponent = BaseHiCacheController = BasePrefixCache = object
CLIP_MAX_NEW_TOKENS = 4096


class Record:
    def __init__(self, *args, **kw):
        self.args = args
        self.__dict__.update(kw)


EvictParams = CacheOperation = _OngoingLoadBack = Record


@dataclasses.dataclass
class Prep:
    allocated_mamba_slot: object = None


PrepareLoadBackResult = Prep


class Transfer:
    def __init__(
        self, name, host_indices=None, device_indices=None, nodes_to_load=None, **kw
    ):
        self.name = name
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.nodes_to_load = nodes_to_load
        self.indices_from_pool = None
        self.anchor_index_parts = None
        self.__dict__.update(kw)


PoolTransfer = Transfer


def extract(path, cls=None, methods=None):
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == cls
    )
    if isinstance(node, ast.ClassDef) and methods is not None:
        node.body = [
            n
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name in methods
        ]
        if cls == "Scheduler":
            node.bases = []
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n.decorator_list = [
                d
                for d in n.decorator_list
                if isinstance(d, ast.Name) and d.id in ("contextmanager", "property")
            ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), g)


def load(variant):
    def p(f):
        root = VARIANT_ROOTS.get(
            variant, BASELINE_ROOT if variant == "baseline" else SOURCE_ROOT
        )
        x = root / f
        return x if x.exists() else SOURCE_ROOT / f

    extract(p("mem_cache/allocator/mamba.py"), "MambaSlotAllocator")
    extract(
        p("mem_cache/memory_pool.py"),
        "ReqToTokenPool",
        ["available_size", "alloc", "alloc_rows"],
    )
    extract(
        p("mem_cache/memory_pool.py"),
        "HybridReqToTokenPool",
        [
            "alloc",
            "mamba_admission_slots",
            "_initial_mamba_ping_pong_slots",
            "_init_mamba_ping_pong_buffer",
            "get_mamba_ping_pong_other_idx",
        ],
    )
    extract(p("mem_cache/allocation.py"), "alloc_req_slots")
    extract(p("mem_cache/base_prefix_cache.py"), "InitLoadBackParams")
    extract(
        p("mem_cache/unified_cache/components/mamba.py"),
        "MambaComponent",
        ["prepare_load_back", "finalize_load_back", "build_hicache_transfers"],
    )
    extract(
        p("mem_cache/hybrid_cache/hybrid_cache_controller.py"),
        "HybridCacheController",
        ["load", "_resolve_device_transfers"],
    )
    extract(
        p("mem_cache/unified_radix_cache.py"),
        "UnifiedRadixCache",
        ["load_back", "_load_back_transfers", "init_load_back"],
    )
    extract(
        p("managers/scheduler.py"), "Scheduler", ["_ensure_mamba_admission_capacity"]
    )
    extract(
        p("managers/schedule_policy.py"), "PrefillAdder", ["add_one_req", "_lock_node"]
    )


class KV:
    def __init__(self):
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.req_pool_idx = None
        self.kv_allocated_len = 0
        self.mamba_needs_clear = False

    @property
    def holds_mamba(self):
        return self.mamba_pool_idx is not None

    @property
    def holds_kv(self):
        return self.req_pool_idx is not None


def make_req(name):
    r = NS(
        rid=name,
        kv=KV(),
        sampling_params=NS(ignore_eos=False, max_new_tokens=1),
        output_ids=[],
        full_untruncated_fill_ids=list(range(128)),
        prefix_indices=torch.empty(0, dtype=torch.int64),
        last_node=0,
        best_match_node=1,
        host_hit_length=64,
        swa_host_hit_length=0,
        mamba_host_hit_length=1,
    )
    r.needs_host_load_back = lambda: True
    return r


def fixture(free, previous=False, extra=True):
    pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
    pool.enable_mamba_extra_buffer = extra
    pool.enable_mamba_extra_buffer_lazy = False
    pool.mamba_ping_pong_track_buffer_size = 2
    pool.mamba_ckpt_pool = None
    pool.mamba_allocator = MambaSlotAllocator(97, "cpu")
    pool.mamba_allocator.alloc(97 - free)
    pool.free_slots = [5, 6, 7, 8]
    pool.req_generation = torch.zeros(9, dtype=torch.int64)
    pool._on_alloc_rows = None
    pool.req_index_to_mamba_index_mapping = torch.zeros(9, dtype=torch.int32)
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
        (9, 2), dtype=torch.int64
    )
    pool.mamba_pool = NS(replayssm_write_pos=None)
    chunk = make_req("chunk")
    chunk.kv.mamba_pool_idx = torch.tensor(1)
    chunk.kv.mamba_ping_pong_track_buffer = torch.tensor([2, 3])
    chunk.kv.req_pool_idx = 1
    chunk.kv.kv_allocated_len = 64
    prior = [chunk]
    if previous:
        earlier = make_req("earlier")
        earlier.kv.mamba_pool_idx = torch.tensor(4)
        prior.append(earlier)
    req = make_req("candidate")
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.req_to_token_pool = pool
    cache.buffer_pipeline = None
    cache.linker = None
    cache.load_back_threshold = 1
    cache.ongoing_load_back = {}
    cache.locks = 0
    cache.host_locks = 0
    cache.cached = []
    cache.evict_calls = 0

    def inc(node):
        cache.locks += 1
        return NS(delta=0, to_dec_params=lambda: None)

    def dec(node, *args):
        cache.locks -= 1

    def inch(node):
        cache.host_locks += 1
        return NS(to_dec_params=lambda: None)

    def dech(node, *args):
        cache.host_locks -= 1

    cache.inc_lock_ref = inc
    cache.dec_lock_ref = dec
    cache.inc_host_lock_ref = inch
    cache.dec_host_lock_ref = dech
    cache.supports_mamba = lambda: True
    cache.get_session_kv = lambda req: None
    cache.is_tree_cache = lambda: True
    cache.evict_for_alloc = lambda params: None
    cache._component_available_size = lambda ct: 10000
    cache._build_sidecar_transfers = lambda *args: []
    cache._apply_cache_actions = lambda actions: None
    cache.token_to_kv_pool_allocator = NS()
    cd = NS(value=None, host_value=torch.tensor([9]))
    node = NS(id=1, component_data={CT.MAMBA: cd})
    tree = NS(
        component_has_host_value_only=lambda node, ct: cd.value is None
        and cd.host_value is not None,
        is_full_device_evicted=lambda node: True,
        empty_match_result=NS(device_indices=torch.empty(0, dtype=torch.int64)),
    )
    comp = MambaComponent.__new__(MambaComponent)
    comp.cache = cache
    comp.tree_core = tree
    comp.component_type = CT.MAMBA
    cache._components_tuple = (comp,)
    cache.tree_core = tree

    def build(node_id, req=None):
        return Transfer(PN.KV, host_indices=torch.arange(64)), {
            CT.MAMBA: comp.build_hicache_transfers(
                node, Phase.LOAD_BACK, mamba_pool_idx=req.kv.mamba_pool_idx
            )
        }

    def commit(node, device, kv, xfers):
        cd.value = xfers[CT.MAMBA][0].device_indices
        cache.cached.append(cd.value)
        return []

    tree.build_load_back_spec = build
    tree.commit_load_back = commit
    tree.collect_full_device_indices = lambda *args: torch.arange(64)
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.device = "cpu"
    controller.load_queue = []
    controller.mem_pool_device_allocator = NS(
        alloc=lambda n: torch.arange(n), free=lambda ids: None
    )
    entry = NS(
        device_indices_from_anchor_fn=None,
        device_alloc_fn=None,
        device_free_fn=None,
        device_pool=pool.mamba_allocator,
        device_evict_fn=lambda size: None,
    )
    controller.mem_pool_host = NS(entry_map={PN.MAMBA: entry})
    cache.cache_controller = controller
    adder = PrefillAdder.__new__(PrefillAdder)
    adder.tree_cache = cache
    adder.can_run_list = prior
    adder.prefill_max_requests = None
    adder.per_req_token_overhead = 1
    adder.prefill_delayer_single_pass = None
    adder.dllm_config = None
    adder._mamba_gap_budget_for_req = lambda r: 0
    adder._select_prefill_admission = lambda req, **kw: NS(
        prefix_len=kw["host_hit_length"],
        extend_len=128 - kw["host_hit_length"],
        is_chunked=False,
        max_new_tokens=1,
    )
    adder._kv_shard_reserve_scratch = lambda **kw: True
    adder._commit_prefill_admission = lambda req, *args: adder.can_run_list.append(req)
    adder.budget_state = lambda: Verdict.CONTINUE
    sched = Scheduler.__new__(Scheduler)
    sched.req_to_token_pool = pool
    sched.tree_cache = cache
    sched.token_to_kv_pool_allocator = NS()
    return pool, cache, adder, sched, req


def run_case(variant, label, free, previous=False, extra=True):
    load(variant)
    pool, cache, adder, sched, req = fixture(free, previous, extra)
    gate = sched._ensure_mamba_admission_capacity(req, adder.can_run_list)
    assert gate
    before = pool.mamba_allocator.available_size()
    verdict = adder.add_one_req(req, False, None)
    failure = None
    try:
        rows = alloc_req_slots(pool, adder.can_run_list, cache)
    except RuntimeError as e:
        failure = str(e)
        rows = None
    result = dict(
        variant=variant,
        case=label,
        gate=gate,
        free_before=before,
        verdict=verdict,
        load_queue=len(cache.cache_controller.load_queue),
        free_after=pool.mamba_allocator.available_size(),
        rows=rows,
        failure=failure,
        locks=cache.locks,
        host_locks=cache.host_locks,
    )
    if variant == "candidate":
        assert failure is None, result
        assert len(pool.mamba_allocator.free_slots.unique()) == len(
            pool.mamba_allocator.free_slots
        )
        assert pool.mamba_allocator.available_size() >= 0
        assert all(r.kv.holds_mamba for r in adder.can_run_list)
        if extra:
            assert all(
                r.kv.mamba_ping_pong_track_buffer is not None
                for r in adder.can_run_list
            )
        if not cache.cache_controller.load_queue:
            assert cache.locks == cache.host_locks == 0, result
    print(json.dumps(result))
    return result


def rollback_controls():
    load("candidate")
    for mode in (
        "reservation_failure",
        "full_allocation_failure",
        "repeat_checkpoint_shortage",
    ):
        pool, cache, adder, sched, req = fixture(3)
        if mode == "full_allocation_failure":
            cache.cache_controller.mem_pool_device_allocator.alloc = lambda n: None
        reserve = 6 if mode == "reservation_failure" else 3
        repetitions = 20 if mode == "repeat_checkpoint_shortage" else 1
        for _ in range(repetitions):
            assert cache.load_back(1, req=req, mamba_slots_to_reserve=reserve) is False
            assert pool.mamba_allocator.available_size() == 3
            assert req.kv.mamba_pool_idx is None
            assert cache.locks == cache.host_locks == 0
            assert len(cache.cache_controller.load_queue) == 0
        print(
            json.dumps(
                {
                    "variant": "candidate",
                    "case": mode,
                    "repetitions": repetitions,
                    "result": "PASS",
                    "slots_returned": 3,
                    "locks": 0,
                }
            )
        )
    pool, cache, adder, sched, req = fixture(3)
    assert adder.add_one_req(req, False, None) == Verdict.OTHER
    assert req not in adder.can_run_list and req.kv.mamba_pool_idx is None
    # A different owner's completion returns one slot; retry restores normally.
    pool.mamba_allocator.free(torch.tensor([93]))
    assert adder.add_one_req(req, False, None) == Verdict.CONTINUE
    assert req in adder.can_run_list and len(cache.cache_controller.load_queue) == 1
    assert alloc_req_slots(pool, adder.can_run_list, cache) == [1, 8]
    print(
        json.dumps(
            {
                "variant": "candidate",
                "case": "deferred_then_retry",
                "result": "PASS",
                "request_preserved": True,
                "restored_after_capacity_return": True,
            }
        )
    )
    pool, cache, adder, sched, req = fixture(3)
    req.kv.mamba_pool_idx = torch.tensor(4)
    cache.cache_controller.mem_pool_device_allocator.alloc = lambda n: None
    assert cache.load_back(1, req=req, mamba_slots_to_reserve=2) is False
    assert (
        req.kv.mamba_pool_idx.item() == 4 and pool.mamba_allocator.available_size() == 3
    )
    assert cache.locks == cache.host_locks == 0
    print(
        json.dumps(
            {
                "variant": "candidate",
                "case": "preexisting_main_preserved_on_failure",
                "result": "PASS",
            }
        )
    )


def ablations():
    specs = [
        (
            "ablation_no_reservation",
            "mem_cache/unified_radix_cache.py",
            "reserve = max(0, mamba_slots_to_reserve - prepared_slots)",
            "reserve = 0",
            "incident_chunk_plus_new",
            3,
            False,
        ),
        (
            "ablation_current_only",
            "managers/schedule_policy.py",
            "for candidate in [*self.can_run_list, req]",
            "for candidate in [req]",
            "prior_unallocated_pingpong",
            5,
            True,
        ),
        (
            "ablation_no_prepared_credit",
            "mem_cache/unified_radix_cache.py",
            "reserve = max(0, mamba_slots_to_reserve - prepared_slots)",
            "reserve = mamba_slots_to_reserve",
            "enough_for_checkpoint",
            4,
            False,
        ),
    ]
    for name, file, old, new, label, free, prior in specs:
        with tempfile.TemporaryDirectory(
            prefix="mamba-admission-ablation-"
        ) as directory:
            VARIANT_ROOTS[name] = Path(directory)
            p = Path(directory) / file
            p.parent.mkdir(parents=True, exist_ok=True)
            s = (SOURCE_ROOT / file).read_text()
            assert s.count(old) == 1
            p.write_text(s.replace(old, new))
            result = run_case(name, label, free, prior)
            del VARIANT_ROOTS[name]
        if name == "ablation_no_prepared_credit":
            assert result["load_queue"] == 0 and result["failure"] is None
        else:
            assert result["failure"] is not None


def run_suite():
    results = []
    variants = (
        ("baseline", "candidate") if BASELINE_ROOT is not None else ("candidate",)
    )
    for variant in variants:
        for label, free, previous, extra in [
            ("incident_chunk_plus_new", 3, False, True),
            ("prior_unallocated_pingpong", 5, True, True),
            ("enough_for_checkpoint", 4, False, True),
            ("no_extra_buffer", 2, False, False),
            ("ample_capacity", 12, True, True),
        ]:
            results.append(run_case(variant, label, free, previous, extra))
    if BASELINE_ROOT is not None:
        assert results[0]["failure"] and results[1]["failure"]
        assert results[0]["failure"].endswith(
            "req_to_token_pool.available_size()=4, num_reqs=2, "
        )
    candidate = [r for r in results if r["variant"] == "candidate"]
    assert candidate[0]["load_queue"] == 0 and candidate[2]["load_queue"] == 1
    rollback_controls()
    ablations()
    print(
        json.dumps(
            {
                "result": "PASS",
                "matrix_cases": len(results),
                "rollback_and_retry_controls": 5,
                "ablations": 3,
                "execution": "CPU production method bodies; tree/transport fixture",
            }
        )
    )


class TestMambaLoadBackAdmissionCPU(unittest.TestCase):
    def test_load_back_preserves_batch_admission(self):
        run_suite()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=SOURCE_ROOT,
        help="sglang/srt directory for the candidate",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="optional unpatched sglang/srt directory for red/green replay",
    )
    args = parser.parse_args()
    SOURCE_ROOT = args.source_root.resolve()
    BASELINE_ROOT = args.baseline_root.resolve() if args.baseline_root else None
    run_suite()
