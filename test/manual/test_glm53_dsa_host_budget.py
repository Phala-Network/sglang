"""Pure CPU tests of the candidate's real budget/allocator/constructor code.

No SGLang installation, torch import, real mmap, GPU, remote access or large RAM
allocation is used. Eight-process tests exercise the actual context manager
with a small object-collective transport and integer-only memory simulation.
"""

from __future__ import annotations

import abc
import ast
import functools
import hashlib
import importlib.util
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import types
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional

HERE = Path(__file__).resolve().parent
SOURCE = Path(
    os.environ.get(
        "SGLANG_TEST_SOURCE_ROOT", str(HERE.parents[1] / "python/sglang/srt")
    )
)
GB, GIB, PAGE = 10**9, 1024**3, 2 * 1024**2
PREFIX = "sglang.srt.mem_cache"


def install(name, module=None):
    parts = name.split(".")
    for i in range(1, len(parts)):
        package_name = ".".join(parts[:i])
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package
    module = module or types.ModuleType(name)
    sys.modules[name] = module
    return module


def execute(path, name, namespace, names=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)
    ]
    for item in tree.body:
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            continue
        if names is None or getattr(item, "name", None) in names:
            body.append(item)
    module = install(name)
    module.__dict__.update(namespace)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(path),
            "exec",
        ),
        module.__dict__,
    )
    return module


class DType:
    def __init__(self, size):
        self.itemsize = size


class Tensor:
    next_pointer = 4096

    def __init__(self, shape=(), dtype=None, owner=None):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype = dtype or DType(1)
        self.owner = owner
        self.pointer = Tensor.next_pointer
        Tensor.next_pointer += max(4096, self.numel() * self.element_size())

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self.dtype.itemsize

    def data_ptr(self):
        return self.pointer

    def reshape(self, shape):
        self.shape = tuple(shape)
        return self

    def transpose(self, left, right):
        shape = list(self.shape)
        shape[left], shape[right] = shape[right], shape[left]
        return Tensor(shape, self.dtype, self)

    def __getitem__(self, index):
        return Tensor(self.shape[1:], self.dtype, self)

    def __len__(self):
        return self.shape[0]


class Cuda:
    def __init__(self):
        self.registered = set()
        self.calls = 0
        self.fail_at = None
        self.fail_unregister = False

    def cudaHostRegister(self, ptr, size, flags):
        self.calls += 1
        if self.calls == self.fail_at:
            return 1
        self.registered.add(ptr)
        return 0

    def cudaHostUnregister(self, ptr):
        if self.fail_unregister:
            return 1
        self.registered.discard(ptr)
        return 0

    def cudaGetErrorString(self, rc):
        return "simulated-error"


def fixture(world_size=1, rank=0, transport=None):
    env = NS(value="", enabled=True, reserve=128)
    cuda = Cuda()
    events = []
    torch = install("torch")
    torch.dtype = DType
    torch.Tensor = Tensor
    for name, size in {"uint8": 1, "bool": 1, "int64": 8, "uint64": 8}.items():
        setattr(torch, name, DType(size))
    torch.empty = lambda dims, dtype, **kw: Tensor(dims, dtype)
    torch.zeros = torch.empty
    torch.arange = lambda count, dtype: Tensor((count,), dtype)
    torch.tensor = lambda values, dtype, **kw: Tensor((len(values),), dtype)
    torch.frombuffer = lambda owner, dtype, count: Tensor((count,), dtype, owner)
    torch.cat = lambda values: Tensor((sum(v.numel() for v in values),), torch.uint64)
    torch.cuda = NS(cudart=lambda: cuda)
    round_number = [0]

    def all_gather(output, packet, group):
        shared, barrier = transport
        key = (round_number[0], rank)
        shared[key] = packet
        barrier.wait(timeout=20)
        output[:] = [shared[(round_number[0], i)] for i in range(world_size)]
        barrier.wait(timeout=20)
        round_number[0] += 1

    torch.distributed = NS(
        is_available=lambda: True,
        is_initialized=lambda: world_size > 1,
        all_gather_object=all_gather,
    )
    psutil = install("psutil")
    psutil.virtual_memory = lambda: NS(available=100 * GIB)
    parallel = NS(nnodes=1, tp_size=world_size, dcp_enabled=False)
    runtime = install("sglang.srt.runtime_context")
    runtime.get_parallel = lambda: parallel
    state = install("sglang.srt.distributed.parallel_state")
    state.get_world_group = lambda: NS(world_size=world_size, cpu_group="cpu")
    envs = NS(
        SGLANG_HUGEPAGE_SIZE=NS(get=lambda: env.value),
        SGLANG_HICACHE_DSA_STARTUP_BUDGET=NS(get=lambda: env.enabled),
        SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=NS(get=lambda: 256),
        SGLANG_HICACHE_HOST_MEMORY_RESERVE_GB=NS(get=lambda: env.reserve),
    )
    install("sglang.srt.environ").envs = envs
    mmap_stub = NS(PAGESIZE=4096, MAP_SHARED=1, MAP_ANONYMOUS=0x20, mmap=object)
    fail_huge = [False]

    def huge_map(nbytes, allocated, flags):
        events.append(("huge", allocated))
        if fail_huge[0]:
            raise OSError(12, "simulated hugepage exhaustion")
        return object()

    def plain_map(*args):
        events.append(("plain", args[1]))
        return object()

    mmap_module = execute(
        SOURCE / "mem_cache/storage/mmap/mmap_allocator.py",
        PREFIX + ".storage.mmap.mmap_allocator",
        dict(
            math=math,
            torch=torch,
            envs=envs,
            mmap=mmap_stub,
            _libc=object(),
            _MAP_HUGETLB=0x40000,
            _MAP_HUGE_2MB=21 << 26,
            _MAP_HUGE_1GB=30 << 26,
            _alloc_hugepage=huge_map,
            _mmap_prefaulted=plain_map,
            logger=logging.getLogger("mmap"),
        ),
        {"requested_hugepage_bytes", "alloc_mmap", "alloc_shm"},
    )
    storage_package = sys.modules[PREFIX + ".storage.mmap"]
    storage_package.alloc_mmap = mmap_module.alloc_mmap
    storage_package.alloc_shm = mmap_module.alloc_shm

    name = PREFIX + ".pool_host.allocation_budget"
    spec = importlib.util.spec_from_file_location(
        name, SOURCE / "mem_cache/pool_host/allocation_budget.py"
    )
    budget = install(name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(budget)
    real_snapshot = budget._resource_snapshot
    budget._resource_snapshot = lambda huge: (100 * GIB, 100 * PAGE if huge else 0)
    common = execute(
        SOURCE / "mem_cache/pool_host/common.py",
        PREFIX + ".pool_host.common",
        dict(
            json=json,
            logging=logging,
            math=math,
            mmap=mmap_stub,
            os=os,
            is_hip=lambda: False,
            lru_cache=functools.lru_cache,
            defaultdict=defaultdict,
            torch=torch,
            envs=envs,
            active_host_allocation_budget=budget.active_host_allocation_budget,
            alloc_mmap=mmap_module.alloc_mmap,
            requested_hugepage_bytes=mmap_module.requested_hugepage_bytes,
            get_memory=lambda: NS(hicache_storage_backend=None),
        ),
    )
    base = execute(
        SOURCE / "mem_cache/pool_host/base.py",
        PREFIX + ".pool_host.base",
        dict(
            abc=abc,
            logging=logging,
            threading=threading,
            wraps=functools.wraps,
            Optional=Optional,
            psutil=psutil,
            torch=torch,
            KVCache=object,
            get_world_group=state.get_world_group,
            get_parallel=runtime.get_parallel,
            get_allocator_from_storage=common.get_allocator_from_storage,
            _cuda_host_unregister=common._cuda_host_unregister,
            requested_hugepage_bytes=mmap_module.requested_hugepage_bytes,
            available_hugepage_bytes=budget.available_hugepage_bytes,
            host_slot_metadata_bytes=budget.host_slot_metadata_bytes,
            active_host_allocation_budget=budget.active_host_allocation_budget,
            is_cuda=lambda: True,
            is_hip=lambda: False,
        ),
    )
    return NS(
        env=env,
        cuda=cuda,
        events=events,
        torch=torch,
        psutil=psutil,
        parallel=parallel,
        mmap=mmap_module,
        fail_huge=fail_huge,
        budget=budget,
        real_snapshot=real_snapshot,
        common=common,
        base=base,
    )


def load_small_dsa(f):
    # Execute the candidate MLA constructor with lightweight device doubles.
    old = SOURCE / "mem_cache/pool_host/mla.py"
    shared = dict(
        torch=f.torch,
        HostKVCache=f.base.HostKVCache,
        make_kernel_ptr_table=lambda refs, device, **kwargs: Tensor(
            (len(refs),), f.torch.uint64
        ),
        HiSparseHostPoolMixin=type("Mixin", (), {}),
        MLATokenToKVPoolFP4=type("FP4Device", (), {}),
        ALLOC_MEMORY_FUNCS=f.common.ALLOC_MEMORY_FUNCS,
        _WRITE_BACK_STAGING_PAGE_CHUNK=64,
        _is_cuda=True,
        _is_hip=False,
        _is_npu=False,
        _is_xpu=False,
        _is_mps=False,
        can_use_hicache_jit_kernel=lambda **kw: False,
        can_use_write_back_jit_kernel=lambda **kw: False,
        logger=logging.getLogger("pool"),
    )
    mla = execute(old, PREFIX + ".pool_host.mla", shared, {"MLATokenToKVPoolHost"})
    dsa_device_class = type(
        "DSADevice", (), {"index_k_with_scale_buffer_dtype": f.torch.uint8}
    )
    dsa = execute(
        SOURCE / "mem_cache/pool_host/dsa.py",
        PREFIX + ".pool_host.dsa",
        shared
        | dict(
            threading=threading,
            DSATokenToKVPool=dsa_device_class,
            get_allocator_from_storage=f.common.get_allocator_from_storage,
            _cuda_host_unregister=f.common._cuda_host_unregister,
            active_host_allocation_budget=f.budget.active_host_allocation_budget,
            host_slot_metadata_bytes=f.budget.host_slot_metadata_bytes,
            host_memory_budget_bytes=f.base.host_memory_budget_bytes,
        ),
        {"DSAIndexerPoolHost"},
    )
    device = dsa_device_class()
    device.__dict__.update(
        store_dtype=f.torch.uint8,
        size=8,
        start_layer=0,
        end_layer=2,
        layer_num=2,
        layer_shard_enabled=False,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        kv_cache_dim=6,
        device="cuda",
        index_head_dim=4,
        quant_block_size=4,
        kv_buffer=[Tensor((8, 1, 6)) for _ in range(2)],
        index_k_with_scale_buffer=[Tensor((8, 8)) for _ in range(2)],
    )
    device.data_ptrs = Tensor((2,), f.torch.uint64)
    return mla, dsa, device


def worker(rank, shared, barrier, output, mode, fail_rank):
    try:
        f = fixture(8, rank, (shared, barrier))
        f.env.value = "2MB" if mode.startswith("huge") else ""
        huge_per_rank = sum(
            f.budget._round_up(value, PAGE) for value in (240 * GB, 55 * GB)
        )
        total_huge = 8 * huge_per_rank
        ordinary = 3000 * GB if mode == "ordinary" else 1000 * GB
        if mode == "oversubscribed":
            ordinary = 2100 * GB
        ordinary -= f.budget.host_memory_reserve_bytes()
        f.budget._resource_snapshot = lambda size: (ordinary, total_huge if size else 0)
        if mode == "huge_short":
            f.budget._resource_snapshot = lambda size: (ordinary, total_huge - PAGE)
        params = NS(pp_size=1)
        pools = []
        caught = None
        budget = None
        try:
            with f.budget.dsa_host_allocation_budget(params) as budget:
                for index, value in enumerate((240 * GB, 55 * GB)):
                    pool = NS(
                        allocator=f.common.HostTensorAllocator(),
                        pin_memory=True,
                        _destroyed=False,
                    )
                    pool.destroy = lambda p=pool: f.base.HostKVCache.destroy(p)
                    budget.register_pool(pool, 1 * 1024**2)
                    if fail_rank == rank and index == 1:
                        f.cuda.fail_at = f.cuda.calls + 1
                    tensor = f.common.alloc_with_host_register(
                        (value,), f.torch.uint8, "cpu", True, pool.allocator
                    )
                    setattr(
                        pool,
                        "kv_buffer" if index == 0 else "index_k_with_scale_buffer",
                        tensor,
                    )
                    pools.append(pool)
        except Exception as exc:
            caught = str(exc)
        output[rank] = dict(
            error=caught,
            registered=len(f.cuda.registered),
            closed=budget.closed if budget else None,
            remaining=budget.remaining if budget else None,
            initial=budget.initial if budget else None,
            destroyed=[p._destroyed for p in pools],
            events=f.events,
        )
    except BaseException as exc:
        output[rank] = {"worker_error": repr(exc)}


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture()

    def test_cumulative_credit_rejects_real_oversubscription(self):
        f = self.f
        b = f.budget.HostAllocationBudget(3 * PAGE, 0, 0)
        a = f.common.HostTensorAllocator()
        b.claim_mapping(2 * PAGE, a)
        with self.assertRaises(ValueError):
            b.claim_mapping(2 * PAGE, a)
        self.assertEqual(b.remaining[False], PAGE)

    def test_thread_claims_are_atomic(self):
        f = self.f
        b = f.budget.HostAllocationBudget(PAGE, 0, 0)
        start = threading.Barrier(8)
        results = []

        def claim():
            start.wait()
            try:
                b.claim_mapping(PAGE, f.common.HostTensorAllocator())
                results.append(True)
            except ValueError:
                results.append(False)

        threads = [threading.Thread(target=claim) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sum(results), 1)

    def test_each_huge_mapping_is_rounded(self):
        f = self.f
        f.env.value = "2MB"
        b = f.budget.HostAllocationBudget(GIB, 3 * PAGE, PAGE)
        a = f.common.HostTensorAllocator()
        b.claim_mapping(PAGE + 1, a)
        with self.assertRaises(ValueError):
            b.claim_mapping(PAGE + 1, a)

    def test_metadata_cannot_spend_huge_credit(self):
        f = self.f
        f.env.value = "2MB"
        b = f.budget.HostAllocationBudget(0, 100 * PAGE, PAGE)
        with self.assertRaises(ValueError):
            b.register_pool(NS(allocator=f.common.HostTensorAllocator()), 1)

    def test_strict_huge_mmap_failure_never_falls_back(self):
        f = self.f
        f.env.value = "2MB"
        f.fail_huge[0] = True
        with self.assertRaises(OSError):
            f.mmap.alloc_mmap((10,), f.torch.uint8)
        self.assertEqual(f.events, [("huge", PAGE)])

    def test_strict_huge_missing_libc_never_falls_back(self):
        f = self.f
        f.env.value = "2MB"
        f.mmap._libc = None
        with self.assertRaises(RuntimeError):
            f.mmap.alloc_mmap((10,), f.torch.uint8)
        self.assertEqual(f.events, [])

    def test_strict_huge_shm_and_native_rejected(self):
        f = self.f
        f.env.value = "2MB"
        with self.assertRaises(ValueError):
            f.mmap.alloc_shm((10,), f.torch.uint8)
        for a in (f.common.ShmHostTensorAllocator(), type("Native", (), {})()):
            with self.assertRaises(ValueError):
                f.common.host_mapping_page_size(a)
        self.assertEqual(f.events, [])

    def test_invalid_hugepage_value_rejected(self):
        self.f.env.value = "2MiB"
        with self.assertRaises(ValueError):
            self.f.mmap.alloc_mmap((10,), self.f.torch.uint8)

    def test_ordinary_path_preserved(self):
        f = self.f
        f.mmap.alloc_mmap((4097,), f.torch.uint8)
        self.assertEqual(f.events, [("plain", 8192)])

    def test_mode_change_during_transaction_rejected(self):
        f = self.f
        b = f.budget.HostAllocationBudget(GIB, 0, 0)
        f.env.value = "2MB"
        with self.assertRaises(RuntimeError):
            b.claim_mapping(PAGE, f.common.HostTensorAllocator())

    def test_real_mla_dsa_constructors_consume_actual_shapes(self):
        f = self.f
        mla, dsa, device = load_small_dsa(f)
        with f.budget.dsa_host_allocation_budget(NS(pp_size=1)) as b:
            kv = mla.MLATokenToKVPoolHost(device, 8, 0, 4, "layer_first")
            index = dsa.DSAIndexerPoolHost(device, kv, "layer_first")
            self.assertEqual(kv.size, 68)
            self.assertEqual(index.indexer_page_num, 18)
            self.assertEqual(b.initial[False] - b.remaining[False], 8 * 4096)
        self.assertEqual(len(f.cuda.registered), 2)
        kv.destroy()
        index.destroy()
        self.assertIsNone(kv.kv_buffer)
        self.assertIsNone(index.index_k_with_scale_buffer)
        self.assertIsNone(index.index_k_data_refs)
        self.assertEqual(len(f.cuda.registered), 0)

    def test_constructor_pin_failure_rolls_back_main_and_partial_index(self):
        f = self.f
        mla, dsa, device = load_small_dsa(f)
        f.cuda.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "DSA host allocation failed"):
            with f.budget.dsa_host_allocation_budget(NS(pp_size=1)) as b:
                kv = mla.MLATokenToKVPoolHost(device, 8, 0, 4, "layer_first")
                dsa.DSAIndexerPoolHost(device, kv, "layer_first")
        self.assertEqual(b.remaining, b.initial)
        self.assertTrue(b.closed)
        self.assertIsNone(kv.kv_buffer)
        self.assertIsNone(kv.data_refs)
        self.assertEqual(len(f.cuda.registered), 0)
        self.assertIsNone(f.budget.active_host_allocation_budget())

    def test_real_constructor_mtp_and_cp_shapes_in_both_page_modes(self):
        for mode in ("", "2MB"):
            for mtp in (1, 2, 3):
                f = fixture()
                f.env.value = mode
                mla, dsa, device = load_small_dsa(f)
                device.layer_shard_enabled = True
                device.layer_shard_size = 2
                drafts = []
                for _ in range(mtp):
                    draft = NS(
                        store_dtype=f.torch.uint8,
                        kv_cache_dim=6,
                        data_ptrs=Tensor((1,), f.torch.uint64),
                        kv_buffer=[Tensor((8, 1, 6))],
                        index_k_with_scale_buffer=[Tensor((8, 8))],
                    )
                    drafts.append(draft)
                with f.budget.dsa_host_allocation_budget(NS(pp_size=1)) as b:
                    kv = mla.MLATokenToKVPoolHost(
                        device, 8, 0, 4, "layer_first", mtp_draft_device_pools=drafts
                    )
                    index = dsa.DSAIndexerPoolHost(device, kv, "layer_first")
                    self.assertEqual(kv.layer_num, 1 + mtp)
                    self.assertEqual(index.layer_num, 1 + mtp)
                    align = PAGE if mode else 4096
                    expected = sum(
                        f.budget._round_up(buf.numel(), align)
                        for buf in (kv.kv_buffer, index.index_k_with_scale_buffer)
                    )
                    used = b.initial[bool(mode)] - b.remaining[bool(mode)]
                    self.assertEqual(used, expected + (0 if mode else 6 * 4096))
                kv.destroy()
                index.destroy()

    def test_v0520_dummy_dsa_does_not_allocate_or_spend_budget(self):
        mla, dsa, device = load_small_dsa(self.f)
        anchor = NS(page_size=4, mtp_draft_device_pools=(), size=8, page_num=3)
        before = list(self.f.events)
        dsa.active_host_allocation_budget = lambda: self.fail(
            "dummy must not enter physical allocation budget"
        )
        index = dsa.DSAIndexerPoolHost(
            device, anchor, "page_first_direct", is_dummy=True
        )
        self.assertIsNone(index.index_k_with_scale_buffer)
        self.assertIsNone(index.index_k_device_ptrs)
        self.assertEqual(before, self.f.events)

    def test_actual_assembler_gates_controller_until_pair_succeeds(self):
        for fail_index in (False, True):
            f = fixture()
            mla, dsa, device = load_small_dsa(f)
            controllers = []

            def controller(*args, **kwargs):
                self.assertIsNone(f.budget.active_host_allocation_budget())
                controllers.append(args[1])
                return object()

            memory = NS(
                hicache_ratio=8,
                hicache_size=0,
                hicache_mem_layout="layer_first",
                hicache_write_policy="write_back",
                hicache_io_backend="kernel",
                hicache_host_memory_mode="cache",
            )
            assembler = execute(
                SOURCE / "mem_cache/hybrid_cache/hybrid_pool_assembler.py",
                PREFIX + ".hybrid_cache.hybrid_pool_assembler",
                dict(
                    dsa_host_allocation_budget=f.budget.dsa_host_allocation_budget,
                    MLATokenToKVPoolHost=mla.MLATokenToKVPoolHost,
                    _get_allocator_type=lambda: "default",
                    get_parallel=lambda: f.parallel,
                    get_memory=lambda: memory,
                    build_pool_entry=lambda **kwargs: NS(**kwargs),
                    PoolName=NS(KV="kv"),
                    HostPoolGroup=lambda entries: NS(entries=entries),
                    HybridCacheController=controller,
                ),
                {"build_kv_host_pool", "build_anchor_sidecar_stack"},
            )
            params = NS(
                pp_size=1,
                mtp_draft_device_pools=(),
                page_size=4,
                token_to_kv_pool_allocator=object(),
                tp_cache_group=None,
                attn_cp_cache_group=None,
                attn_tp_cache_group=None,
                pp_cache_group=None,
            )
            if fail_index:
                f.cuda.fail_at = 2
            call = lambda: assembler.build_anchor_sidecar_stack(
                params=params,
                kv_pool=device,
                sidecar_pool_name="index",
                full_layer_mapping={0: 0, 1: 1},
                load_cache_event=None,
                storage_backend=None,
                use_mla=True,
                sidecar_host_pool_factory=lambda kv: dsa.DSAIndexerPoolHost(
                    device, kv, "layer_first"
                ),
            )
            if fail_index:
                with self.assertRaises(RuntimeError):
                    call()
                self.assertEqual(controllers, [])
                self.assertEqual(len(f.cuda.registered), 0)
            else:
                group, _ = call()
                self.assertEqual(len(controllers), 1)
                for entry in group.entries:
                    entry.host_pool.destroy()

    def test_registration_rollback_metadata_is_empty_after_first_failure(self):
        f = self.f
        f.cuda.fail_at = 1
        tensor = Tensor((10,), f.torch.uint8)
        with self.assertRaises(RuntimeError):
            f.common._cuda_host_register(tensor)
        self.assertEqual(
            getattr(tensor, f.common._CUDA_HOST_REGISTERED_RANGES_ATTR), []
        )

    def test_rollback_reports_unregister_failure_and_keeps_owner(self):
        f = self.f
        b = f.budget.HostAllocationBudget(GIB, 0, 0)
        tensor = f.common.alloc_with_host_register(
            (10,), f.torch.uint8, "cpu", True, f.common.HostTensorAllocator()
        )
        b.record_buffer(tensor)
        f.cuda.fail_unregister = True
        with self.assertRaisesRegex(RuntimeError, "rollback failed"):
            b.rollback()
        self.assertEqual(len(b.buffers), 1)

    def test_unsupported_topology_rejected_before_collective(self):
        for name, value in (("nnodes", 2), ("dcp_enabled", True)):
            f = fixture()
            setattr(f.parallel, name, value)
            with self.assertRaises(NotImplementedError):
                with f.budget.dsa_host_allocation_budget(NS(pp_size=1)):
                    pass
        with self.assertRaises(NotImplementedError):
            with self.f.budget.dsa_host_allocation_budget(NS(pp_size=2)):
                pass

    def test_nested_transaction_does_not_issue_collective(self):
        f = self.f
        with self.assertRaisesRegex(RuntimeError, "already active"):
            with f.budget.dsa_host_allocation_budget(NS(pp_size=1)):
                with f.budget.dsa_host_allocation_budget(NS(pp_size=1)):
                    pass

    def test_opt_in_disabled_preserves_ordinary_legacy_path(self):
        f = self.f
        f.env.enabled = False
        f.parallel.nnodes = 2
        with f.budget.dsa_host_allocation_budget(
            NS(pp_size=2), storage_backend="mooncake"
        ) as b:
            self.assertIsNone(b)
        f.env.value = "2MB"
        with self.assertRaisesRegex(
            ValueError, "requires SGLANG_HICACHE_DSA_STARTUP_BUDGET"
        ):
            with f.budget.dsa_host_allocation_budget(NS(pp_size=1)):
                pass

    def test_opt_in_zero_preserves_legacy_ten_gib_reserve(self):
        f = self.f
        f.env.enabled = False
        self.assertEqual(f.base.HICACHE_HOST_MEMORY_RESERVE_BYTES, 10 * GIB)
        self.assertEqual(f.budget.host_memory_reserve_bytes(), 128 * GIB)
        f.psutil.virtual_memory = lambda: NS(available=100 * GIB)
        with f.budget.dsa_host_allocation_budget(NS(pp_size=1)) as b:
            self.assertIsNone(b)
            self.assertEqual(f.base.host_memory_budget_bytes(), 90 * GIB)

    def test_external_l3_and_buffer_only_fail_closed_in_budget_mode(self):
        for kwargs in (
            {"storage_backend": "mooncake"},
            {"storage_backend": "dynamic"},
            {"storage_backend": "shm"},
            {"host_memory_mode": "buffer_only"},
        ):
            with self.assertRaisesRegex(NotImplementedError, "external L3/segment"):
                with self.f.budget.dsa_host_allocation_budget(NS(pp_size=1), **kwargs):
                    pass

    def test_proc_sys_snapshot_reservations_cgroup_and_numa(self):
        f = self.f
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def put(path, text):
                dest = root / path.lstrip("/")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(str(text), encoding="utf-8")

            put("/proc/self/cgroup", "0::/job\n")
            put(
                "/proc/self/mountinfo",
                "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
            )
            put("/proc/self/status", "Mems_allowed_list:\t0-1\n")
            put("/sys/devices/system/node/online", "0-1\n")
            for path, val in {
                "memory.max": 256 * GIB,
                "memory.current": 8 * GIB,
                "hugetlb.2MB.max": 30 * 1024**2,
                "hugetlb.2MB.current": 2 * 1024**2,
                "hugetlb.2MB.rsvd.max": 20 * 1024**2,
                "hugetlb.2MB.rsvd.current": 2 * 1024**2,
            }.items():
                put("/sys/fs/cgroup/job/" + path, val)
            put("/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages", 20)
            put("/sys/kernel/mm/hugepages/hugepages-2048kB/resv_hugepages", 4)
            f.budget.Path = lambda value: root / str(value).lstrip("/")
            f.psutil.virtual_memory = lambda: NS(available=320 * GIB)
            self.assertEqual(f.real_snapshot(PAGE), (120 * GIB, 18 * 1024**2))
            put("/proc/self/status", "Mems_allowed_list:\t0\n")
            put(
                "/sys/devices/system/node/node0/hugepages/hugepages-2048kB/free_hugepages",
                8,
            )
            self.assertEqual(f.real_snapshot(PAGE)[1], 4 * PAGE)
            put("/sys/kernel/mm/hugepages/hugepages-2048kB/resv_hugepages", 21)
            self.assertEqual(f.real_snapshot(PAGE)[1], 0)
            f.psutil.virtual_memory = lambda: NS(available=128 * GIB)
            self.assertEqual(f.real_snapshot(PAGE)[0], 0)

    def run_workers(self, mode, fail_rank=None):
        with mp.Manager() as manager:
            shared, output = manager.dict(), manager.dict()
            barrier = manager.Barrier(8)
            processes = [
                mp.Process(
                    target=worker, args=(rank, shared, barrier, output, mode, fail_rank)
                )
                for rank in range(8)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=35)
                if process.is_alive():
                    process.terminate()
                self.assertEqual(process.exitcode, 0)
            results = dict(output)
        self.assertEqual(len(results), 8)
        for result in results.values():
            self.assertNotIn("worker_error", result, result)
        return results

    def test_eight_process_ordinary_feasible_interleaving(self):
        for result in self.run_workers("ordinary").values():
            self.assertIsNone(result["error"], result)
            self.assertEqual(result["registered"], 2)

    def test_eight_process_true_oversubscription_rolls_back(self):
        for result in self.run_workers("oversubscribed").values():
            self.assertIn("DSA host allocation failed", result["error"])
            self.assertEqual(result["registered"], 0)
            self.assertEqual(result["remaining"], result["initial"])

    def test_eight_process_huge_success_with_low_ordinary_memory(self):
        for result in self.run_workers("huge").values():
            self.assertIsNone(result["error"], result)
            self.assertTrue(all(event[0] == "huge" for event in result["events"]))

    def test_eight_process_one_hugepage_short_rejected(self):
        for result in self.run_workers("huge_short").values():
            self.assertIn("DSA host allocation failed", result["error"])
            self.assertEqual(result["registered"], 0)

    def test_one_rank_pin_failure_rolls_back_every_rank(self):
        for result in self.run_workers("ordinary", fail_rank=3).values():
            self.assertIn("cudaHostRegister failed", result["error"])
            self.assertEqual(result["registered"], 0)
            self.assertTrue(all(result["destroyed"]))


if __name__ == "__main__":
    mp.freeze_support()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CandidateTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
        "scope": "pure CPU; integer simulation; exact candidate control flow; no real mmap or GPU",
        "candidate_sha256": {
            str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(SOURCE.rglob("*.py"))
        },
    }
    Path(
        os.environ.get("SGLANG_TEST_RESULT_PATH", "/tmp/candidate-test-results.json")
    ).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sys.exit(0 if result.wasSuccessful() else 1)
