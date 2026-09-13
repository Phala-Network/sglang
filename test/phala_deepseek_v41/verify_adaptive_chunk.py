"""Tests against installed source-integrated adapters; no runtime package mounts."""
import os, sys, types, inspect

FAIL = 0
def check(label, got, want=True):
    global FAIL
    ok = (got == want)
    FAIL += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + ("" if ok else f": {got!r} != {want!r}"))

import sglang.srt.managers.scheduler as sched_mod
Scheduler = sched_mod.Scheduler

print("upstream call site still exists")
src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
check("reads self.dynamic_chunk_sizer", "self.dynamic_chunk_sizer is not None" in src)
check("keys off len(self.chunked_req.prefix_indices)",
      "history_len = len(self.chunked_req.prefix_indices)" in src)
check("calls .predict(history_len)", ".predict(history_len)" in src)
check("overrides chunked_prefill_size", "chunked_prefill_size = dynamic_size" in src)
_init = Scheduler.maybe_init_dynamic_chunk_sizer
_init = getattr(_init, "__wrapped__", _init)   # read upstream even when wrapped
init_src = inspect.getsource(_init)
check("upstream initialiser is pp-gated", "pp_size > 1" in init_src)
check("upstream initialiser assigns the slot", "self.dynamic_chunk_sizer" in init_src)
check("the adder receives it as the chunk budget",
      "chunked_prefill_size," in src and "PrefillAdder(" in src)

print(f"env {'SET' if os.environ.get('DSV41_ADAPTIVE_CHUNK') else 'UNSET'}: wrapper presence")
wrapped = getattr(Scheduler.maybe_init_dynamic_chunk_sizer, "_dsv41_adaptive_chunk", False)
want_wrapped = bool(os.environ.get("DSV41_ADAPTIVE_CHUNK"))
check("maybe_init_dynamic_chunk_sizer wrapped == env set", wrapped, want_wrapped)

if not want_wrapped:
    print("       (upstream untouched, as intended)")
    sys.exit(1 if FAIL else 0)

from sglang.srt.phala_compat import dsv41_adaptive_chunk as ac

print("drive the wrapper on a stub Scheduler")
sched_mod.get_schedule = lambda: types.SimpleNamespace(enable_dynamic_chunking=False)
stub = object.__new__(Scheduler)
stub.ps = types.SimpleNamespace(pp_size=1)
stub.chunked_prefill_size = 16384
stub.page_size = 256
Scheduler.maybe_init_dynamic_chunk_sizer(stub)
check("AdaptiveChunkSizer installed", isinstance(stub.dynamic_chunk_sizer, ac.AdaptiveChunkSizer))

sizer = stub.dynamic_chunk_sizer
budget = sizer.budget_bytes
print(f"       budget {budget/(1<<30):.1f} GiB (pinned via env), base {sizer.base_chunk_size}")
check("short prompt keeps the static chunk", sizer.predict(0), None)
check("14,146-token prompt keeps it", sizer.predict(14146), None)
check("121,660-token prompt keeps it", sizer.predict(121660), None)
long_chunk = sizer.predict(900_000)
check("900K history shrinks", isinstance(long_chunk, int) and 2048 <= long_chunk < 16384)
check("shrunk value is page aligned", long_chunk % 256, 0)
check("budget respected at 900K",
      ac.DEFAULT_BYTES_PER_TOKEN_PAIR * long_chunk * (900_000 + long_chunk) <= budget)

print("pp_size > 1 path stays upstream's")
check("the original is still reachable via __wrapped__",
      getattr(Scheduler.maybe_init_dynamic_chunk_sizer, "__wrapped__", None) is not None)
# inspect.getsource follows __wrapped__, so read the wrapper out of apply()'s
# own source; the behavioural version of this check is in the module selftest
# ("an upstream PP sizer is not displaced"), where the original does fill the slot.
check("the wrapper returns early when the slot is already filled",
      'if getattr(self, "dynamic_chunk_sizer", None) is not None:'
      in inspect.getsource(ac.apply))

print()
print(f"{'FAILED: %d check(s)' % FAIL if FAIL else 'all checks passed'}")
sys.exit(1 if FAIL else 0)
