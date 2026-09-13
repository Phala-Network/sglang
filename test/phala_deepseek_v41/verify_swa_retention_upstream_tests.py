"""Tests against installed source-integrated adapters; no runtime package mounts."""
import collections, os, re, sys, unittest

sys.path.insert(0, "/sgl-workspace/sglang/test/registered/unit/mem_cache")
import test_unified_radix_cache_unittest as T  # noqa: E402

T.get_device = lambda *a, **k: "cpu"

patched = "--patched" in sys.argv
args = [a for a in sys.argv[1:] if a != "--patched"]
pattern = re.compile(args[0] if args else r"swa|evict|lru")
if patched:
    import sglang.srt.mem_cache.unified_cache.components.swa_component as swa_mod
    import dsv41_swa_retention as R
    os.environ[R.ENV_PARTS] = "finish,evict"
    R.apply(swa_mod)

loader = unittest.TestLoader()
suite = unittest.TestSuite()
for name in dir(T):
    obj = getattr(T, name)
    if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
        for tname in loader.getTestCaseNames(obj):
            if pattern.search(tname):
                suite.addTest(obj(tname))
label = "PATCHED" if patched else "UPSTREAM"
result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w")).run(suite)
show = set(filter(None, os.environ.get("SHOW_TRACEBACK", "").split(",")))
for kind, items in (("FAIL", result.failures), ("ERROR", result.errors)):
    for t, tb in items:
        print(f"NONPASS {kind} {t.id()}")
        if t.id() in show:
            print("TRACEBACK", t.id(), "\n" + tb[-3000:])
nonpass = collections.Counter(t.id().split(".")[1] for t, _ in result.failures + result.errors)
total = collections.Counter(t.id().split(".")[1] for t in suite)
skipped = collections.Counter(t.id().split(".")[1] for t, _ in result.skipped)
for cls in sorted(total):
    if "SWA" in cls:
        print(f"CLASS {cls}: total={total[cls]} nonpass={nonpass[cls]} skipped={skipped[cls]}")
print(f"RESULT {label}: run={result.testsRun} failures={len(result.failures)} errors={len(result.errors)} skipped={len(result.skipped)}")
