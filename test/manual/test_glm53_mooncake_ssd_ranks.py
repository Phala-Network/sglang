"""Finite CPU-only path regressions; extracts exact changed source, no serving imports."""

import ast
import hashlib
import json
import os
import posixpath
import sys
import textwrap
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
installed_root = os.environ.get("SGLANG_TEST_INSTALLED_ROOT")
candidate = (
    Path(installed_root) if installed_root else HERE.parents[1]
) / "python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py"


text = candidate.read_text()
tree = ast.parse(text)
compile(tree, str(candidate), "exec")
helper = next(
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "_ssd_offload_path_for_rank"
)
created = []
logs = []
rank_calls = []
world = types.SimpleNamespace(rank=0, world_size=8)
fake_os = types.SimpleNamespace(
    path=types.SimpleNamespace(
        isabs=posixpath.isabs,
        normpath=posixpath.normpath,
        join=posixpath.join,
        islink=lambda p: False,
    ),
    sep="/",
    makedirs=lambda path, exist_ok: created.append((path, exist_ok)),
)
ns = {"os": fake_os}
exec(
    compile(
        ast.Module(body=[helper], type_ignores=[]), "<exact-source-helper>", "exec"
    ),
    ns,
)
modules = {
    name: types.ModuleType(name)
    for name in [
        "sglang",
        "sglang.srt",
        "sglang.srt.distributed",
        "sglang.srt.distributed.parallel_state",
    ]
}


def get_world_group():
    rank_calls.append(world.rank)
    return world


modules["sglang.srt.distributed.parallel_state"].get_world_group = get_world_group
prior = {name: sys.modules.get(name) for name in modules}
sys.modules.update(modules)
start = text.index("                setup_kwargs = {}")
end = text.index(
    "\n                ",
    text.index(
        'setup_kwargs["ssd_offload_path"] = self.config.ssd_offload_path', start
    ),
)
fragment = compile(textwrap.dedent(text[start:end]), str(candidate), "exec")
checks = []


def record(name, ok):
    assert ok, name
    checks.append({"check": name, "pass": True})


def run(enabled, path, rank=0):
    created.clear()
    logs.clear()
    rank_calls.clear()
    world.rank = rank
    scope = dict(
        ns,
        self=types.SimpleNamespace(
            config=types.SimpleNamespace(
                enable_ssd_offload=enabled, ssd_offload_path=path
            )
        ),
        storage_config=types.SimpleNamespace(tp_rank=0, tp_size=1),
        logger=types.SimpleNamespace(info=lambda *args: logs.append(args)),
    )
    exec(fragment, scope)
    return scope["setup_kwargs"]


try:
    paths = []
    for rank in range(8):
        result = run(True, "/workspace/data/mooncake-ssd", rank)
        path = result["ssd_offload_path"]
        paths.append(path)
        record(
            "global rank "
            + str(rank)
            + " gets independent path despite local attention tp_rank0",
            result
            == {
                "enable_ssd_offload": True,
                "ssd_offload_path": "/workspace/data/mooncake-ssd/rank-" + str(rank),
            }
            and created == [(path, True)]
            and rank_calls == [rank]
            and len(logs) == 1,
        )
    record("all8 physical worker directories unique", len(set(paths)) == 8)
    record(
        "restart-stable path has no PID dependence",
        run(True, "/workspace/data/mooncake-ssd", 5)["ssd_offload_path"] == paths[5],
    )
    for path in (None, "", "relative-ignored-while-disabled", "/old/configured/path"):
        result = run(False, path)
        record(
            "disabled behavior retained for " + repr(path),
            result == ({} if path is None else {"ssd_offload_path": path})
            and not created
            and not rank_calls,
        )
    for path in (None, "", "relative/path", "/cache/../another"):
        try:
            run(True, path)
        except ValueError:
            record("enabled rejects unsafe base " + repr(path), not created)
        else:
            raise AssertionError(path)
    for rank in (-1, True, "0"):
        try:
            run(True, "/cache/ssd", rank)
        except ValueError:
            record("invalid world rank rejected " + repr(rank), not created)
        else:
            raise AssertionError(rank)
    fake_os.path.islink = lambda path: True
    try:
        run(True, "/cache/ssd", 0)
    except ValueError:
        record("existing symlink target rejected before native setup", True)
    else:
        raise AssertionError("symlink")
finally:
    for name, value in prior.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value

result = {
    "status": "INSTALLED_SOURCE_CPU_PATH_CHECKS_PASS"
    if installed_root
    else "LOCAL_SOURCE_CPU_PATH_CHECKS_PASS",
    "checks_passed": len(checks),
    "checks": checks,
    "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    "serving_runtime_imported": False,
    "model_requests": 0,
    "native_SSD_operations": 0,
    "remote_calls": 0,
    "scope": "Exact helper and changed setup block; installed global-rank accessor is a CPU collaborator. No model/native runtime import.",
}

print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
