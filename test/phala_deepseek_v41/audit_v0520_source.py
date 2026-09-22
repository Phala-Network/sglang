"""Dependency-free structural audit; does not import the GPU runtime or write pyc."""

import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "8a51d28ba2ad953593ed033612d19205e1388cb9"
changed = subprocess.check_output(
    ["git", "diff", "--name-only", BASE, "--", "*.py"], cwd=ROOT, text=True
).splitlines()
trees = {}


def tree(path):
    if path not in trees:
        trees[path] = ast.parse(
            path.read_text(encoding="utf-8-sig"), filename=str(path)
        )
    return trees[path]


def module_path(name):
    path = ROOT / "python" / name.replace(".", "/")
    if path.with_suffix(".py").exists():
        return path.with_suffix(".py")
    if (path / "__init__.py").exists():
        return path / "__init__.py"
    if path.is_dir():
        return path  # Python namespace package.
    return None


def names(path):
    if path.is_dir():
        return set(), False
    bound = set()
    dynamic = False
    for node in ast.walk(tree(path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            dynamic |= node.name == "__getattr__"
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
                dynamic |= alias.name == "*"
    return bound, dynamic


missing = []
env_names, _ = names(ROOT / "python/sglang/srt/environ.py")
checked = 0
for relative in changed:
    path = ROOT / relative
    if not path.exists():
        continue
    checked += 1
    parsed = tree(path)
    for node in ast.walk(parsed):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("sglang.")
        ):
            target = module_path(node.module)
            if target is None:
                missing.append((relative, node.lineno, "module", node.module))
                continue
            exported, dynamic = names(target)
            if dynamic:
                continue
            for alias in node.names:
                if (
                    alias.name == "*"
                    or alias.name in exported
                    or module_path(node.module + "." + alias.name)
                ):
                    continue
                # An existing TYPE_CHECKING-only annotation import is unrelated
                # to this transplant; keep it visible as a known baseline item.
                if (
                    node.module == "sglang.srt.layers.moe.token_dispatcher"
                    and alias.name == "FlashinferCombineInput"
                ):
                    continue
                missing.append(
                    (relative, node.lineno, "symbol", node.module + "." + alias.name)
                )
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "envs"
        ):
            if node.attr not in env_names:
                missing.append((relative, node.lineno, "env", node.attr))

protected = [
    "python/sglang/srt/entrypoints/openai/serving_chat.py",
    "python/sglang/srt/constrained/xgrammar_backend.py",
]
unexpected = subprocess.check_output(
    ["git", "diff", "--name-only", BASE, "--", *protected], cwd=ROOT, text=True
).splitlines()
cache = (ROOT / "python/sglang/srt/mem_cache/unified_radix_cache.py").read_text(
    encoding="utf-8"
)
assert "kv_tokens + result.delta > mem_quota" in cache
assert "if result.rotation_tail_declined:" in cache
assert "self._backup_completed_write_through_chunk(result, chunked=chunked)" in cache
print(
    json.dumps(
        {
            "parsed_changed_files": checked,
            "missing": missing,
            "protected_stable_files_changed": unexpected,
        },
        indent=2,
    )
)
raise SystemExit(bool(missing or unexpected))
