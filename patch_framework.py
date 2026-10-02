"""Integrate only framework logging configuration into installed startup sites."""
import ast
import hashlib
import json
from pathlib import Path
import sys

root = Path("/opt/phala-source/python/sglang/srt")
rows = []
import_line = "from sglang.srt.utils.framework_log_privacy import configure_framework_log_privacy"
call_line = "configure_framework_log_privacy()"
for path in sorted(root.rglob("*.py")):
    if path.name == "framework_log_privacy.py":
        continue
    before = path.read_text()
    source = before.lstrip("\ufeff")
    tree = ast.parse(source)
    starts = set()
    granian_starts = set()
    insertion_indents = {}
    for node in ast.walk(tree):
        # Locate statements that construct/configure an actual Uvicorn server.
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return)):
            continue
        for call in ast.walk(node):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "uvicorn"
                    and call.func.attr in {"Config", "run"}):
                starts.add(node.lineno)
        if (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "server" for target in node.targets)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "Server" and any(keyword.arg is None and
                isinstance(keyword.value, ast.Name) and keyword.value.id == "granian_kwargs"
                for keyword in node.value.keywords)):
            granian_starts.add(node.lineno)
    # Main SGLang helper must finish path-filter setup before privacy filtering.
    if path.relative_to(root).as_posix() == "utils/common.py":
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "set_uvicorn_logging_configs":
                starts.add(node.end_lineno + 1)
                insertion_indents[node.end_lineno + 1] = " " * node.body[-1].col_offset
    if not starts and not granian_starts:
        continue
    lines = source.splitlines(keepends=True)
    for lineno in sorted(starts | granian_starts, reverse=True):
        if lineno in insertion_indents:
            indent = insertion_indents[lineno]
        elif lineno > len(lines):
            indent = "    "
        else:
            target = lines[lineno - 1]
            indent = target[:len(target) - len(target.lstrip(" \t"))]
            if path.name == "common.py" and not indent:
                indent = "    "
        if lineno in granian_starts:
            lines[lineno-1:lineno-1] = [
                indent + "from sglang.srt.utils.framework_log_privacy import configure_granian_log_privacy\n",
                indent + "granian_kwargs['log_dictconfig'] = configure_granian_log_privacy()\n"]
        else:
            lines[lineno-1:lineno-1] = [indent + import_line + "\n", indent + call_line + "\n"]
    after = ("\ufeff" if before.startswith("\ufeff") else "") + "".join(lines)
    compile(after.encode(), str(path), "exec")
    if path.relative_to(root).as_posix() == "utils/common.py":
        installed_tree = ast.parse(after.lstrip("\ufeff"))
        function = next(node for node in installed_tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "set_uvicorn_logging_configs")
        path_calls = [node.lineno for node in ast.walk(function) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == "_configure_uvicorn_access_log_filter"]
        privacy_calls = [node.lineno for node in ast.walk(function) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == "configure_framework_log_privacy"]
        assert len(privacy_calls) == 1 and path_calls and privacy_calls[0] > max(path_calls)
    path.write_text(after)
    rows.append({"file": str(path.relative_to(root)), "integration_lines": sorted(starts),
                 "granian_integration_lines": sorted(granian_starts),
                 "base_sha256": hashlib.sha256(before.encode()).hexdigest(),
                 "final_sha256": hashlib.sha256(after.encode()).hexdigest()})
if not rows:
    raise SystemExit("No framework startup integrations found")
print(json.dumps({"files": rows}, indent=2))
