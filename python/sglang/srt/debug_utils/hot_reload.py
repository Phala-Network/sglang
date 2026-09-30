"""Debug-only hot reload: run a patch script inside a live scheduler.

With SGLANG_DEBUG_HOT_RELOAD_SCRIPT=/path/patch.py set, SIGUSR2 to a scheduler
process runs that script at the start of its next loop iteration, with
`scheduler` and `reload_function` in its globals. Send it to every scheduler
process (`pkill -USR2 -f sglang::scheduler`): ranks reload independently.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import signal
import sys
import traceback
from typing import Any, Callable, Optional

from sglang.srt.debug_utils.source_patcher.code_patcher import _resolve_target

logger = logging.getLogger(__name__)


def reload_function(qualified_name: str) -> None:
    """Recompile `pkg.mod.Class.method` from the source now on disk and swap its
    code in place, so every existing reference runs the new body."""
    target: Callable[..., Any] = inspect.unwrap(_resolve_target(qualified_name))
    module = sys.modules[target.__module__]
    path = inspect.getsourcefile(module)
    with open(path) as f:
        scope: Any = ast.parse(f.read(), path)
    # Keep only the path to the function, undecorated and with no class bases, so
    # compiling it runs nothing else yet methods still get their `__class__` cell.
    names = target.__qualname__.split(".")
    root = None
    for name in names:
        node = next(
            n
            for n in scope.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == name
        )
        node.decorator_list = []
        if isinstance(node, ast.ClassDef):
            node.bases, node.keywords = [], []
        if root is None:
            root = node
        else:
            scope.body = [node]
        scope = node
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module([root], []), path, "exec"), vars(module), namespace)
    fresh: Any = namespace[names[0]]
    for name in names[1:]:
        fresh = vars(fresh)[name]
    target.__code__ = fresh.__code__


class HotReloader:
    def __init__(self, script: str) -> None:
        self.script = script
        self.pending = False
        signal.signal(signal.SIGUSR2, self._request)
        logger.warning("Debug hot reload armed: SIGUSR2 runs %s", script)

    def _request(self, signum: int, frame: Any) -> None:
        self.pending = True

    def poll(self, scheduler: Any) -> None:
        if not self.pending:
            return
        self.pending = False
        try:
            with open(self.script, "rb") as f:
                source = f.read()
            digest = hashlib.sha256(source).hexdigest()[:16]
            exec(
                compile(source, self.script, "exec"),
                {"scheduler": scheduler, "reload_function": reload_function},
            )
            logger.warning("Hot reload applied: %s sha256=%s", self.script, digest)
        except Exception:
            logger.error(
                "Hot reload failed: %s\n%s", self.script, traceback.format_exc()
            )


def maybe_create_hot_reloader(script: Optional[str]) -> Optional[HotReloader]:
    return HotReloader(script) if script else None
