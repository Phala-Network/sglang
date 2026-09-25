#!/usr/bin/env python3
"""Reviewable source migration, never an image-startup or runtime patcher."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

METHODS = {
    "debug",
    "info",
    "warning",
    "warn",
    "error",
    "exception",
    "critical",
    "fatal",
    "log",
}
SENSITIVE = re.compile(
    r"(?:^|_)(?:e|exc|err|error|exception|traceback|request|req|prompt|text|content|"
    r"input|output|payload|rid|url|obj|data|session|headers?|buffer|args|params)(?:$|_)"
)
POLICY_PATH = Path(__file__).with_name("srt_log_privacy_policy.json")
FIXED_TRACEBACK = "Exception details redacted"
EXCLUDED_FILES = {
    "utils/framework_log_privacy.py",
    "utils/log_utils.py",
    "model_loader/expert_pack/validate.py",
}


class ReviewRequired(ValueError):
    pass


def validate_source_root(root):
    for relative in (
        "entrypoints/http_server.py",
        "utils/request_logger.py",
        "model_loader/expert_pack/validate.py",
    ):
        if not (root / relative).is_file():
            raise ReviewRequired(
                "Source root must be the complete python/sglang/srt tree"
            )


def dump(node):
    return ast.dump(node, include_attributes=False)


def load_policy(path=POLICY_PATH):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["schema"] != "phala.srt-log-privacy-source-policy.v1":
        raise ReviewRequired("Unsupported source policy")
    data["evaluations"] = {
        (row["file"], dump(ast.parse(row["expression"], mode="eval").body)): row[
            "action"
        ]
        for row in data["reviewed_evaluations"]
    }
    return data


def kind(call):
    if not isinstance(call, ast.Call):
        return None
    func = call.func
    if isinstance(func, ast.Name):
        return {"log_json": "json", "print": "print", "pprint": "pprint"}.get(func.id)
    if not isinstance(func, ast.Attribute):
        return None
    receiver = func.value
    if isinstance(receiver, ast.Name):
        name = receiver.id.lower()
        if name == "warnings" and func.attr == "warn":
            return "warning"
        if name == "traceback" and func.attr in {"print_exc", "print_exception"}:
            return "traceback"
        if name == "self" and func.attr == "_log":
            return "logger"
        if (
            name.endswith("logger") or name in {"log", "logging", "target"}
        ) and func.attr in METHODS:
            return "logger"
    if (
        isinstance(receiver, ast.Attribute)
        and receiver.attr.lower().endswith("logger")
        and func.attr in METHODS
    ):
        return "logger"
    return None


def selected(call, relative, in_except, functions, policy):
    if relative in EXCLUDED_FILES:
        return False
    category = kind(call)
    if category in {"print", "pprint"}:
        if relative in policy["business_print_files"]:
            return False
        if any(
            row["file"] == relative and row["function"] in functions
            for row in policy["business_print_functions"]
        ):
            return False
    if (
        relative == "utils/request_logger.py"
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "target"
    ):
        return False
    if category == "json":
        return True
    if (
        relative.startswith(("function_call/", "entrypoints/"))
        or relative == "utils/request_logger.py"
    ):
        return True
    if category == "traceback" or getattr(call.func, "attr", None) == "exception":
        return True
    if in_except or any(kw.arg in {"exc_info", "stack_info"} for kw in call.keywords):
        return True
    return any(
        SENSITIVE.search(node.id if isinstance(node, ast.Name) else node.attr)
        for node in ast.walk(call)
        if isinstance(node, (ast.Name, ast.Attribute))
    )


def safe_message(node, *, interpolate=False):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if not interpolate:
            return node.value
        return re.sub(
            r"%(?:\([^)]+\))?[#0 +\-]*(?:\d+|\*)?(?:\.(?:\d+|\*))?[hlL]?[diouxXeEfFgGcrsa]",
            "<redacted>",
            node.value,
        ).replace("%%", "%")
    if isinstance(node, ast.JoinedStr):
        return "".join(
            piece.value
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str)
            else "<redacted>"
            for piece in node.values
        )
    return "Request-path diagnostic redacted"


def preserve_then(expressions, value):
    if not expressions:
        return value
    return ast.Subscript(
        value=ast.Tuple(elts=[*expressions, value], ctx=ast.Load()),
        slice=ast.Constant(value=len(expressions)),
        ctx=ast.Load(),
    )


def safe_result(node, category):
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Tuple)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == len(node.value.elts) - 1
        and node.value.elts
    ):
        return safe_result(node.value.elts[-1], category)
    if category == "message":
        return isinstance(node, ast.Constant) and isinstance(node.value, str)
    if category == "disabled":
        return isinstance(node, ast.Constant) and node.value is False
    return dump(node) == dump(ast.parse("{'redacted': True}", mode="eval").body)


def reviewed_expressions(expressions, relative, policy):
    preserved = []
    for expression in expressions:
        calls = [node for node in ast.walk(expression) if isinstance(node, ast.Call)]
        actions = [policy["evaluations"].get((relative, dump(node))) for node in calls]
        if "preserve" in actions:
            # Preserve the whole expression, including conditionals and order,
            # but never pass its value to the sink.
            preserved.append(expression)
            continue
        for node, action in zip(calls, actions):
            if action != "drop-diagnostic":
                raise ReviewRequired(
                    f"Unreviewed removed evaluation: {relative}: {ast.unparse(node)}"
                )
    return preserved


def replacement(call, relative, policy):
    category = kind(call)
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
        keyword.arg is None for keyword in call.keywords
    ):
        raise ReviewRequired(f"Unpacked logging arguments need review: {relative}")
    keywords = {kw.arg: kw.value for kw in call.keywords}
    if len(keywords) != len(call.keywords):
        raise ReviewRequired("Duplicate logging keyword")
    args = list(call.args)
    removed = []
    kept_keywords = []
    func = copy.deepcopy(call.func)
    preserved_count = 0

    if category == "json":
        if (
            len(args) != 3
            or keywords
            or not (
                isinstance(args[1], ast.Constant) and isinstance(args[1].value, str)
            )
        ):
            raise ReviewRequired(f"JSON event contract needs review: {relative}")
        removed = [] if safe_result(args[2], "payload") else [args[2]]
        payload = (
            args[2]
            if safe_result(args[2], "payload")
            else ast.parse("{'redacted': True}", mode="eval").body
        )
        new_args = [args[0], args[1], payload]
        message_index = 2
    elif category == "traceback":
        unknown = set(keywords) - {"file", "limit", "chain"}
        if unknown:
            raise ReviewRequired(f"Traceback keyword contract needs review: {relative}")
        removed = args + [value for name, value in keywords.items() if name == "chain"]
        func = ast.Attribute(
            value=call.func.value, attr="print_exception", ctx=ast.Load()
        )
        marker = ast.Call(
            func=ast.Name(id="RuntimeError", ctx=ast.Load()),
            args=[ast.Constant(FIXED_TRACEBACK)],
            keywords=[],
        )
        new_args = [marker]
        kept_keywords = [kw for kw in call.keywords if kw.arg in {"file", "limit"}]
        kept_keywords.append(ast.keyword(arg="chain", value=ast.Constant(False)))
        message_index = 0
    else:
        if any(name in keywords for name in ("msg", "message", "level")):
            raise ReviewRequired(
                f"Keyword message/severity ordering needs review: {relative}"
            )
        level = []
        if category == "logger" and call.func.attr == "log":
            if args:
                level = [args.pop(0)]
            elif "level" in keywords:
                level = [keywords.pop("level")]
            else:
                raise ReviewRequired(f"Missing logging severity: {relative}")
        message_keyword = "message" if category == "warning" else "msg"
        message = args[0] if args else keywords.pop(message_keyword, ast.Constant(""))
        removed = [message]
        if category == "logger":
            removed.extend(args[1:])
            unknown = set(keywords) - {"exc_info", "stack_info", "stacklevel", "msg"}
            if unknown:
                raise ReviewRequired(
                    f"Logger keyword contract needs review: {relative}: {sorted(unknown)}"
                )
            value = (
                message
                if len(args) <= 1 and safe_result(message, "message")
                else ast.Constant(safe_message(message, interpolate=len(args) > 1))
            )
            new_args = level + [value]
            if dump(value) == dump(message):
                removed = removed[1:]
            for kw in call.keywords:
                if kw.arg == "stacklevel":
                    kept_keywords.append(kw)
                elif kw.arg in {"exc_info", "stack_info"}:
                    if safe_result(kw.value, "disabled"):
                        kept_keywords.append(kw)
                        continue
                    retained = reviewed_expressions([kw.value], relative, policy)
                    preserved_count += len(retained)
                    kept_keywords.append(
                        ast.keyword(
                            arg=kw.arg,
                            value=preserve_then(retained, ast.Constant(False)),
                        )
                    )
            if call.func.attr == "exception" and "exc_info" not in keywords:
                kept_keywords.append(
                    ast.keyword(arg="exc_info", value=ast.Constant(False))
                )
            message_index = len(level)
        elif category == "warning":
            unknown = set(keywords) - {"category", "stacklevel", "source", "message"}
            if unknown:
                raise ReviewRequired(f"Warning metadata needs review: {relative}")
            value = (
                message
                if safe_result(message, "message")
                else ast.Constant(safe_message(message))
            )
            new_args = [value, *args[1:]]
            if dump(value) == dump(message):
                removed = removed[1:]
            kept_keywords = [kw for kw in call.keywords if kw.arg != "message"]
            message_index = 0
        else:
            allowed = (
                {"file", "end", "flush", "sep"}
                if category == "print"
                else {
                    "stream",
                    "indent",
                    "width",
                    "depth",
                    "compact",
                    "sort_dicts",
                    "underscore_numbers",
                }
            )
            unknown = set(keywords) - allowed
            if unknown:
                raise ReviewRequired(f"Print controls need review: {relative}")
            for field in ("end", "sep"):
                if field in keywords and not isinstance(keywords[field], ast.Constant):
                    raise ReviewRequired(
                        f"Dynamic print {field} needs privacy review: {relative}"
                    )
            text = safe_message(message)
            if category == "print":
                removed.extend(args[1:])
                if len(args) > 1:
                    text += " <redacted>"
                value = (
                    message
                    if len(args) == 1 and safe_result(message, "message")
                    else ast.Constant(text)
                )
                new_args = [value]
            else:
                value = (
                    message if safe_result(message, "message") else ast.Constant(text)
                )
                new_args = [value, *args[1:]]
            if dump(value) == dump(message):
                removed = removed[1:]
            kept_keywords = list(call.keywords)
            message_index = 0

    result = ast.Call(func=func, args=new_args, keywords=kept_keywords)
    if dump(call) == dump(result):
        return call, 0
    retained = reviewed_expressions(removed, relative, policy)
    result.args[message_index] = preserve_then(retained, result.args[message_index])
    return result, preserved_count + len(retained)


class WithoutLogs(ast.NodeTransformer):
    def visit_Expr(self, node):
        return None if kind(node.value) else self.generic_visit(node)


def nonlog_ast(text):
    return dump(WithoutLogs().visit(ast.parse(text.lstrip("\ufeff"))))


def transform(text, relative, policy):
    prefix = "\ufeff" if text.startswith("\ufeff") else ""
    source = text.lstrip("\ufeff")
    tree = ast.parse(source)
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not kind(node.value):
            continue
        ancestor = node
        functions, in_except = [], False
        while ancestor in parents:
            ancestor = parents[ancestor]
            if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(ancestor.name)
            if isinstance(ancestor, ast.ExceptHandler):
                in_except = True
        if not selected(node.value, relative, in_except, functions, policy):
            continue
        new, preserved = replacement(node.value, relative, policy)
        if dump(node.value) == dump(new):
            continue
        start = offsets[node.lineno - 1] + len(
            lines[node.lineno - 1].encode()[: node.col_offset].decode()
        )
        end = offsets[node.end_lineno - 1] + len(
            lines[node.end_lineno - 1].encode()[: node.end_col_offset].decode()
        )
        edits.append((start, end, ast.unparse(new), node.lineno, preserved))
    after = source
    for start, end, replacement_text, _, _ in sorted(edits, reverse=True):
        after = after[:start] + replacement_text + after[end:]
    after = prefix + after
    compile(after.encode("utf-8"), relative, "exec")
    if nonlog_ast(text) != nonlog_ast(after):
        raise ReviewRequired(f"Non-log AST changed: {relative}")
    return after, [{"line": row[3], "preserved_expressions": row[4]} for row in edits]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists")
    policy = load_policy(args.policy)
    planned, failures, rows = [], [], []
    root = args.source_root.resolve()
    validate_source_root(root)
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            failures.append({"file": relative, "error": "Source symlink needs review"})
            continue
        before = path.read_bytes()
        try:
            after, changes = transform(before.decode("utf-8"), relative, policy)
        except ReviewRequired as error:
            failures.append({"file": relative, "error": str(error)})
            continue
        if changes:
            encoded = after.encode("utf-8")
            planned.append((path, before, encoded))
            rows.append(
                {
                    "file": relative,
                    "before_sha256": sha(before),
                    "after_sha256": sha(encoded),
                    "changes": changes,
                    "nonlog_ast_equal": True,
                }
            )
    report = {
        "scope": "source-migration-not-runtime-or-image-acceptance",
        "status": "needs-review" if failures else "ready",
        "policy_sha256": sha(args.policy.read_bytes()),
        "files": rows,
        "failures": failures,
        "applied": False,
    }
    if args.apply and not failures:
        for path, before, _ in planned:
            if path.read_bytes() != before:
                raise ReviewRequired(f"Source changed during planning: {path}")
        for path, _, after in planned:
            path.write_bytes(after)
        report["applied"] = True
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "files": len(rows),
                "expressions": sum(len(row["changes"]) for row in rows),
                "failures": failures,
                "applied": report["applied"],
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
