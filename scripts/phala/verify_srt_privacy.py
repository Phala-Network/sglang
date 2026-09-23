#!/usr/bin/env python3
"""Verify the exact reviewed migration and unchanged non-log behavior."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from redact_srt_logs import (
    POLICY_PATH,
    WithoutLogs,
    load_policy,
    transform,
    validate_source_root,
)


class ReviewedImports(ast.NodeTransformer):
    def __init__(self, allowed):
        self.allowed = set(allowed)

    def visit_Import(self, node):
        node.names = [alias for alias in node.names if alias.name not in self.allowed]
        return node if node.names else None

    def visit_ImportFrom(self, node):
        node.names = [
            alias
            for alias in node.names
            if f"{node.module}.{alias.name}" not in self.allowed
        ]
        return node if node.names else None


def structural_ast(source, allowed):
    tree = ast.parse(source.lstrip("\ufeff"))
    tree = WithoutLogs().visit(tree)
    return ast.dump(ReviewedImports(allowed).visit(tree), include_attributes=False)


def migration_matches(before, after, relative, policy):
    expected, _ = transform(before, relative, policy)
    allowed = policy["approved_import_removals"].get(relative, [])

    expected_tree = ReviewedImports(allowed).visit(ast.parse(expected.lstrip("\ufeff")))
    return ast.dump(expected_tree, include_attributes=False) == ast.dump(
        ast.parse(after.lstrip("\ufeff")), include_attributes=False
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report exists")
    policy = load_policy(POLICY_PATH)
    source = args.source.resolve()
    validate_source_root(source / "python/sglang/srt")
    if args.baseline and args.baseline != policy["migration_baseline"]:
        parser.error("Baseline must match the reviewed migration policy")

    def git(*command):
        return subprocess.check_output(["git", "-C", str(source), *command])

    baseline = (
        git(
            "rev-parse",
            "--verify",
            "--end-of-options",
            (args.baseline or policy["migration_baseline"]) + "^{commit}",
        )
        .decode()
        .strip()
    )
    git("merge-base", "--is-ancestor", policy["reviewed_source"], baseline)
    prefix = "python/sglang/srt/"
    untracked = (
        git("ls-files", "--others", "--exclude-standard", "--", prefix)
        .decode()
        .splitlines()
    )
    if untracked:
        raise ValueError("Untracked runtime source is outside the declared port")
    paths = git("diff", "--name-only", baseline, "--", prefix).decode().splitlines()
    rows = []
    for name in paths:
        before = git("show", f"{baseline}:{name}").decode("utf-8")
        after = (source / name).read_text(encoding="utf-8")
        allowed = policy["approved_import_removals"].get(name.removeprefix(prefix), [])
        compile(after.encode("utf-8"), name, "exec")
        equal = structural_ast(before, allowed) == structural_ast(after, allowed)
        exact = migration_matches(before, after, name.removeprefix(prefix), policy)
        rows.append(
            {
                "file": name,
                "nonlog_ast_equal": equal,
                "reviewed_migration_ast_equal": exact,
                "approved_import_removals": allowed,
                "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
                "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
            }
        )
    report = {
        "scope": "reviewed-source-migration-not-runtime-qualification",
        "baseline": baseline,
        "files": rows,
        "passed": bool(rows)
        and all(
            row["nonlog_ast_equal"] and row["reviewed_migration_ast_equal"]
            for row in rows
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "files": len(rows),
                "failures": [
                    row["file"]
                    for row in rows
                    if not row["nonlog_ast_equal"]
                    or not row["reviewed_migration_ast_equal"]
                ],
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
