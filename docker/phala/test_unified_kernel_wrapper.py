"""Offline contract tests for the unified SGLang kernel wrapper installer."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
BASE_VERSION = "0.4.7"
BASE_BYTES = b"# stock kvcacheio wrapper\n"
SOURCE_BYTES = b"# engine-owned kvcacheio wrapper\n"
RECORD_RELATIVE = "sglang_kernel-0.4.7.dist-info/RECORD"


def load_installer():
    spec = importlib.util.spec_from_file_location(
        "sglang_unified_kernel_wrapper_test_installer",
        ROOT / "install_unified_kernel_wrapper.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


INSTALLER = load_installer()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii")
    return "sha256=" + encoded.rstrip("=")


def csv_bytes(rows: list[list[str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def read_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


class FakeDistribution:
    def __init__(self, root: Path, version: str, members: list[str]):
        self._root = root
        self.version = version
        self.files = members

    def locate_file(self, relative: str) -> Path:
        return self._root / relative


def make_fixture(
    root: Path,
    *,
    version: str = BASE_VERSION,
    source_bytes: bytes = SOURCE_BYTES,
    base_bytes: bytes = BASE_BYTES,
    members: list[str] | None = None,
    target_exists: bool = True,
    record_exists: bool = True,
    record_rows: list[list[str]] | None = None,
    receipt_exists: bool = False,
    source_is_target: bool = False,
):
    distribution_root = root / "site-packages"
    target = distribution_root / INSTALLER.TARGET_RELATIVE
    record = distribution_root / RECORD_RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    record.parent.mkdir(parents=True, exist_ok=True)

    other_relative = "sgl_kernel/other.py"
    other = distribution_root / other_relative
    other.write_bytes(b"unrelated module\n")
    if target_exists:
        target.write_bytes(source_bytes if source_is_target else base_bytes)
    if record_rows is None:
        record_rows = [
            [
                INSTALLER.TARGET_RELATIVE,
                record_digest(base_bytes),
                str(len(base_bytes)),
            ],
            [RECORD_RELATIVE, "", ""],
            [
                other_relative,
                record_digest(other.read_bytes()),
                str(other.stat().st_size),
            ],
        ]
    if record_exists:
        record.write_bytes(csv_bytes(record_rows))

    source = target if source_is_target else root / "frozen-engine" / "kvcacheio.py"
    if not source_is_target:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(source_bytes)

    receipt = root / "receipts" / "wrapper.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if receipt_exists:
        receipt.write_bytes(b"existing receipt\n")

    if members is None:
        members = [INSTALLER.TARGET_RELATIVE, RECORD_RELATIVE, other_relative]
    distribution = FakeDistribution(distribution_root, version, members)
    return SimpleNamespace(
        distribution=distribution,
        root=distribution_root,
        source=source,
        target=target,
        record=record,
        receipt=receipt,
        source_bytes=source_bytes,
        base_bytes=base_bytes,
    )


def install_fixture(fixture, **overrides):
    values = {
        "wrapper": fixture.source,
        "expected_sha256": digest(fixture.source.read_bytes()),
        "base_sha256": digest(fixture.base_bytes),
        "base_version": BASE_VERSION,
        "receipt": fixture.receipt,
    }
    values.update(overrides)
    with patch.object(
        INSTALLER.importlib.metadata,
        "distribution",
        return_value=fixture.distribution,
    ):
        return INSTALLER.install(**values)


class UnifiedKernelWrapperInstallerTests(unittest.TestCase):
    def assert_rejected_without_writes(self, fixture, **overrides):
        target_before = fixture.target.read_bytes() if fixture.target.exists() else None
        record_before = fixture.record.read_bytes() if fixture.record.exists() else None
        receipt_before = (
            fixture.receipt.read_bytes() if fixture.receipt.exists() else None
        )
        with self.assertRaises(INSTALLER.WrapperInstallError):
            install_fixture(fixture, **overrides)
        self.assertEqual(
            fixture.target.read_bytes() if fixture.target.exists() else None,
            target_before,
        )
        self.assertEqual(
            fixture.record.read_bytes() if fixture.record.exists() else None,
            record_before,
        )
        self.assertEqual(
            fixture.receipt.read_bytes() if fixture.receipt.exists() else None,
            receipt_before,
        )

    def test_install_replaces_wrapper_updates_record_and_writes_identity_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(Path(directory))
            before_rows = read_rows(fixture.record)

            result = install_fixture(fixture)

            self.assertEqual(fixture.target.read_bytes(), SOURCE_BYTES)
            rows = read_rows(fixture.record)
            self.assertEqual(len(rows), len(before_rows))
            self.assertEqual(
                [row for row in rows if row[0] == INSTALLER.TARGET_RELATIVE],
                [
                    [
                        INSTALLER.TARGET_RELATIVE,
                        record_digest(SOURCE_BYTES),
                        str(len(SOURCE_BYTES)),
                    ]
                ],
            )
            unrelated_before = [
                row for row in before_rows if row[0] != INSTALLER.TARGET_RELATIVE
            ]
            unrelated_after = [
                row for row in rows if row[0] != INSTALLER.TARGET_RELATIVE
            ]
            self.assertEqual(unrelated_after, unrelated_before)

            receipt = json.loads(fixture.receipt.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "installed")
            self.assertEqual(receipt["schema"], INSTALLER.RECEIPT_SCHEMA)
            self.assertEqual(receipt["distribution"], "sglang-kernel")
            self.assertEqual(receipt["base_binary_package_version"], BASE_VERSION)
            self.assertFalse(
                receipt["base_binary_package_version_is_python_wrapper_identity"]
            )
            self.assertTrue(receipt["source_owned_wrapper_modified"])
            self.assertEqual(receipt["wrapper_source_sha256"], digest(SOURCE_BYTES))
            self.assertEqual(receipt["target_preimage_sha256"], digest(BASE_BYTES))
            self.assertEqual(receipt["target_sha256"], digest(SOURCE_BYTES))
            self.assertIn("Docker RUN layer", receipt["transaction_boundary"])
            self.assertIn(
                "not a cross-file atomic transaction", receipt["transaction_boundary"]
            )

    def test_member_path_escape_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(
                Path(directory),
                members=[INSTALLER.TARGET_RELATIVE, RECORD_RELATIVE, "../outside.py"],
            )
            self.assert_rejected_without_writes(fixture)

    def test_record_member_path_escape_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(
                Path(directory),
                members=[
                    INSTALLER.TARGET_RELATIVE,
                    "../outside/RECORD",
                    "sgl_kernel/other.py",
                ],
            )
            self.assert_rejected_without_writes(fixture)

    def test_target_not_owned_by_distribution_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(
                Path(directory),
                members=[RECORD_RELATIVE, "sgl_kernel/other.py"],
            )
            self.assert_rejected_without_writes(fixture)

    def test_distribution_version_mismatch_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(Path(directory), version="0.4.8")
            self.assert_rejected_without_writes(fixture)

    def test_source_hash_mismatch_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(Path(directory))
            self.assert_rejected_without_writes(
                fixture,
                expected_sha256=digest(b"a different frozen wrapper\n"),
            )

    def test_installed_target_preimage_mismatch_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(Path(directory))
            fixture.target.write_bytes(b"tampered wrapper\n")
            self.assert_rejected_without_writes(fixture)

    def test_missing_target_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(Path(directory), target_exists=False)
            self.assert_rejected_without_writes(fixture)

    def test_duplicate_record_members_are_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(
                Path(directory),
                members=[
                    INSTALLER.TARGET_RELATIVE,
                    RECORD_RELATIVE,
                    "other-package-1.0.dist-info/RECORD",
                    "sgl_kernel/other.py",
                ],
            )
            self.assert_rejected_without_writes(fixture)

    def test_existing_receipt_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(Path(directory), receipt_exists=True)
            self.assert_rejected_without_writes(fixture)

    def test_source_and_target_alias_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(Path(directory), source_is_target=True)
            self.assert_rejected_without_writes(fixture)

    def test_in_root_symlink_redirect_is_not_distribution_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(Path(directory))
            redirected = fixture.target.parent / "unowned.py"
            redirected.write_bytes(BASE_BYTES)
            original_resolve = Path.resolve

            def resolve(path, *args, **kwargs):
                if path == fixture.target:
                    return original_resolve(redirected)
                return original_resolve(path, *args, **kwargs)

            with patch.object(Path, "resolve", resolve):
                self.assert_rejected_without_writes(fixture)
            self.assertEqual(redirected.read_bytes(), BASE_BYTES)


if __name__ == "__main__":
    unittest.main()
