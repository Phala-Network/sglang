#!/usr/bin/env python3
"""Install the engine-owned sgl_kernel Python wrapper into an installed wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile


DIST_NAME = "sglang-kernel"
TARGET_RELATIVE = "sgl_kernel/kvcacheio.py"
RECEIPT_SCHEMA = "phala.sglang.unified-kernel-wrapper-receipt.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class WrapperInstallError(ValueError):
    """The installed distribution or wrapper contract is not safe to update."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex_digest(value: str, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value.lower()):
        raise WrapperInstallError(f"{field} must be a 64-hex SHA-256")
    return value.lower()


def _record_digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii")
    return "sha256=" + encoded.rstrip("=")


def _relative_member(value: object, field: str) -> str:
    text = value.as_posix() if hasattr(value, "as_posix") else str(value)
    if "\\" in text:
        raise WrapperInstallError(f"{field} must use POSIX RECORD paths")
    path = PurePosixPath(text)
    if path.is_absolute() or not text or ".." in path.parts:
        raise WrapperInstallError(f"{field} escapes the distribution root")
    return path.as_posix()


def _inside(root: Path, relative: str, field: str) -> Path:
    declared = root / Path(*PurePosixPath(relative).parts)
    path = declared.resolve()
    if not path.is_relative_to(root.resolve()):
        raise WrapperInstallError(f"{field} escapes the distribution root")
    if path != declared:
        raise WrapperInstallError(f"{field} must not redirect through a symlink")
    return path


def _distribution_members(distribution):
    members = distribution.files
    if members is None:
        raise WrapperInstallError("sglang-kernel distribution has no RECORD file list")
    normalized = [_relative_member(member, "distribution member") for member in members]
    if len(normalized) != len(set(normalized)):
        raise WrapperInstallError("sglang-kernel distribution has duplicate members")
    return normalized


def _record_rows(path: Path):
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
    except (OSError, UnicodeError, csv.Error) as error:
        raise WrapperInstallError(
            f"cannot read distribution RECORD: {error}"
        ) from error
    if any(len(row) != 3 for row in rows):
        raise WrapperInstallError("distribution RECORD contains a malformed row")
    return rows


def _record_bytes(rows) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _atomic_replace(path: Path, data: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _validate(
    wrapper: Path,
    expected_sha256: str,
    base_sha256: str,
    base_version: str,
    receipt: Path,
):
    expected_sha256 = _hex_digest(expected_sha256, "expected-sha256")
    base_sha256 = _hex_digest(base_sha256, "base-sha256")
    if expected_sha256 == base_sha256:
        raise WrapperInstallError("expected-sha256 must differ from base-sha256")
    if not isinstance(base_version, str) or not base_version.strip():
        raise WrapperInstallError("base-version must be nonempty")

    wrapper = wrapper.expanduser().resolve(strict=True)
    receipt = receipt.expanduser().resolve()
    if not wrapper.is_file():
        raise WrapperInstallError("wrapper source must be a regular file")
    if receipt.exists():
        raise WrapperInstallError("receipt already exists")
    if not receipt.parent.is_dir():
        raise WrapperInstallError("receipt parent directory must exist")

    source_bytes = wrapper.read_bytes()
    if sha256(source_bytes) != expected_sha256:
        raise WrapperInstallError("wrapper source SHA-256 differs")

    try:
        distribution = importlib.metadata.distribution(DIST_NAME)
    except importlib.metadata.PackageNotFoundError as error:
        raise WrapperInstallError(
            "sglang-kernel distribution is not installed"
        ) from error
    if distribution.version != base_version:
        raise WrapperInstallError(
            f"sglang-kernel version differs: {distribution.version!r} != {base_version!r}"
        )

    members = _distribution_members(distribution)
    if TARGET_RELATIVE not in members:
        raise WrapperInstallError("target wrapper is not owned by sglang-kernel")
    record_members = [
        member
        for member in members
        if PurePosixPath(member).name == "RECORD"
        and any(part.endswith(".dist-info") for part in PurePosixPath(member).parts)
    ]
    if len(record_members) != 1:
        raise WrapperInstallError("sglang-kernel must have exactly one RECORD member")

    distribution_root = Path(distribution.locate_file("")).resolve()
    target = _inside(distribution_root, TARGET_RELATIVE, "target wrapper")
    record_relative = record_members[0]
    record = _inside(distribution_root, record_relative, "RECORD")
    if wrapper == target or wrapper == record or target == record:
        raise WrapperInstallError("wrapper, target and RECORD must be different files")
    if not target.is_file() or not record.is_file():
        raise WrapperInstallError("distribution target or RECORD is missing")

    target_before = target.read_bytes()
    if sha256(target_before) != base_sha256:
        raise WrapperInstallError("installed target preimage SHA-256 differs")
    rows = _record_rows(record)
    target_rows = [row for row in rows if row[0] == TARGET_RELATIVE]
    record_rows = [row for row in rows if row[0] == record_relative]
    if len(target_rows) != 1:
        raise WrapperInstallError("RECORD must contain exactly one target wrapper row")
    if len(record_rows) != 1:
        raise WrapperInstallError("RECORD must contain exactly one self row")
    target_row = target_rows[0]
    if target_row[1] != _record_digest(target_before) or target_row[2] != str(
        len(target_before)
    ):
        raise WrapperInstallError(
            "RECORD target preimage does not match the installed target"
        )

    updated_rows = [
        [TARGET_RELATIVE, _record_digest(source_bytes), str(len(source_bytes))]
        if row[0] == TARGET_RELATIVE
        else row
        for row in rows
    ]
    new_record_bytes = _record_bytes(updated_rows)
    if wrapper == record or wrapper == target:
        raise WrapperInstallError("source and distribution metadata must not alias")
    receipt_data = {
        "schema": RECEIPT_SCHEMA,
        "distribution": DIST_NAME,
        "base_binary_package_version": distribution.version,
        "base_binary_package_version_is_python_wrapper_identity": False,
        "source_owned_wrapper_modified": True,
        "wrapper_source_sha256": expected_sha256,
        "target": TARGET_RELATIVE,
        "target_preimage_sha256": base_sha256,
        "target_sha256": expected_sha256,
        "record": record_relative,
        "record_target_sha256": _record_digest(source_bytes),
        "record_target_size": len(source_bytes),
        "transaction_boundary": (
            "Docker RUN layer; target and RECORD replacements are sequential, "
            "not a cross-file atomic transaction"
        ),
    }
    return {
        "wrapper": wrapper,
        "target": target,
        "record": record,
        "source_bytes": source_bytes,
        "record_bytes": new_record_bytes,
        "receipt": receipt,
        "receipt_data": receipt_data,
    }


def install(
    wrapper: Path,
    expected_sha256: str,
    base_sha256: str,
    base_version: str,
    receipt: Path,
) -> dict:
    checked = _validate(wrapper, expected_sha256, base_sha256, base_version, receipt)
    _atomic_replace(checked["target"], checked["source_bytes"])
    _atomic_replace(checked["record"], checked["record_bytes"])
    with checked["receipt"].open("xb") as stream:
        stream.write(
            (
                json.dumps(checked["receipt_data"], indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
        )
    return {"status": "installed", **checked["receipt_data"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--base-version", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = install(
            args.wrapper,
            args.expected_sha256,
            args.base_sha256,
            args.base_version,
            args.receipt,
        )
    except (OSError, WrapperInstallError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
