"""Canonicalize build-time wheel ordering, after verifying every RECORD entry.

This deliberately narrow tool changes only ZIP ordering/compression and RECORD
serialization. Payload bytes and per-member metadata remain unchanged. It does
not normalize away a compiler, timestamp, permission, or dependency difference.
Signed wheels, links, ambiguous paths and incomplete RECORDs fail closed.
"""
import argparse
import base64
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(name):
    require(name and not any(c in name for c in "\\\x00\r\n:"), "unsafe member path")
    require(all(part not in ("", ".", "..") for part in name.split("/")),
            "unsafe member path")


def read_verified(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)), "duplicate ZIP member")
        require(not any(n.endswith(("/RECORD.jws", "/RECORD.p7s")) for n in names),
                "signed wheels cannot be rewritten")
        records = [n for n in names if n.endswith(".dist-info/RECORD")]
        require(len(records) == 1 and records[0].count("/") == 1,
                "expected exactly one top-level dist-info RECORD")
        record = records[0]
        payload = {}
        for info in infos:
            safe_name(info.filename)
            require(not info.is_dir(), "explicit directory entries are unsupported")
            require(not info.flag_bits & 1, "encrypted wheel is unsupported")
            require(stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG),
                    "non-regular ZIP member")
            require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                    "unsupported wheel compression")
            payload[info.filename] = archive.read(info)  # Also validates CRC.
        rows = list(csv.reader(io.StringIO(payload[record].decode("utf-8"), newline=""),
                               strict=True))
        entries = {}
        for row in rows:
            require(len(row) == 3, "invalid RECORD row")
            name, digest, size = row
            safe_name(name)
            require(name not in entries, "duplicate RECORD entry")
            require(name in payload, "RECORD references missing member")
            entries[name] = row
            if name == record:
                require(digest == size == "", "RECORD self-entry must be unhashed")
                continue
            algorithm, separator, encoded = digest.partition("=")
            require(separator and algorithm in ("sha256", "sha384", "sha512"),
                    "missing or weak RECORD hash")
            actual = base64.urlsafe_b64encode(hashlib.new(algorithm, payload[name]).digest())
            require(encoded == actual.rstrip(b"=").decode("ascii"), "RECORD hash mismatch")
            require(size == str(len(payload[name])), "RECORD size mismatch")
        require(set(entries) == set(payload), "RECORD does not cover all members")
        return infos, payload, entries, record, archive.comment


def member_metadata(info):
    # Compression sizes/CRCs/offsets and encoding flags are rewritten by zipfile;
    # retain the original archive's semantic member metadata without alteration.
    return (info.date_time, info.comment, info.extra, info.create_system,
            info.create_version, info.extract_version, info.internal_attr,
            info.external_attr, info.volume, info.compress_type)


def canonicalize_bytes(data):
    infos, payload, entries, record, comment = read_verified(data)
    record_text = io.StringIO(newline="")
    writer = csv.writer(record_text, lineterminator="\n")
    writer.writerows(entries[name] for name in sorted(entries))
    canonical_record = record_text.getvalue().encode("utf-8")
    dist_info = record.split("/")[0] + "/"
    # Wheel guidance recommends dist-info at the end; RECORD itself goes last.
    ordered = sorted(infos, key=lambda i: (i.filename == record,
                                          i.filename.startswith(dist_info), i.filename))
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", allowZip64=True) as archive:
        archive.comment = comment
        for info in ordered:
            archive.writestr(copy.copy(info),
                             canonical_record if info.filename == record else payload[info.filename],
                             compress_type=info.compress_type, compresslevel=9)
    result = target.getvalue()
    new_infos, new_payload, new_entries, new_record, new_comment = read_verified(result)
    require(new_record == record and new_entries == entries and new_comment == comment,
            "RECORD or archive metadata changed")
    require({n: b for n, b in payload.items() if n != record} ==
            {n: b for n, b in new_payload.items() if n != record}, "wheel payload changed")
    require({i.filename: member_metadata(i) for i in infos} ==
            {i.filename: member_metadata(i) for i in new_infos}, "member metadata changed")
    return result


def canonicalize_file(source, destination):
    source, destination = Path(source), Path(destination)
    require(source.is_file() and not source.is_symlink(), "input must be a regular wheel file")
    require(not destination.is_symlink(), "output must not be a symlink")
    original = source.read_bytes()
    result = canonicalize_bytes(original)
    require(canonicalize_bytes(result) == result, "normalization is not idempotent")
    fd, temporary = tempfile.mkstemp(prefix=".canonical-wheel-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(result)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, stat.S_IMODE(source.stat().st_mode))
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"wheel": destination.name,
            "input_sha256": hashlib.sha256(original).hexdigest(),
            "output_sha256": hashlib.sha256(result).hexdigest(),
            "payload_bytes_preserved": True, "record_hashes_verified": True,
            "member_metadata_preserved": True, "idempotent": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path,
                        help="write a separate wheel; otherwise replace the build output atomically")
    args = parser.parse_args()
    print(json.dumps(canonicalize_file(args.wheel, args.output or args.wheel), sort_keys=True))


if __name__ == "__main__":
    main()
