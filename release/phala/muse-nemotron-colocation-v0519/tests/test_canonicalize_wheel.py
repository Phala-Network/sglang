"""Build-only wheel regression tests; run with Python unittest on the builder."""
import base64
import csv
import hashlib
import io
from pathlib import Path
import stat
import sys
import tempfile
import unittest
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canonicalize_wheel import canonicalize_bytes, canonicalize_file, read_verified

RECORD = "example-1.0.dist-info/RECORD"
PAYLOAD = {"example/__init__.py": b"value = 42\n",
           "example/native.so": bytes(range(256)) * 7,
           "example/a,unicode-\u4e2d.txt": b"example data\n",
           "example-1.0.dist-info/METADATA": b"Name: example\nVersion: 1.0\n"}


def wheel(reverse=False, rows_change=None, payload_change=None, infos_change=None, source_payload=None):
    payload = dict(PAYLOAD if source_payload is None else source_payload)
    rows = [[n, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b"=").decode(),
             str(len(b))] for n, b in payload.items()] + [[RECORD, "", ""]]
    if rows_change:
        rows_change(rows)
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(list(reversed(rows)) if reverse else rows)
    payload[RECORD] = record.getvalue().encode()
    if payload_change:
        payload_change(payload)
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as archive:
        archive.comment = b"build fixture"
        for name in sorted(payload, reverse=reverse):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 11, 23, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            if infos_change:
                infos_change(info)
            archive.writestr(info, payload[name])
    return target.getvalue()


class CanonicalWheelTests(unittest.TestCase):
    def test_different_member_and_record_order_has_identical_result(self):
        a, b = wheel(), wheel(reverse=True)
        self.assertNotEqual(a, b)
        self.assertEqual(canonicalize_bytes(a), canonicalize_bytes(b))

    def test_payload_metadata_and_idempotence(self):
        original = wheel()
        result = canonicalize_bytes(original)
        self.assertEqual(result, canonicalize_bytes(result))
        _, payload, rows, _, comment = read_verified(result)
        self.assertEqual(PAYLOAD, {n: b for n, b in payload.items() if n != RECORD})
        self.assertEqual(list(rows), sorted(rows))
        self.assertEqual(comment, b"build fixture")
        with zipfile.ZipFile(io.BytesIO(result)) as archive:
            self.assertEqual(archive.namelist()[-1], RECORD)

    def test_real_payload_differences_are_not_hidden(self):
        original = wheel()
        changed = dict(PAYLOAD, **{"example/native.so": b"different valid library payload"})
        self.assertNotEqual(canonicalize_bytes(original),
                            canonicalize_bytes(wheel(source_payload=changed)))

    def test_timestamp_and_permission_differences_are_not_hidden(self):
        original = wheel()
        altered = wheel(infos_change=lambda i: setattr(i, "date_time", (2026, 9, 12, 0, 0, 0)))
        self.assertNotEqual(canonicalize_bytes(original), canonicalize_bytes(altered))
        altered = wheel(infos_change=lambda i: setattr(i, "external_attr", (stat.S_IFREG | 0o755) << 16))
        self.assertNotEqual(canonicalize_bytes(original), canonicalize_bytes(altered))

    def test_atomic_file_and_separate_output(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "example.whl"
            target = Path(root) / "canonical.whl"
            data = wheel()
            source.write_bytes(data)
            evidence = canonicalize_file(source, target)
            self.assertEqual(source.read_bytes(), data)
            self.assertEqual(target.read_bytes(), canonicalize_bytes(data))
            self.assertTrue(evidence["payload_bytes_preserved"])
            canonicalize_file(source, source)
            self.assertEqual(source.read_bytes(), target.read_bytes())

    def test_hash_and_size_tampering_are_rejected(self):
        for change in [lambda rows: rows[0].__setitem__(1, "sha256=broken"),
                       lambda rows: rows[0].__setitem__(2, "999"),
                       lambda rows: rows[0].__setitem__(1, "md5=weak"),
                       lambda rows: rows[0].__setitem__(1, ""),
                       lambda rows: rows[-1].__setitem__(2, "0")]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                canonicalize_bytes(wheel(rows_change=change))

    def test_missing_extra_duplicate_records_are_rejected(self):
        for change in [lambda rows: rows.pop(0),
                       lambda rows: rows.append(list(rows[0])),
                       lambda rows: rows.append(["missing.txt", "sha256=x", "0"])]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                canonicalize_bytes(wheel(rows_change=change))

    def test_changed_payload_is_rejected_before_rewriting(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            canonicalize_bytes(wheel(payload_change=lambda p: p.update({"example/native.so": b"bad"})))

    def test_unsafe_and_signed_members_are_rejected(self):
        for name in ["../escape", "/absolute", "x/../escape", "x//y", "x/./y", "x\\y", "C:bad",
                     "example-1.0.dist-info/RECORD.jws", "example-1.0.dist-info/RECORD.p7s"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                canonicalize_bytes(wheel(payload_change=lambda p: p.update({name: b"bad"})))

    def test_links_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-regular"):
            canonicalize_bytes(wheel(infos_change=lambda i: setattr(i, "external_attr",
                                                                    (stat.S_IFLNK | 0o777) << 16)))

    def test_duplicate_zip_member_is_rejected(self):
        target = io.BytesIO(wheel())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(target, "a") as archive:
                archive.writestr("example/native.so", b"duplicate")
        with self.assertRaisesRegex(ValueError, "duplicate ZIP"):
            canonicalize_bytes(target.getvalue())

    def test_failed_validation_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as root:
            source, target = Path(root) / "input.whl", Path(root) / "output.whl"
            source.write_bytes(wheel(rows_change=lambda rows: rows.pop(0)))
            target.write_bytes(b"preserve existing file")
            with self.assertRaises(ValueError):
                canonicalize_file(source, target)
            self.assertEqual(target.read_bytes(), b"preserve existing file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
