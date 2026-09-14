import importlib.util
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("canonicalizer", Path(__file__).with_name("canonicalize_wheel.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_order_record_and_timestamp_are_canonical(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1789361237")
    payload = {"pkg/a.py": b"print('test')\n", "pkg/lib.so": b"\x7fELFopaque-native-bytes", "pkg-1.dist-info/METADATA": b"Name: pkg\nVersion: 1\n"}
    for index, items in enumerate((list(payload.items()), list(reversed(payload.items())))):
        path = tmp_path / f"{index}.whl"
        with zipfile.ZipFile(path, "w") as wheel:
            for name, data in items:
                info = zipfile.ZipInfo(name, (2024 + index, 1, 1, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                wheel.writestr(info, data)
            wheel.writestr("pkg-1.dist-info/RECORD", f"old-record-{index}")
        module.canonicalize(path)
        with zipfile.ZipFile(path) as wheel:
            for name, data in payload.items():
                assert wheel.read(name) == data
        before = path.read_bytes()
        module.canonicalize(path)
        assert path.read_bytes() == before
    assert (tmp_path / "0.whl").read_bytes() == (tmp_path / "1.whl").read_bytes()


def test_duplicate_member_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1789361237")
    path = tmp_path / "bad.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("pkg/a.py", b"first")
        with pytest.warns(UserWarning):
            wheel.writestr("pkg/a.py", b"second")
        wheel.writestr("pkg-1.dist-info/RECORD", b"")
    with pytest.raises(AssertionError, match="Duplicate"):
        module.canonicalize(path)
