"""Canonical wheel envelope; preserve every non-RECORD member's exact bytes."""
import base64
import csv
import datetime
import hashlib
import io
import os
from pathlib import Path
import sys
import zipfile


def canonicalize(path: Path) -> None:
    epoch = int(os.environ["SOURCE_DATE_EPOCH"])
    stamp = datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc)
    assert stamp.year >= 1980
    date_time = (stamp.year, stamp.month, stamp.day, stamp.hour, stamp.minute, stamp.second // 2 * 2)
    with zipfile.ZipFile(path) as original:
        names = original.namelist()
        assert len(names) == len(set(names)), "Duplicate wheel members"
        infos = {i.filename: i for i in original.infolist()}
        contents = {name: original.read(name) for name in names}
    records = [n for n in names if n.endswith(".dist-info/RECORD")]
    assert len(records) == 1
    record = records[0]
    table = io.StringIO(newline="")
    writer = csv.writer(table, lineterminator="\n")
    for name in sorted(contents):
        if name == record:
            continue
        payload = contents[name]
        h = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + h, len(payload)))
    writer.writerow((record, "", ""))
    contents[record] = table.getvalue().encode()
    candidate = path.with_suffix(".canonical.whl")
    assert not candidate.exists()
    with zipfile.ZipFile(candidate, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as result:
        for name in sorted(contents):
            info = zipfile.ZipInfo(name, date_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = infos[name].create_system
            info.external_attr = infos[name].external_attr
            result.writestr(info, contents[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(candidate) as check:
        assert set(check.namelist()) == set(names)
        assert all(check.read(name) == payload for name, payload in contents.items())
    candidate.replace(path)
    print(path.name + " sha256=" + hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        canonicalize(Path(filename))
