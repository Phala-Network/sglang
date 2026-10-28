"""Create the fixed ModelExpress wheel source from a verified upstream archive."""

import argparse
import hashlib
import pathlib
import subprocess
import tarfile
import urllib.request

UPSTREAM = "df8bc7ca1730fd40a2af0416b589d84fcd1b090c"
ARCHIVE_SHA256 = "8054eca52b2a83087f6e0679aa89fa1adcdfc259e1008cf6a39478b741edfdd5"


def prepare(output, archive=None):
    here = pathlib.Path(__file__).resolve().parent
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    archive = archive or output.parent / "modelexpress-upstream.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(
            f"https://codeload.github.com/ai-dynamo/modelexpress/tar.gz/{UPSTREAM}",
            archive,
        )
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("upstream archive digest mismatch")
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            parts = pathlib.PurePosixPath(member.name).parts
            if len(parts) <= 3 or parts[1:3] != ("modelexpress_client", "python"):
                continue
            relative = pathlib.PurePosixPath(*parts[3:])
            if ".." in relative.parts or member.issym() or member.islnk():
                raise ValueError("unsafe archive member")
            target = output / str(relative)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tf.extractfile(member).read())
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(here / "modelexpress.patch")],
        cwd=output,
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--archive", type=pathlib.Path)
    args = parser.parse_args()
    prepare(args.output, args.archive)
