import importlib.util
import sysconfig
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("verify_wheel", Path(__file__).with_name("verify_wheel.py"))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
VERSION = "0.5.19+phala.mn.r2"


@pytest.mark.parametrize("defect", [None, "version", "dependency", "native", "source", "content", "pyc"])
def test_source_wheel_contract(tmp_path, defect):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_bytes(b"# frozen source\n")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    entries = {"sglang/__init__.py": b"# frozen source\n",
               "sglang-" + VERSION + ".dist-info/METADATA":
               ("Metadata-Version: 2.4\nName: sglang\nVersion: " + VERSION + "\nRequires-Dist: xgrammar==0.2.6\n").encode()}
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    for module in ["srt/rust_extensions/_server", "srt/rust_extensions/_grpc",
                   "srt/rust_extensions/_multimodal", "srt/mem_cache/rust_tree_core/mem_cache"]:
        entries["sglang/" + module + suffix] = b"synthetic native fixture"
    metadata = next(n for n in entries if n.endswith("/METADATA"))
    if defect == "version":
        entries[metadata] = entries[metadata].replace(VERSION.encode(), b"0.5.19")
    elif defect == "dependency":
        entries[metadata] = entries[metadata].replace(b"xgrammar==0.2.6", b"xgrammar==0.2.1")
    elif defect == "native":
        entries.pop("sglang/srt/rust_extensions/_grpc" + suffix)
    elif defect == "source":
        entries.pop("sglang/__init__.py")
    elif defect == "content":
        entries["sglang/__init__.py"] = b"# different code\n"
    elif defect == "pyc":
        entries["sglang/__pycache__/__init__.pyc"] = b"uncontrolled bytecode"
    with zipfile.ZipFile(wheels / ("sglang-" + VERSION + ".whl"), "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    if defect:
        with pytest.raises(AssertionError):
            checker.verify(wheels, source, VERSION)
    else:
        result = checker.verify(wheels, source, VERSION)
        assert result["source_python_files_verified"] == 1
        assert not result["editable"]
