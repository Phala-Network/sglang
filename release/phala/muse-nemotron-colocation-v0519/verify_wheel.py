"""Fail the image build if source, native modules or wheel metadata are missing."""
import email.parser
import hashlib
import json
import sys
import sysconfig
import zipfile
from pathlib import Path

DEVELOPER_ONLY_SOURCES = frozenset({
    "multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py",
    "multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/diffusion_skill_env.py",
})


def verify(wheel_dir, source_dir, expected_version):
    wheels = list(Path(wheel_dir).glob("sglang-*.whl"))
    assert len(wheels) == 1, wheels
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_names = [n for n in names if n.endswith(".dist-info/METADATA")]
        assert len(metadata_names) == 1
        metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_names[0]))
        assert metadata["Name"] == "sglang"
        assert metadata["Version"] == expected_version
        assert "xgrammar==0.2.6" in metadata.get_all("Requires-Dist", [])
        assert not any("__editable__" in n or n.endswith(".pyc") for n in names)
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        modules = ["srt/rust_extensions/_server", "srt/rust_extensions/_grpc",
                   "srt/rust_extensions/_multimodal", "srt/mem_cache/rust_tree_core/mem_cache"]
        for module in modules:
            assert "sglang/" + module + suffix in names, module
        checked = 0
        excluded = []
        source_dir = Path(source_dir)
        for path in sorted(source_dir.rglob("*.py")):
            relative = path.relative_to(source_dir).as_posix()
            if relative.startswith("kernels/aot/"):
                continue  # Canonical upstream package exclusion.
            if relative in DEVELOPER_ONLY_SOURCES:
                assert "sglang/" + relative not in names
                excluded.append(relative)
                continue  # Upstream developer skill scripts are not runtime code.
            member = "sglang/" + relative
            assert member in names, ("missing source file", member)
            assert archive.read(member) == path.read_bytes(), ("source mismatch", member)
            checked += 1
        result = {"wheel": wheel.name, "version": expected_version,
                  "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  "source_python_files_verified": checked, "native_extensions_verified": modules,
                  "developer_only_sources_excluded": excluded,
                  "xgrammar_requirement": "0.2.6", "editable": False}
    (Path(wheel_dir) / "wheel-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    verify(*sys.argv[1:])
