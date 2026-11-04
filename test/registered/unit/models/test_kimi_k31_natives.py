import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.utils import k31_natives


def _site(root: Path, limit: int, marker: bool = True, module: bool = True) -> Path:
    header = root / k31_natives._DYNBLOCK_HEADER
    header.parent.mkdir(parents=True)
    header.write_text(f"static constexpr int DynBlockKernelMaxNumExperts = {limit};\n")
    if module:
        cache = root / "flashinfer_jit_cache/lib"
        cache.mkdir(parents=True)
        (cache / "trtllm_gen_routing.so").touch()
        if marker:
            (cache / "k31-dynblock.json").write_text(
                json.dumps({"DynBlockKernelMaxNumExperts": 1024})
            )
    return root


class TestDynBlock(unittest.TestCase):
    def _problem(self, **kwargs):
        with tempfile.TemporaryDirectory() as root:
            return k31_natives._dynblock_problem(_site(Path(root), **kwargs))

    def test_rebuilt_modules_pass(self):
        self.assertIsNone(self._problem(limit=1024))

    def test_stock_limit_is_reported(self):
        self.assertIn("does not cover 896", self._problem(limit=512))

    def test_patched_header_without_a_rebuild_is_reported(self):
        self.assertIn("not rebuilt", self._problem(limit=1024, marker=False))

    def test_no_module_is_reported(self):
        self.assertIn("no FlashInfer", self._problem(limit=1024, module=False))


class TestRequire(unittest.TestCase):
    def _require(self, *, cuda=True, sm100=True):
        with (
            mock.patch.object(
                k31_natives.torch.cuda, "is_available", return_value=cuda
            ),
            mock.patch.object(
                k31_natives,
                "get_platform",
                return_value=types.SimpleNamespace(is_sm100=sm100),
            ),
            mock.patch.object(
                k31_natives,
                "missing_k31_natives",
                return_value=["flashmla_ops lacks x"],
            ),
        ):
            k31_natives.require_k31_natives()

    def test_missing_natives_refuse_to_load(self):
        with self.assertRaisesRegex(RuntimeError, "flashmla_ops lacks x"):
            self._require()

    def test_override_accepts_the_fallback(self):
        with envs.SGLANG_K31_ALLOW_NATIVE_FALLBACK.override(True):
            self._require()

    def test_other_platforms_have_no_such_natives(self):
        self._require(cuda=False)
        self._require(sm100=False)


if __name__ == "__main__":
    unittest.main()
