"""Check installed FSS packaging and CPU nogds behavior without model weights."""

import gc
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    import fastsafetensors
    import torch
    from fastsafetensors import SafeTensorsFileLoader, SingleGroup, cpp
    from safetensors.torch import save_file

    from sglang.srt.model_loader import weight_utils
    from sglang.srt.server_args import prepare_server_args

    assert fastsafetensors.__version__ == "0.5.0"
    assert (SingleGroup().size(), SingleGroup().rank()) == (1, 0)
    linked = subprocess.check_output(["ldd", cpp.__file__], text=True, timeout=20)
    assert "not found" not in linked, linked
    assert weight_utils.SafeTensorsFileLoader is SafeTensorsFileLoader
    assert weight_utils.SingleGroup is SingleGroup
    expected = {
        "float": torch.arange(24, dtype=torch.float32).reshape(3, 8),
        "bfloat": torch.arange(8, dtype=torch.bfloat16),
        "integer": torch.tensor([0, 7, -4], dtype=torch.int64),
        "empty": torch.empty(0),
    }
    with tempfile.TemporaryDirectory(prefix="fss-cpu-") as directory:
        fixture = str(Path(directory) / "tiny.safetensors")
        save_file(expected, fixture)
        loader = SafeTensorsFileLoader(
            SingleGroup(), torch.device("cpu"), nogds=True, set_numa=False
        )
        loader.add_filenames({0: [fixture]})
        buffers = loader.copy_files_to_device()
        actual = {key: buffers.get_tensor(key) for key in buffers.key_to_rank_lidx}
        assert set(actual) == set(expected)
        assert all(
            actual[key].dtype == expected[key].dtype
            and torch.equal(actual[key], expected[key])
            for key in expected
        )
        buffers.close()
        loader.close()
        assert all(torch.equal(actual[key], expected[key]) for key in expected)
        del actual, buffers, loader
        gc.collect()
    args = prepare_server_args(
        [
            "--model-path",
            "/opt/testdata/nonexistent-model-packaging-only",
            "--load-format",
            "fastsafetensors",
            "--model-loader-extra-config",
            '{"enable_gds":false}',
            "--schedule-policy",
            "hrrn",
        ]
    )
    assert args.load_format == "fastsafetensors"
    config = args.model_loader_extra_config
    if isinstance(config, str):
        config = json.loads(config)
    assert config["enable_gds"] is False and args.schedule_policy == "hrrn"
    print(
        json.dumps(
            {
                "passed": True,
                "fastsafetensors": fastsafetensors.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cpu_nogds_roundtrip": sorted(expected),
                "retained_after_close": True,
                "sglang_binding": True,
                "cuda_model_acceptance": False,
            }
        )
    )


if __name__ == "__main__":
    main()
