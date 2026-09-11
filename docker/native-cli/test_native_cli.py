"""Image-local packaging regression; no network, source overlays or GPU required."""

import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    assert sys.executable == "/opt/sglang/bin/python3"
    assert importlib.metadata.version("huggingface_hub") == "1.30.0"
    assert shutil.which("hf") == "/opt/sglang/bin/hf"
    assert shutil.which("sglang") == "/opt/sglang/bin/sglang"
    script = Path(shutil.which("hf")).read_text()
    assert script.splitlines()[0] == "#!/opt/sglang/bin/python3"
    assert "from huggingface_hub.cli.hf import main" in script
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    checks = []
    for argv, expected in [(["hf", "--help"], ["download"]),
                           (["hf", "download", "--help"], ["--revision", "--max-workers"])]:
        result = subprocess.run(argv, env=env, text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert all(token in result.stdout for token in expected), result.stdout
        checks.append({"argv": argv, "exit": result.returncode})
    print(json.dumps({"passed": True, "checks": checks, "runtime_inference_tested": False}))


if __name__ == "__main__":
    main()
