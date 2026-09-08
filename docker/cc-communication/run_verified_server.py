"""Opt-in startup GPU gate followed by the unchanged model-server command."""

import os
import signal
import subprocess
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]
    if not args:
        raise ValueError("Expected the original server command")
    if os.environ.get("PHALA_CC_GPU_PREFLIGHT", "0") == "1":
        if "--model-path" not in args:
            raise ValueError("GPU preflight requires the pinned local model path")
        model_path = args[args.index("--model-path") + 1]
        environment = dict(os.environ, PHALA_CC_MODEL_CONFIG=str(Path(model_path) / "config.json"))
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   "--nnodes=1", "--nproc-per-node=8",
                   str(Path(__file__).with_name("gpu_preflight.py"))]
        process = subprocess.Popen(command, env=environment, start_new_session=True)
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError("CC GPU preflight timed out; model launch refused") from None
        if code != 0:
            raise RuntimeError(f"CC GPU preflight failed ({code}); model launch refused")
        print("PHALA_CC_GPU_PREFLIGHT=PASS; starting original model command", flush=True)
    os.execvp(args[0], args)


if __name__ == "__main__":
    main()
