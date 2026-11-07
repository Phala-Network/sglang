"""Build once from root-frozen source and qualify on builder201; no publication."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TASK_ROOT = Path("/opt/testdata/kimi-k31-video-20261107")
TASK = "kimi-k31-video-20261107"
BASE = "ghcr.io/phala-network/sglang:phala-dev-kimi-k31-3@sha256:08acd79a162875bc9bfded2ff42cd526a8b0b4e1c4f96e32c1997b61246f226b"
VERSION = "phala-dev-kimi-k31-4"
ENV_CHECK = r'''
import importlib.metadata as md,json,subprocess,sys
packages={d.metadata['Name'].lower().replace('_','-'):d.version for d in md.distributions() if d.metadata['Name']}
p=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,timeout=120)
print(json.dumps({'packages':packages,'pip_check':{'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--revision", required=True, help="Exact root-frozen complete source commit; never infer HEAD")
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    assert sys.platform.startswith("linux") and os.uname().nodename == "phala-tdx-prod3", "authorized builder201 only"
    task = TASK_ROOT.resolve()
    repo, fixture, run_dir = args.repo.resolve(), args.fixture.resolve(), args.run_dir.resolve()
    assert all(task in p.parents for p in [repo, fixture, run_dir]), "all files must remain within the exact201 task root"
    assert (fixture / "manifest.json").is_file()
    fixture_files = list(fixture.rglob("*"))
    assert fixture_files and not any(p.is_symlink() or p.is_file() and p.suffix in {".py", ".pyc", ".so", ".sh"} for p in fixture_files)
    assert not run_dir.exists(), "fresh attempt directory required"
    run_dir.mkdir(parents=True)
    run_id = run_dir.name
    assert all(c.isalnum() or c == "-" for c in run_id)
    local_tag = "ghcr.io/phala-network/sglang:" + VERSION + "-candidate-" + run_id
    steps, containers = [], []
    iid = None
    result = {"passed": False, "source_revision": args.revision, "base_image": BASE, "version": VERSION, "published": False}

    def run(name, argv, timeout=120, check=True):
        started = time.monotonic()
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        (run_dir / (name + ".log")).write_text(p.stdout + p.stderr)
        steps.append({"step": name, "exit": p.returncode, "seconds": round(time.monotonic() - started, 3)})
        (run_dir / "progress.json").write_text(json.dumps(steps, indent=2) + "\n")
        if check and p.returncode:
            raise RuntimeError(name + " failed; inspect its task log")
        return p

    def inspect(image):
        return json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]

    def cpu_container(name, image, command, *, entrypoint="python3", fixture_copy=False, timeout=900):
        cname = TASK + "-" + run_id + "-" + name
        assert subprocess.run(["docker", "container", "inspect", cname], capture_output=True).returncode != 0
        argv = ["docker", "create", "--name", cname, "--label", "phala.task=" + TASK,
                "--label", "phala.role=installed-qualification", "--pull=never", "--runtime=runc", "--network=none",
                "--cpus=8", "--memory=16g", "--memory-swap=16g", "--shm-size=1g", "--pids-limit=1024",
                "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "CUDA_VISIBLE_DEVICES=", "-e", "OMP_NUM_THREADS=2",
                "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
                "-e", "KIMI_K31_TEST_DATA=/opt/testdata/k31-hf-fixture", "--entrypoint", entrypoint, image, *command]
        run(name + "-create", argv)
        containers.append(cname)
        if fixture_copy:
            run(name + "-fixture", ["docker", "cp", str(fixture), cname + ":/opt/testdata/k31-hf-fixture"])
        c = json.loads(subprocess.check_output(["docker", "container", "inspect", cname], text=True))[0]
        assert not c["Mounts"] and not c["HostConfig"].get("DeviceRequests"), "installed-image checks must have no source mounts or GPU"
        assert c["HostConfig"]["Runtime"] == "runc" and c["HostConfig"]["NetworkMode"] == "none"
        run(name, ["docker", "start", "--attach", cname], timeout=timeout)
        c = json.loads(subprocess.check_output(["docker", "container", "inspect", cname], text=True))[0]
        assert c["State"]["Status"] == "exited" and c["State"]["ExitCode"] == 0, name
        output = (run_dir / (name + ".log")).read_text()
        run(name + "-remove", ["docker", "rm", c["Id"]])
        containers.remove(cname)
        return output

    try:
        run("prepare-context", [sys.executable, "-B", str(Path(__file__).with_name("prepare_context.py")),
                                "--repo", str(repo), "--revision", args.revision, "--output", str(run_dir / "context")])
        # The development stage pulls this exact base once; never infer cache ownership.
        base = inspect(BASE)
        result["base_image_id"] = base["Id"]
        result["base_layers"] = len(base["RootFS"]["Layers"])
        run("build", ["docker", "buildx", "build", "--builder", "default", "--load", "--network=none", "--progress=plain",
                      "--label", "phala.task=" + TASK, "--build-arg", "PHALA_ENGINE_REVISION=" + args.revision,
                      "--build-arg", "PHALA_RELEASE_VERSION=" + VERSION, "--metadata-file", str(run_dir / "build-metadata.json"),
                      "-t", local_tag, str(run_dir / "context")], timeout=1800)
        candidate = inspect(local_tag)
        iid = candidate["Id"]
        assert candidate["RootFS"]["Layers"][:len(base["RootFS"]["Layers"])] == base["RootFS"]["Layers"], "inherited native/dependency layers changed"
        for field in ["Env", "Entrypoint", "Cmd", "User", "WorkingDir"]:
            assert candidate["Config"].get(field) == base["Config"].get(field), field
        labels = candidate["Config"]["Labels"]
        assert labels["org.opencontainers.image.revision"] == args.revision
        assert labels["org.opencontainers.image.version"] == VERSION
        assert labels["org.opencontainers.image.source"] == "https://github.com/Phala-Network/sglang"
        baseline_env = json.loads(cpu_container("baseline-env", BASE, ["-B", "-c", ENV_CHECK], timeout=180))
        final_env = json.loads(cpu_container("final-env", iid, ["-B", "-c", ENV_CHECK], timeout=180))
        (run_dir / "baseline-environment.json").write_text(json.dumps(baseline_env, indent=2) + "\n")
        (run_dir / "final-environment.json").write_text(json.dumps(final_env, indent=2) + "\n")
        assert baseline_env == final_env, "package versions or pip check differ from image3"
        cpu_container("installed-cpu", iid, ["-B", "/opt/phala/k31-video/qualify_installed.py"], fixture_copy=True)
        cpu_container("hf-help", iid, ["--help"], entrypoint="hf")
        cpu_container("hf-download-help", iid, ["download", "--help"], entrypoint="hf")
        result.update({"passed": True, "image_id": iid, "local_candidate_tag": local_tag,
                       "source_identity_passed": True, "inherited_layers_unchanged": True,
                       "environment_and_pip_check_equal_to_image3": True, "installed_cpu_hf_native_serve_cli_passed": True,
                       "direct_hf_executable_passed": True, "source_mounts": False, "runtime_install": False, "gpu_tested": False})
    except Exception as exc:
        result["error"] = str(exc)
        raise
    finally:
        cleanup = []
        for cname in containers:
            raw = subprocess.run(["docker", "container", "inspect", cname], capture_output=True, text=True)
            if raw.returncode == 0:
                c = json.loads(raw.stdout)[0]
                assert c["Config"]["Labels"].get("phala.task") == TASK
                p = subprocess.run(["docker", "rm", "-f", c["Id"]], capture_output=True, text=True)
                cleanup.append({"container_id": c["Id"], "exit": p.returncode})
        result.update({"finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "steps": steps,
                       "cleanup": cleanup, "cleanup_passed": all(row["exit"] == 0 for row in cleanup),
                       "build_cache_untouched": True, "base_image_retained_for_task_owner": BASE,
                       "candidate_image_cleanup_pending_owner_handoff": iid})
        (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
