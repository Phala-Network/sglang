"""Linux CPU integration: real Uvicorn 0.54, four workers and child watchdog."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

import psutil
import requests
import uvicorn
from sglang.srt.utils.engine_failure import create_engine_failure_state
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="base-a-test-cpu")


@unittest.skipUnless(sys.platform == "linux", "requires real POSIX signals")
class TestEngineMultiprocess(unittest.TestCase):
    def start_server(self, *options):
        self.state = create_engine_failure_state()
        self.addCleanup(self.remove_files)
        self.log = open(self.state + ".log", "w+")
        self.addCleanup(self.log.close)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        fixture_dir = str(Path(__file__).parent)
        env = dict(
            os.environ,
            SGLANG_TEST_ENGINE_STATE=self.state,
            PYTHONPATH=fixture_dir + os.pathsep + os.environ.get("PYTHONPATH", ""),
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("engine_failure_fixture.py")),
                "--port",
                str(port),
                *options,
            ],
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        self.addCleanup(self.stop_server)
        self.worker_pids = self.wait_all_workers(200, timeout=90)
        self.child_pids = {
            child.pid
            for child in psutil.Process(self.process.pid).children(recursive=True)
        }
        self.children = psutil.Process(self.process.pid).children(recursive=True)
        self.assertEqual(len(self.worker_pids), 4)
        print(
            json.dumps(
                {
                    "phase": "four_workers_ready",
                    "parent": self.process.pid,
                    "workers": sorted(self.worker_pids),
                    "tree": sorted(self.child_pids),
                    "uvicorn": uvicorn.__version__,
                }
            ),
            flush=True,
        )

    def remove_files(self):
        for suffix in ("", ".trigger", ".release", ".pids", ".log"):
            Path(self.state + suffix).unlink(missing_ok=True)

    def stop_server(self):
        for child in getattr(self, "children", []):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        if self.process.poll() is None:
            process = psutil.Process(self.process.pid)
            children = process.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.kill()
            self.process.wait(timeout=10)

    def wait_all_workers(self, status, *, path="/ready", timeout=10):
        seen = set()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.log.flush()
                self.fail(Path(self.state + ".log").read_text()[-6000:])
            try:
                response = requests.get(
                    self.url + path, headers={"Connection": "close"}, timeout=1
                )
                if response.status_code == status:
                    seen.add(int(response.headers["X-Worker-Pid"]))
                else:
                    self.assertEqual(response.status_code, status)
                if len(seen) == 4:
                    return seen
            except requests.RequestException:
                time.sleep(0.05)
        self.fail(f"Only observed workers {seen} for {path} status={status}")

    def assert_tree_exits(self, timeout):
        self.process.wait(timeout=timeout)
        deadline = time.monotonic() + 10
        alive = set(self.child_pids)
        while alive and time.monotonic() < deadline:
            alive = {
                pid
                for pid in alive
                if psutil.pid_exists(pid)
                and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
            }
            if alive:
                time.sleep(0.05)
        self.assertFalse(alive, f"Surviving engine processes: {alive}")
        print(
            json.dumps(
                {"phase": "engine_tree_exited", "exit_code": self.process.returncode}
            ),
            flush=True,
        )

    def test_original_uvicorn_discards_sigquit_negative_control(self):
        if uvicorn.__version__ != "0.54.0":
            self.skipTest("negative control targets the incident's Uvicorn version")
        self.start_server("--original")
        os.kill(self.process.pid, signal.SIGQUIT)
        time.sleep(1)
        self.assertIsNone(self.process.poll())
        self.assertEqual(self.wait_all_workers(200), self.worker_pids)
        os.kill(self.process.pid, signal.SIGTERM)
        self.assert_tree_exits(15)

    def run_fault(self, *options):
        self.start_server(*options)
        # More than two active-timeout intervals of idle must remain healthy.
        time.sleep(1.2)
        self.assertEqual(self.wait_all_workers(200), self.worker_pids)
        Path(self.state + ".trigger").touch()
        deadline = time.monotonic() + 5
        while Path(self.state).read_bytes() != b"1" and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(Path(self.state).read_bytes(), b"1")
        self.assertEqual(self.wait_all_workers(503), self.worker_pids)
        self.assertEqual(self.wait_all_workers(503, path="/health"), self.worker_pids)
        print(
            json.dumps(
                {
                    "phase": "all_workers_unready_before_signal",
                    "workers": sorted(self.worker_pids),
                    "ready": 503,
                    "health": 503,
                }
            ),
            flush=True,
        )
        Path(self.state + ".release").touch()
        self.assert_tree_exits(10)

    def test_hard_timeout_marks_all_workers_then_cleans_entire_tree(self):
        self.run_fault()

    def test_unresponsive_parent_has_bounded_tree_cleanup(self):
        self.run_fault("--ignore-quit")

    def test_parent_exiting_without_cleanup_does_not_orphan_engine(self):
        self.run_fault("--exit-on-quit")

    def test_scheduler_exits_if_parent_was_already_gone(self):
        self.start_server()
        scheduler_pid = json.loads(Path(self.state + ".pids").read_text())["scheduler"]
        self.process.kill()
        self.process.wait(timeout=5)
        Path(self.state + ".release").touch()
        Path(self.state + ".trigger").touch()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                active = psutil.Process(scheduler_pid).status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                active = False
            if not active:
                break
            time.sleep(0.05)
        self.assertFalse(active, "scheduler survived its already-exited parent")


if __name__ == "__main__":
    unittest.main()
