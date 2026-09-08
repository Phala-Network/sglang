"""CPU/thread contracts for the CC D2H backport, not CUDA performance tests."""

import ast
import gc
import importlib.util
import os
import queue
import sys
import threading
import unittest
import weakref
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(os.environ.get("SGLANG_SOURCE_ROOT", "/sglang/python/sglang"))
WORKER = Path(
    os.environ.get("ASYNC_D2H_SOURCE", ROOT / "srt/managers/async_d2h_copy_worker.py")
)
spec = importlib.util.spec_from_file_location("d2h_worker_under_test", WORKER)
worker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker_module)
Worker = worker_module.AsyncD2HCopyWorker
Done = worker_module.HostCopyDone


class Device:
    """Thread-local streams and host gates stand in for CUDA ordering only."""

    def __init__(self):
        self.ready = threading.Event()
        self.ready.set()
        self.finish = threading.Event()
        self.finish.set()
        self.sync_started = threading.Event()
        self.local = threading.local()
        self.ready_error = None
        self.sync_error = None
        self.recorded_thread = None
        self.recorded_stream = None

    def Stream(self):
        device = self

        class Stream:
            def synchronize(self):
                device.sync_started.set()
                if not device.finish.wait(3):
                    raise TimeoutError("test copy completion gate timed out")
                if device.sync_error:
                    raise device.sync_error

        return Stream()

    def Event(self):
        device = self

        class Event:
            def record(self):
                device.recorded_thread = threading.get_ident()
                device.recorded_stream = getattr(device.local, "stream", None)

            def synchronize(self):
                if not device.ready.wait(3):
                    raise TimeoutError("test source readiness gate timed out")
                if device.ready_error:
                    raise device.ready_error

        return Event()

    @contextmanager
    def stream(self, stream):
        previous = getattr(self.local, "stream", None)
        self.local.stream = stream
        try:
            yield
        finally:
            self.local.stream = previous


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.device = Device()
        self.worker = Worker(self.device)

    def tearDown(self):
        self.device.ready.set()
        self.device.finish.set()
        self.worker.shutdown(timeout=3)
        self.assertFalse(self.worker._thread.is_alive())

    def wait_done(self, done):
        self.assertTrue(done._done.wait(3), "copy completion was not signaled")

    def test_record_is_not_early_completion(self):
        done = Done()
        done.record()
        self.assertFalse(done.query())
        done.set_done()
        self.assertTrue(done.query())
        done.synchronize()

    def test_submit_returns_before_source_is_ready(self):
        self.device.ready.clear()
        called = threading.Event()
        producer = object()
        with self.device.stream(producer):
            done = self.worker.submit(called.set)
        self.assertEqual(self.device.recorded_thread, threading.get_ident())
        self.assertIs(self.device.recorded_stream, producer)
        self.assertFalse(called.is_set())
        self.assertFalse(done.query())
        self.device.ready.set()
        self.wait_done(done)
        done.synchronize()
        self.assertTrue(called.is_set())

    def test_private_stream_finishes_before_handle_signals(self):
        self.device.finish.clear()
        observations = []

        def copy():
            observations.append((threading.get_ident(), self.device.local.stream))

        done = self.worker.submit(copy)
        self.assertTrue(self.device.sync_started.wait(3))
        self.assertNotEqual(observations[0][0], threading.get_ident())
        self.assertIs(observations[0][1], self.worker.d2h_copy_stream)
        self.assertFalse(done.query())
        self.device.finish.set()
        self.wait_done(done)
        done.synchronize()

    def test_source_copy_and_stream_errors_propagate(self):
        for stage in ("source", "copy", "stream"):
            with self.subTest(stage=stage):
                error = RuntimeError("injected " + stage)
                self.device.ready_error = error if stage == "source" else None
                self.device.sync_error = error if stage == "stream" else None

                def copy():
                    if stage == "copy":
                        raise error

                done = self.worker.submit(copy)
                self.wait_done(done)
                with self.assertRaisesRegex(
                    RuntimeError, "destination tensors are invalid"
                ) as caught:
                    done.synchronize()
                self.assertIs(caught.exception.__cause__, error)
        self.device.ready_error = self.device.sync_error = None
        done = self.worker.submit(lambda: None)
        self.wait_done(done)
        done.synchronize()

    def test_fifo_multiple_pending_steps(self):
        seen = []
        handles = [self.worker.submit(partial(seen.append, i)) for i in range(64)]
        for done in handles:
            self.wait_done(done)
            done.synchronize()
        self.assertEqual(seen, list(range(64)))

    def test_submit_after_shutdown_rejected(self):
        self.worker.shutdown()
        self.worker.shutdown()
        with self.assertRaisesRegex(RuntimeError, "shut down"):
            self.worker.submit(lambda: None)

    def test_completed_callback_is_released(self):
        self.worker.shutdown()
        waiting_for_next = threading.Event()

        class ObservedQueue(queue.Queue):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def get(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    waiting_for_next.set()
                return super().get(*args, **kwargs)

        with patch.object(worker_module.queue, "Queue", ObservedQueue):
            self.worker = Worker(self.device)

        class Callback:
            def __call__(self):
                pass

        callback = Callback()
        reference = weakref.ref(callback)
        done = self.worker.submit(callback)
        del callback
        self.wait_done(done)
        self.assertTrue(waiting_for_next.wait(3))
        gc.collect()
        self.assertIsNone(reference(), "idle worker retained the last batch callback")

    def test_shutdown_timeout_then_join_is_safe(self):
        self.device.ready.clear()
        done = self.worker.submit(lambda: None)
        self.worker.shutdown(timeout=0.01)
        self.assertTrue(self.worker._thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, "shut down"):
            self.worker.submit(lambda: None)
        self.device.ready.set()
        self.worker.shutdown(timeout=3)
        self.wait_done(done)
        done.synchronize()


class SchedulerContractTests(unittest.TestCase):
    def test_opt_in_and_cc_gate(self):
        module = SimpleNamespace(is_confidential_compute=lambda: True)
        with patch.dict(sys.modules, {"sglang.srt.utils.confidential_compute": module}):
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(worker_module.cc_async_d2h_enabled())
            for value, expected in (("0", False), ("1", True)):
                with patch.dict(os.environ, {"SGLANG_CC_ASYNC_D2H": value}):
                    self.assertEqual(worker_module.cc_async_d2h_enabled(), expected)
            module.is_confidential_compute = lambda: False
            with patch.dict(os.environ, {"SGLANG_CC_ASYNC_D2H": "1"}):
                with self.assertRaisesRegex(ValueError, "requires"):
                    worker_module.cc_async_d2h_enabled()
            with patch.dict(os.environ, {"SGLANG_CC_ASYNC_D2H": "yes"}):
                with self.assertRaises(ValueError):
                    worker_module.cc_async_d2h_enabled()

    def helper(self):
        tree = ast.parse((ROOT / "srt/managers/scheduler.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_copy_overlap_result_to_cpu"
        )
        ns = {
            "GenerationBatchResult": object,
            "ScheduleBatch": object,
            "partial": partial,
        }
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]), "scheduler-helper", "exec"
            ),
            ns,
        )
        return ns[method.name], cls

    def test_scheduler_publishes_handle_before_worker_can_run(self):
        helper, _ = self.helper()
        seen = []
        result = SimpleNamespace(copy_done=None)

        class ImmediateWorker:
            def submit(self, fn, *, done):
                self_test.assertIs(result.copy_done, done)
                self_test.assertFalse(done.query())
                fn()
                done.set_done()
                return done

        self_test = self
        result.copy_to_cpu = lambda **kw: (seen.append(kw), result.copy_done.record())
        module = SimpleNamespace(HostCopyDone=Done)
        scheduler = SimpleNamespace(
            enable_async_d2h_copy=True, async_d2h_worker=ImmediateWorker()
        )
        batch = SimpleNamespace(return_logprob=False, return_hidden_states=True)
        with patch.dict(
            sys.modules, {"sglang.srt.managers.async_d2h_copy_worker": module}
        ):
            helper(scheduler, result, batch)
        result.copy_done.synchronize()
        self.assertEqual(
            seen, [{"return_logprob": False, "return_hidden_states": True}]
        )

    def test_non_cc_copy_behavior_is_preserved(self):
        helper, _ = self.helper()
        seen = []
        event = object()
        result = SimpleNamespace(
            copy_done=event, copy_to_cpu=lambda **kw: seen.append(kw)
        )
        helper(
            SimpleNamespace(enable_async_d2h_copy=False),
            result,
            SimpleNamespace(return_logprob=False, return_hidden_states=False),
        )
        self.assertIs(result.copy_done, event)
        self.assertEqual(
            seen, [{"return_logprob": False, "return_hidden_states": False}]
        )

    def test_normal_and_delayed_sampling_use_same_helper(self):
        _, cls = self.helper()
        for name in ("run_batch", "launch_batch_sample_if_needed"):
            method = next(
                n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
            )
            self.assertEqual(
                sum(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "_copy_overlap_result_to_cpu"
                    for n in ast.walk(method)
                ),
                1,
                name,
            )

    def test_pinned_destination_and_record_stream_are_preserved(self):
        tree = ast.parse((ROOT / "srt/managers/utils.py").read_text())
        method = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_async_d2h"
        )
        calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)]
        self.assertTrue(
            any(
                isinstance(n.func, ast.Attribute) and n.func.attr == "record_stream"
                for n in calls
            )
        )
        self.assertTrue(
            any(
                k.arg == "pin_memory"
                and isinstance(k.value, ast.Constant)
                and k.value.value is True
                for n in calls
                for k in n.keywords
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
