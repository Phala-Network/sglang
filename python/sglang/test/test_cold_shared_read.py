import importlib.util
import json
import pathlib
import threading
import unittest

SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "srt"
    / "mem_cache"
    / "cold_shared_read.py"
)
spec = importlib.util.spec_from_file_location("cold_shared_read_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def events(records):
    return [
        json.loads(line.split("cold_shared_read ", 1)[1]) for line in records.output
    ]


class ColdSharedReadTest(unittest.TestCase):
    def test_zero_get_terminal_summary(self):
        with self.assertLogs(module.logger, level="INFO") as records:
            trace = module.ColdSharedReadTrace("rid", "namespace", None, 0)
            trace.terminal("finished")
        begin, end = events(records)
        self.assertEqual((begin["event"], end["event"]), ("begin", "end"))
        self.assertEqual(end["get_calls"], 0)
        self.assertTrue(end["complete"])
        self.assertNotIn('"namespace":', " ".join(records.output))

    def test_miss_error_and_late_cancelled_io(self):
        with self.assertLogs(module.logger, level="INFO") as records:
            trace = module.ColdSharedReadTrace("rid2", "ns2", "salt", 0)
            trace.operation_begin()
            trace.terminal("abort")
            self.assertEqual(len(events(records)), 1)

            done = threading.Event()

            def worker():
                trace.get_begin(2)
                trace.get_end([-1, -1])
                trace.get_begin(1)
                trace.get_end(error=True)
                trace.operation_end()
                done.set()

            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(done.wait(5))
            thread.join()
        end = events(records)[-1]
        self.assertEqual(end["terminal"], "abort")
        self.assertEqual((end["get_calls"], end["get_keys"]), (2, 3))
        self.assertEqual(end["get_errors"], 1)
        self.assertEqual(end["returned_bytes"], 0)

    def test_concurrent_identity_and_unattributed_get(self):
        with self.assertLogs(module.logger, level="INFO") as records:
            first = module.ColdSharedReadTrace("same-rid", "a", None, 0)
            second = module.ColdSharedReadTrace("same-rid", "b", None, 0)
            first.get_begin(1)
            first.get_end([64])
            first.terminal("finished")
            module.record_unattributed_get()
            second.terminal("finished")
        ends = [event for event in events(records) if event["event"] == "end"]
        self.assertNotEqual(ends[0]["epoch"], ends[1]["epoch"])
        self.assertEqual(ends[0]["get_calls"], 1)
        self.assertEqual(ends[0]["returned_bytes"], 64)
        self.assertTrue(ends[0]["complete"])
        self.assertFalse(ends[1]["complete"])
        self.assertEqual(ends[1]["unattributed_get_calls"], 1)

    def test_get_after_terminal_is_rejected_before_backend_call(self):
        with self.assertLogs(module.logger, level="INFO") as records:
            trace = module.ColdSharedReadTrace("rid3", "ns3", None, 0)
            trace.terminal("abort")
            with self.assertRaisesRegex(RuntimeError, "after terminal summary"):
                trace.get_begin(1)
        self.assertEqual(
            [event["event"] for event in events(records)],
            ["begin", "end", "late_get_rejected"],
        )


if __name__ == "__main__":
    unittest.main()
