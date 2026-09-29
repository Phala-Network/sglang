"""Exercise real PD submission methods with a fake synchronous native engine."""

import ast
import concurrent.futures
import hashlib
import importlib.util
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/disaggregation/mooncake/conn.py"


if importlib.util.find_spec("sglang") is None:
    for package in (
        "sglang",
        "sglang.srt",
        "sglang.srt.disaggregation",
        "sglang.srt.disaggregation.mooncake",
        "sglang.srt.mem_cache",
    ):
        source_package = types.ModuleType(package)
        source_package.__path__ = [str(ROOT / "python" / package.replace(".", "/"))]
        source_package.__spec__ = importlib.util.spec_from_loader(
            package, loader=None, is_package=True
        )
        sys.modules[package] = source_package


def load_manager(namespace):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    owner = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MooncakeKVManager"
    )
    names = {
        "_transfer_data",
        "_transfer_native_batch",
        "_log_pd_transfer_result",
        "_send_kvcache_generic",
        "send_kvcache_slice",
        "send_aux",
        "_send_mamba_state",
        "_await_transfer_futures",
    }
    owner.bases = []
    owner.body = [
        n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            owner,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["MooncakeKVManager"]()


class PDTransferDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.enabled = True
        self.log = Mock()
        self.manager = load_manager(
            {
                "envs": NS(
                    SGLANG_MOONCAKE_PD_TRANSFER_DIAGNOSTICS=NS(
                        get=lambda: self.enabled
                    ),
                    SGLANG_MOONCAKE_SEND_AUX_TCP=NS(get=lambda: False),
                ),
                "concurrent": concurrent,
                "hashlib": hashlib,
                "os": os,
                "threading": threading,
                "logger": self.log,
                "np": np,
                "group_concurrent_contiguous": lambda src, dst: ([src], [dst]),
                "build_transfer_entry_pairs": lambda *args, **kwargs: [(0, 0)],
            }
        )
        self.manager.engine = NS(batch_transfer_sync=Mock(return_value=0))
        self.manager.kv_args = NS(
            engine_rank=3,
            kv_item_lens=[8],
            page_size=1,
            total_kv_head_num=1,
            kv_head_num=1,
            kv_data_ptrs=[10000],
            kv_layer_ids=[0],
            aux_data_ptrs=[10000, 20000],
            aux_item_lens=[4, 12],
        )
        self.manager.attn_tp_size = 1
        self.manager.pp_size = 1
        self.manager.is_mla_backend = True
        self.manager.is_hybrid_mla_backend = False
        self.manager.enable_custom_mem_pool = False
        self.manager.enable_deferred_decode_kv_release = False
        self.manager.max_transfer_batch_indices = 0
        self.manager.get_mla_kv_ptrs_with_pp = lambda src, dst, state: (
            src,
            dst,
            len(src),
        )
        self.peer = "private-peer:12345"
        self.room = 987654321

    def records(self):
        return [call.args[0] % call.args[1:] for call in self.log.info.call_args_list]

    def assert_record(self, record, count, size, result, kind="kv"):
        for expected in (
            "engine=pd",
            "batch_count=1",
            f"entry_count={count}",
            f"submitted_bytes={size}",
            f"native_result={result}",
            f"completed={result == 0}",
            f"kind={kind}",
            "rank=3",
        ):
            self.assertIn(expected, record)
        self.assertIn(
            "room=" + hashlib.sha256(str(self.room).encode()).hexdigest()[:16], record
        )
        self.assertNotIn(self.peer, record)
        self.assertNotIn(str(self.room), record)
        self.assertNotIn("10000", record)
        self.assertNotIn("transferred_bytes", record)

    def test_actual_submission_success_failure_and_exception(self):
        blocks = [(10000, 30000, 4), (20000, 40000, 12)]
        for result in (0, -1):
            with self.subTest(result=result):
                self.log.reset_mock()
                self.manager.engine.batch_transfer_sync.return_value = result
                self.assertEqual(
                    self.manager._transfer_data(
                        self.peer, blocks, diagnostic_room=self.room
                    ),
                    result,
                )
                self.manager.engine.batch_transfer_sync.assert_called_with(
                    self.peer, [10000, 20000], [30000, 40000], [4, 12]
                )
                self.assert_record(self.records()[0], 2, 16, result)
        failure = RuntimeError("secret native args 10000")
        self.manager.engine.batch_transfer_sync.side_effect = failure
        self.log.reset_mock()
        with self.assertRaises(RuntimeError) as caught:
            self.manager._transfer_data(self.peer, blocks, diagnostic_room=self.room)
        self.assertIs(caught.exception, failure)
        self.assert_record(self.records()[0], 2, 16, "exception")
        self.assertNotIn("secret", self.records()[0])

    def test_completion_is_logged_only_after_native_return(self):
        def native(*args):
            self.log.info.assert_not_called()
            return 0

        self.manager.engine.batch_transfer_sync.side_effect = native
        self.manager._transfer_data(
            self.peer, [(10000, 30000, 17)], diagnostic_room=self.room
        )
        self.assert_record(self.records()[0], 1, 17, 0)

    def test_bounded_batches_record_actual_sizes_and_stop_on_failure(self):
        self.manager.max_transfer_batch_indices = 2
        for results, expected_sizes in (([0, 0], [16, 8]), ([-1], [16])):
            with self.subTest(results=results):
                self.log.reset_mock()
                self.manager.engine.batch_transfer_sync.side_effect = results
                with concurrent.futures.ThreadPoolExecutor(2) as executor:
                    ret = self.manager._send_kvcache_generic(
                        self.peer,
                        [10000],
                        [30000],
                        [8],
                        np.array([1, 2, 3]),
                        np.array([4, 5, 6]),
                        executor,
                        diagnostic_room=self.room,
                    )
                self.assertEqual(ret, results[-1])
                self.assertEqual(len(self.records()), len(expected_sizes))
                for record, size, result in zip(
                    self.records(), expected_sizes, results
                ):
                    self.assert_record(record, 1, size, result)

    def test_off_and_empty_do_not_log(self):
        self.enabled = False
        self.assertEqual(self.manager._transfer_data(self.peer, [(1, 2, 9)]), 0)
        self.log.info.assert_not_called()
        self.enabled = True
        self.manager.engine.batch_transfer_sync.reset_mock()
        self.assertEqual(self.manager._transfer_data(self.peer, []), 0)
        self.manager.engine.batch_transfer_sync.assert_not_called()
        self.log.info.assert_not_called()

    def test_generic_batches_and_state_labels(self):
        for custom, kind in ((False, None), (True, "state")):
            with self.subTest(custom=custom):
                self.log.reset_mock()
                self.manager.enable_custom_mem_pool = custom
                self.manager.custom_mem_pool_type = "NVLINK"
                with concurrent.futures.ThreadPoolExecutor(2) as executor:
                    result = self.manager._send_kvcache_generic(
                        self.peer,
                        [10000],
                        [30000],
                        [8],
                        np.array([1, 2]),
                        np.array([3, 4]),
                        executor,
                        diagnostic_room=self.room,
                        diagnostic_kind=kind,
                    )
                self.assertEqual(result, 0)
                self.assert_record(self.records()[0], 1, 16, 0, kind or "kv")

    def test_aux_and_slot_state_labels(self):
        req = NS(mooncake_session_id=self.peer, room=self.room, dst_aux_index=2)
        self.assertEqual(self.manager.send_aux(req, 1, [30000, 40000]), 0)
        self.assert_record(self.records()[0], 2, 16, 0, "aux")
        self.log.reset_mock()
        self.assertEqual(
            self.manager._send_mamba_state(req, [1], [10000], [8], [30000], [2]), 0
        )
        self.assert_record(self.records()[0], 1, 8, 0, "state")

    def test_real_sliced_submission_success_failure_exception_and_off(self):
        for enabled, result in ((True, 0), (True, -1), (False, 0)):
            with self.subTest(enabled=enabled, result=result):
                self.enabled = enabled
                self.log.reset_mock()
                self.manager.engine.batch_transfer_sync.return_value = result
                with concurrent.futures.ThreadPoolExecutor(2) as executor:
                    returned = self.manager.send_kvcache_slice(
                        self.peer,
                        np.array([1, 2]),
                        [30000],
                        np.array([3, 4]),
                        0,
                        1,
                        8,
                        executor,
                        [0],
                        diagnostic_room=self.room,
                    )
                self.assertEqual(returned, result)
                self.manager.engine.batch_transfer_sync.assert_called_with(
                    self.peer, [10008, 10016], [30024, 30032], [8, 8]
                )
                if enabled:
                    self.assert_record(self.records()[0], 2, 16, result)
                else:
                    self.log.info.assert_not_called()
        self.enabled = True
        self.log.reset_mock()
        self.manager.engine.batch_transfer_sync.side_effect = RuntimeError(
            "private native detail"
        )
        with concurrent.futures.ThreadPoolExecutor(2) as executor:
            with self.assertRaisesRegex(RuntimeError, "private native detail"):
                self.manager.send_kvcache_slice(
                    self.peer,
                    np.array([1]),
                    [30000],
                    np.array([2]),
                    0,
                    1,
                    8,
                    executor,
                    [0],
                    diagnostic_room=self.room,
                )
        self.assert_record(self.records()[0], 1, 8, "exception")

    def test_staging_propagates_room_without_changing_default_call(self):
        path = ROOT / "python/sglang/srt/disaggregation/common/staging_handler.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "PrefillStagingStrategy"
        )
        method = next(
            n
            for n in owner.body
            if isinstance(n, ast.FunctionDef) and n.name == "transfer"
        )
        namespace = {}
        module = ast.Module(body=[method], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        send = Mock(return_value=0)
        strategy = NS(kv_manager=NS(send_kvcache_staged=send), staging_buffer=object())
        target = NS(
            dst_tp_rank=0,
            dst_attn_tp_size=1,
            dst_kv_item_len=8,
            dst_kv_layer_ids=[0],
            staging=None,
        )
        for room in (None, self.room):
            self.assertEqual(
                namespace["transfer"](
                    strategy, self.peer, [1], 30000, 16, target, diagnostic_room=room
                ),
                0,
            )
            kwargs = send.call_args.kwargs
            if room is None:
                self.assertNotIn("diagnostic_room", kwargs)
            else:
                self.assertEqual(kwargs["diagnostic_room"], room)

    def test_switch_defaults_off(self):
        source = (ROOT / "python/sglang/srt/environ.py").read_text(encoding="utf-8")
        self.assertIn(
            "SGLANG_MOONCAKE_PD_TRANSFER_DIAGNOSTICS = EnvBool(False)", source
        )


if __name__ == "__main__":
    unittest.main()
