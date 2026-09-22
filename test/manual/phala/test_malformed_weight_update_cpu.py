"""Historical malformed-weight regressions on actual source methods and CPU torch.

No full SGLang import, CUDA IPC or distributed update acceptance is implied.
"""

import ast
import dataclasses
import logging
import pickle
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"


def load_nodes(path, names, namespace, owner=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if owner:
        body = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    nodes = [n for n in body if getattr(n, "name", None) in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


class TestMalformedWeightUpdate(unittest.TestCase):
    def setUp(self):
        self.ns = {
            "torch": torch,
            "dataclass": dataclasses.dataclass,
            "logger": logging.getLogger("malformed-weight-cpu"),
            # Pickle is a CPU serialization fixture, not CUDA IPC acceptance.
            "MultiprocessingSerializer": SimpleNamespace(deserialize=pickle.loads),
            "monkey_patch_torch_reductions": Mock(),
            "_unsupported_derived_weight_cache_error": Mock(return_value=None),
            "dynamic_import": Mock(return_value=Mock()),
            "default_weight_loader": Mock(),
        }
        load_nodes(
            SRT / "weight_sync/tensor_bucket.py",
            {"FlattenedTensorMetadata", "FlattenedTensorBucket"},
            self.ns,
        )
        path = SRT / "model_executor/model_runner_components/weight_updater.py"
        load_nodes(
            path,
            {
                "LocalSerializedTensor",
                "_unwrap_tensor",
                "_model_load_weights_direct",
                "_validate_update_weights_from_tensor_payload",
            },
            self.ns,
        )
        load_nodes(
            path,
            {"update_weights_from_tensor", "_update_weights_from_flattened_bucket"},
            self.ns,
            "WeightUpdater",
        )
        self.model = Mock()
        self.model.named_parameters.return_value = [("weight", torch.zeros(1))]
        self.updater = SimpleNamespace(
            get_model=lambda: self.model,
            _assert_weight_cache_inactive=Mock(),
            tp_rank=0,
            device="cpu",
            custom_weight_loaders={},
        )
        self.updater._update_weights_from_flattened_bucket = lambda **kw: self.ns[
            "_update_weights_from_flattened_bucket"
        ](self.updater, **kw)
        self.validate = self.ns["_validate_update_weights_from_tensor_payload"]

    def update(self, payload, fmt=None):
        return self.ns["update_weights_from_tensor"](self.updater, payload, fmt)

    def assert_rejects(self, payload, fmt=None, contains="Invalid"):
        success, message = self.update(payload, fmt)
        self.assertFalse(success)
        self.assertIn(contains, message)
        self.model.load_weights.assert_not_called()

    def test_non_collection_rejected_before_device_lookup(self):
        del self.updater.device
        self.assert_rejects(7, contains="named_tensors must be")

    def test_ablation_without_validation_loses_controlled_rejection(self):
        self.ns["_validate_update_weights_from_tensor_payload"] = lambda *a, **kw: None
        del self.updater.device
        with self.assertRaises(AttributeError):
            self.update(7)
        self.model.load_weights.assert_not_called()

    def test_bad_entries_and_names(self):
        for payload in [
            [7],
            [("name",)],
            [(7, torch.zeros(1))],
            [("", torch.zeros(1))],
            [("name", 7)],
        ]:
            with self.subTest(payload=payload):
                self.assert_rejects(payload)

    def test_local_serialized_envelope_and_rank(self):
        cls = self.ns["LocalSerializedTensor"]
        for values in [None, [], [7]]:
            self.assert_rejects([("weight", cls(values=values))])
        for rank in [-1, 1, "0"]:
            self.updater.tp_rank = rank
            self.assert_rejects(
                [("weight", cls(values=[b"payload"]))], contains="TP rank"
            )

    def test_corrupt_local_serialization_is_controlled(self):
        cls = self.ns["LocalSerializedTensor"]
        for value in [b"not-a-pickle", pickle.dumps(7)]:
            self.assert_rejects(
                [("weight", cls(values=[value]))], contains="failed to unwrap"
            )

    def test_valid_local_tensor(self):
        cls = self.ns["LocalSerializedTensor"]
        success, _ = self.update(
            [("weight", cls(values=[pickle.dumps(torch.ones(1))]))]
        )
        self.assertTrue(success)
        torch.testing.assert_close(
            self.model.load_weights.call_args.args[0][0][1], torch.ones(1)
        )

    def test_unknown_and_nonstring_format_before_device(self):
        del self.updater.device
        for fmt in ["unknown", 7, [], {}]:
            self.assert_rejects([("weight", torch.zeros(1))], fmt, "load_format")

    def test_default_direct_and_custom_dispatch(self):
        tensor = torch.ones(1)
        self.assertTrue(self.update([("weight", tensor)])[0])
        self.model.load_weights.assert_called_once()
        self.model.load_weights.reset_mock()
        self.assertTrue(self.update([("weight", tensor)], "direct")[0])
        self.ns["default_weight_loader"].assert_called_once()
        self.model.load_weights.assert_not_called()
        self.updater.custom_weight_loaders = {"custom.loader": True}
        self.assertTrue(self.update([("weight", tensor)], "custom.loader")[0])
        self.ns["dynamic_import"].assert_called_once_with("custom.loader")
        self.ns["dynamic_import"].return_value.assert_called_once()

    def bucket(self, **changes):
        meta = self.ns["FlattenedTensorMetadata"](
            name="weight",
            shape=torch.Size([1]),
            dtype=torch.float32,
            start_idx=0,
            end_idx=4,
            numel=4,
        )
        for key, value in changes.items():
            setattr(meta, key, value)
        return {"flattened_tensor": torch.ones(1).view(torch.uint8), "metadata": [meta]}

    def test_flattened_valid_round_trip(self):
        self.assertTrue(self.update(self.bucket(), "flattened_bucket")[0])
        torch.testing.assert_close(
            self.model.load_weights.call_args.args[0][0][1], torch.ones(1)
        )

    def test_flattened_consumer_shape_dtype_failures_before_model_mutation(self):
        # Bounds alone do not prove that dtype view + reshape can consume bytes.
        for change in [dict(shape=[2]), dict(dtype=torch.float64), dict(shape=[True])]:
            with self.subTest(change=change):
                payload = self.bucket(**change)
                # A valid first entry must not be loaded before a bad later one.
                payload["metadata"].insert(0, self.bucket()["metadata"][0])
                self.assert_rejects(
                    payload, "flattened_bucket", "failed to reconstruct"
                )

    def test_flattened_bad_envelopes(self):
        for payload in [
            7,
            {},
            {"flattened_tensor": 7, "metadata": []},
            {"flattened_tensor": torch.zeros(1), "metadata": 7},
            {"flattened_tensor": torch.zeros(1), "metadata": [7]},
        ]:
            self.assert_rejects(payload, "flattened_bucket")

    def test_flattened_metadata_validation(self):
        for change in [
            dict(name=""),
            dict(shape=[-1]),
            dict(shape="1"),
            dict(dtype="float32"),
            dict(start_idx=-1),
            dict(end_idx=5),
            dict(end_idx=0, start_idx=1),
            dict(numel=3),
            dict(numel="4"),
        ]:
            with self.subTest(change=change):
                self.assert_rejects(self.bucket(**change), "flattened_bucket")

    def test_ds_derived_cache_guard_still_precedes_updates(self):
        guard = self.ns["_unsupported_derived_weight_cache_error"]
        guard.return_value = "derived-cache-blocked"
        self.assert_rejects(
            [("weight", torch.zeros(1))], contains="derived-cache-blocked"
        )
        guard.assert_called_once_with(self.model)
        self.updater._assert_weight_cache_inactive.assert_not_called()
        self.ns["monkey_patch_torch_reductions"].assert_not_called()

    def test_weight_cache_guard_still_enforced(self):
        self.updater._assert_weight_cache_inactive.side_effect = RuntimeError(
            "cache-active"
        )
        with self.assertRaisesRegex(RuntimeError, "cache-active"):
            self.update([("weight", torch.zeros(1))])
        self.model.load_weights.assert_not_called()

    def worker(self, rank=0):
        ns = dict(self.ns)
        load_nodes(
            SRT / "managers/tp_worker.py",
            {"_deserialize_own_rank", "update_weights_from_tensor"},
            ns,
            "BaseTpWorker",
        )
        downstream = Mock(return_value=(True, "forwarded"))
        worker = SimpleNamespace(
            ps=SimpleNamespace(tp_rank=rank),
            model_runner=SimpleNamespace(
                weight_updater=SimpleNamespace(update_weights_from_tensor=downstream)
            ),
        )
        worker._deserialize_own_rank = lambda payload: ns["_deserialize_own_rank"](
            worker, payload
        )
        return ns["update_weights_from_tensor"], worker, downstream

    def test_worker_corrupt_payload_controlled(self):
        method, worker, downstream = self.worker()
        for payload in [[b"not-a-pickle"], [], 7]:
            result = method(
                worker,
                SimpleNamespace(serialized_named_tensors=payload, load_format=None),
            )
            self.assertFalse(result[0])
            self.assertIn("serialized payload", result[1])
        downstream.assert_not_called()

    def test_worker_deserializes_only_own_rank(self):
        method, worker, downstream = self.worker(rank=1)
        result = method(
            worker,
            SimpleNamespace(
                serialized_named_tensors=[b"corrupt-other-rank", pickle.dumps(7)],
                load_format=None,
            ),
        )
        self.assertEqual(result, (True, "forwarded"))
        downstream.assert_called_once_with(named_tensors=7, load_format=None)


if __name__ == "__main__":
    unittest.main()
