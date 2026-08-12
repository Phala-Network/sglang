import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    FlattenedTensorMetadata,
    LocalSerializedTensor,
    WeightUpdater,
    _validate_update_weights_from_tensor_payload,
)
from sglang.srt.managers.tp_worker import BaseTpWorker
from sglang.srt.utils import MultiprocessingSerializer


class FakeTpWorker:
    _deserialize_own_rank = BaseTpWorker._deserialize_own_rank

    def __init__(self, updater):
        self.ps = SimpleNamespace(tp_rank=0)
        self.model_runner = SimpleNamespace(weight_updater=updater)


class TestMalformedWeightUpdatePayload(unittest.TestCase):
    def test_rejects_non_collection(self):
        self.assertIn(
            "named_tensors must be",
            _validate_update_weights_from_tensor_payload(7, None),
        )

    @patch(
        "sglang.srt.model_executor.model_runner_components.weight_updater."
        "_unsupported_derived_weight_cache_error",
        return_value=None,
    )
    @patch(
        "sglang.srt.model_executor.model_runner_components.weight_updater."
        "monkey_patch_torch_reductions"
    )
    def test_weight_updater_rejects_integer_without_raising(self, _patch, _error):
        fake_updater = SimpleNamespace(
            _assert_weight_cache_inactive=Mock(),
            tp_rank=0,
            custom_weight_loaders={},
        )
        success, message = WeightUpdater.update_weights_from_tensor(
            fake_updater, 7, None
        )
        self.assertFalse(success)
        self.assertIn("named_tensors must be", message)
        fake_updater._assert_weight_cache_inactive.assert_called_once_with(
            "update_weights_from_tensor"
        )

    def test_tp_worker_rejects_corrupt_serialization(self):
        fake_worker = FakeTpWorker(
            SimpleNamespace(update_weights_from_tensor=Mock())
        )
        request = SimpleNamespace(
            serialized_named_tensors=[b"not-a-pickle"], load_format=None
        )
        success, message = BaseTpWorker.update_weights_from_tensor(
            fake_worker, request
        )
        self.assertFalse(success)
        self.assertIn("Invalid update_weights_from_tensor serialized payload", message)
        fake_worker.model_runner.weight_updater.update_weights_from_tensor.assert_not_called()

    def test_tp_worker_forwards_valid_serialized_shape(self):
        fake_updater = Mock(return_value=(False, "validated"))
        fake_worker = FakeTpWorker(
            SimpleNamespace(update_weights_from_tensor=fake_updater)
        )
        request = SimpleNamespace(
            serialized_named_tensors=[MultiprocessingSerializer.serialize(7)],
            load_format=None,
        )
        self.assertEqual(
            BaseTpWorker.update_weights_from_tensor(fake_worker, request),
            (False, "validated"),
        )
        fake_updater.assert_called_once_with(named_tensors=7, load_format=None)

    def test_rejects_malformed_entries(self):
        cases = [
            [7],
            [("name",)],
            [(7, torch.zeros(1))],
            [("name", 7)],
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertIsNotNone(
                    _validate_update_weights_from_tensor_payload(payload, None)
                )

    def test_accepts_named_tensor_entries(self):
        payload = [
            ("weight", torch.zeros(1)),
            ("local", LocalSerializedTensor(values=[b"payload"])),
        ]
        self.assertIsNone(
            _validate_update_weights_from_tensor_payload(payload, None)
        )

    @patch(
        "sglang.srt.model_executor.model_runner_components.weight_updater."
        "_unsupported_derived_weight_cache_error",
        return_value=None,
    )
    @patch(
        "sglang.srt.model_executor.model_runner_components.weight_updater."
        "monkey_patch_torch_reductions"
    )
    def test_rejects_invalid_local_tensor_without_raising(self, _patch, _error):
        fake_updater = SimpleNamespace(
            _assert_weight_cache_inactive=Mock(),
            tp_rank=0,
            device="cpu",
            custom_weight_loaders={},
        )
        payload = [
            (
                "weight",
                LocalSerializedTensor(
                    values=[MultiprocessingSerializer.serialize(7)]
                ),
            )
        ]
        success, message = WeightUpdater.update_weights_from_tensor(
            fake_updater, payload, None
        )
        self.assertFalse(success)
        self.assertIn("failed to unwrap tensor", message)

    @patch(
        "sglang.srt.model_executor.model_runner_components.weight_updater."
        "_unsupported_derived_weight_cache_error",
        return_value=None,
    )
    @patch(
        "sglang.srt.model_executor.model_runner_components.weight_updater."
        "monkey_patch_torch_reductions"
    )
    def test_rejects_unknown_load_format_without_raising(self, _patch, _error):
        fake_updater = SimpleNamespace(
            _assert_weight_cache_inactive=Mock(),
            tp_rank=0,
            custom_weight_loaders={},
        )
        success, message = WeightUpdater.update_weights_from_tensor(
            fake_updater, [("weight", torch.zeros(1))], "unknown"
        )
        self.assertFalse(success)
        self.assertIn("unknown load_format", message)

    def test_validates_flattened_bucket_shape(self):
        valid = {
            "flattened_tensor": torch.zeros(4, dtype=torch.uint8),
            "metadata": [
                FlattenedTensorMetadata(
                    name="weight",
                    shape=torch.Size([1]),
                    dtype=torch.float32,
                    start_idx=0,
                    end_idx=4,
                    numel=4,
                )
            ],
        }
        invalid = [
            7,
            {},
            {"flattened_tensor": 7, "metadata": []},
            {"flattened_tensor": torch.zeros(1), "metadata": 7},
            {"flattened_tensor": torch.zeros(1), "metadata": [7]},
            {
                "flattened_tensor": torch.zeros(1, dtype=torch.uint8),
                "metadata": [
                    FlattenedTensorMetadata(
                        name="weight",
                        shape=torch.Size([1]),
                        dtype=torch.float32,
                        start_idx=0,
                        end_idx=4,
                        numel=4,
                    )
                ],
            },
        ]
        self.assertIsNone(
            _validate_update_weights_from_tensor_payload(valid, "flattened_bucket")
        )
        for payload in invalid:
            with self.subTest(payload=payload):
                self.assertIsNotNone(
                    _validate_update_weights_from_tensor_payload(
                        payload, "flattened_bucket"
                    )
                )


if __name__ == "__main__":
    unittest.main()
