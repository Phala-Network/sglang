import unittest

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
    compute_visual_patch_tokens,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMultimodalPatchBudget(unittest.TestCase):
    def test_counts_image_and_video_grids(self):
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                model_specific_data={
                    "image_grid_thw": torch.tensor([[1, 4, 8], [1, 2, 2]])
                },
            ),
            MultimodalDataItem(
                modality=Modality.VIDEO,
                model_specific_data={"video_grid_thw": np.array([2, 3, 5])},
            ),
            MultimodalDataItem(modality=Modality.AUDIO),
        ]

        self.assertEqual(compute_visual_patch_tokens(items), [32, 4, 30])

    def test_processor_output_preserves_cached_patch_counts(self):
        output = MultimodalProcessorOutput(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.IMAGE, hash=1, pad_value=1
                )
            ],
            visual_patch_tokens=[64, 32],
        )

        inputs = MultimodalInputs.from_processor_output(output)

        self.assertEqual(inputs.total_visual_patch_tokens(), 96)

    def test_server_args_require_positive_limits(self):
        for name in (
            "max_mm_patch_tokens_per_request",
            "max_prefill_mm_patch_tokens",
        ):
            args = ServerArgs(model_path="dummy", **{name: 0})
            with self.assertRaisesRegex(ValueError, "must be positive"):
                args._validate_mm_patch_budgets()


if __name__ == "__main__":
    unittest.main()
