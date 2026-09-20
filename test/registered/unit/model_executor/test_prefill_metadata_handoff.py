"""Target-to-draft prefill metadata handoff, using real worker methods.

CPU storage is tagged only to exercise CUDA selectors; real CUDA equivalence is
checked separately on the authorized development CVM.
"""

import contextlib
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from test_draft_extend_device_metadata import _TaggedCuda, cpu_positions, make_batch

from sglang.srt.environ import envs
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2


class PrefillMetadataHandoffTest(unittest.TestCase):
    def setUp(self):
        self.context_override = get_context().override_server_args(
            disable_overlap_schedule=True
        )
        self.context_override.install()
        self.addCleanup(self.context_override.restore)
        self.enterContext(
            patch(
                "sglang.srt.model_executor.forward_batch_info.compute_position",
                side_effect=cpu_positions,
            )
        )
        self.enterContext(
            patch(
                "sglang.srt.model_executor.forward_batch_info.enable_num_token_non_padded",
                return_value=False,
            )
        )
        self.plan_stream = self.enterContext(
            patch.object(
                envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM, "get", return_value=False
            )
        )
        self.enterContext(
            patch("sglang.srt.managers.tp_worker.capture_pre_sample_logits")
        )
        for name in (
            "speculative_moe_backend_context",
            "speculative_moe_a2a_backend_context",
            "spec_stage_span",
        ):
            self.enterContext(
                patch(
                    "sglang.srt.speculative.eagle_worker_v2." + name,
                    side_effect=lambda *args, **kw: contextlib.nullcontext(),
                )
            )
        self.enterContext(
            patch(
                "sglang.srt.speculative.eagle_worker_v2.renorm_draft_probs",
                side_effect=lambda logits, *args: logits,
            )
        )
        self.enterContext(
            patch(
                "sglang.srt.speculative.eagle_worker_v2.fast_topk",
                side_effect=lambda probs, k, dim: (
                    probs[:, :k],
                    torch.zeros((probs.shape[0], k), dtype=torch.int64),
                ),
            )
        )
        self.enterContext(
            patch("sglang.srt.speculative.eagle_worker_v2.maybe_detect_nan")
        )
        self.enterContext(
            patch("sglang.srt.speculative.eagle_worker_v2.maybe_detect_inf")
        )
        original_init = ForwardBatch.init_new
        self.frames = []

        def construct(batch, runner, **kwargs):
            fb = original_init(batch, runner, **kwargs)
            if not runner.is_draft_worker and fb.extend_seq_lens is not None:
                fb.extend_seq_lens = fb.extend_seq_lens.as_subclass(_TaggedCuda)
                fb.extend_prefix_lens = fb.extend_prefix_lens.as_subclass(_TaggedCuda)
            self.frames.append((runner.is_draft_worker, fb, dict(kwargs)))
            return fb

        self.enterContext(patch.object(ForwardBatch, "init_new", side_effect=construct))

    def setup_chain(self, *, tagged=True, pad_target=False, draft=True):
        batch, target_runner = make_batch([3, 14], tagged=tagged)
        _, draft_runner = make_batch([3, 14])
        batch.forward_mode = ForwardMode.EXTEND
        batch.input_ids = torch.arange(7, dtype=torch.int64)
        batch.out_cache_loc = torch.arange(7, dtype=torch.int64)
        batch.extend_lens = [3, 4]
        batch.prefix_lens = [0, 10]
        batch.extend_num_tokens = 7
        batch.extend_logprob_start_lens = [0, 0]
        target_runner.is_draft_worker = False
        target_runner.canary_manager = draft_runner.canary_manager = None
        target_runner.tp_group = draft_runner.tp_group = None
        self.original_metadata = None

        def output(fb):
            return SimpleNamespace(
                logits_output=SimpleNamespace(
                    next_token_logits=torch.ones((2, 4)),
                    hidden_states=torch.zeros((7, 4)),
                    mm_input_embeds=None,
                ),
                can_run_graph=False,
                expert_distribution_metrics=None,
                routed_experts_output=None,
                indexer_topk_output=None,
            )

        def target_forward(fb, **kwargs):
            self.original_metadata = (fb.extend_seq_lens, fb.extend_prefix_lens)
            if pad_target:
                # Actual eager/MLP/graph preparation can rebind these fields.
                fb.extend_seq_lens = torch.cat(
                    (fb.extend_seq_lens, torch.tensor([0], dtype=torch.int32))
                )
                fb.extend_prefix_lens = torch.cat(
                    (fb.extend_prefix_lens, torch.tensor([0], dtype=torch.int32))
                )
            return output(fb)

        target_runner.forward = target_forward
        target_runner.sample = lambda logits, fb: torch.tensor(
            [91, 92], dtype=torch.int64
        )
        draft_runner.forward = lambda fb: output(fb)
        target = SimpleNamespace(
            model_runner=target_runner,
            pp_group=SimpleNamespace(is_last_rank=True),
            enable_overlap=False,
            enable_spec=True,
            is_dllm=lambda: False,
            set_hicache_consumer=Mock(),
        )
        self.target_kwargs = []

        def target_call(batch, **kwargs):
            self.target_kwargs.append(dict(kwargs))
            result = TpModelWorker.forward_batch_generation(target, batch, **kwargs)
            self.target_result = result
            return result

        target.forward_batch_generation = target_call
        draft_worker = SimpleNamespace(
            draft_runner=draft_runner,
            speculative_algorithm=SimpleNamespace(is_standalone=lambda: False),
            seed_dsa_topk_from_draft_extend=False,
            topk=1,
            draft_tp_context=lambda *args: contextlib.nullcontext(),
        )
        draft_worker._draft_extend_for_prefill = MethodType(
            EagleDraftWorker._draft_extend_for_prefill, draft_worker
        )
        outer = SimpleNamespace(
            target_worker=target,
            _draft_worker=draft_worker if draft else None,
            draft_worker=draft_worker if draft else None,
            speculative_algorithm=SimpleNamespace(is_standalone=lambda: False),
        )
        return batch, target, draft_worker, outer

    def run_chain(self, batch, outer):
        copies = []
        original_to = torch.Tensor.to

        def move(tensor, *args, **kwargs):
            if (
                tensor.dtype == torch.int32
                and kwargs.get("non_blocking")
                and args
                and args[0] == "cpu"
            ):
                copies.append(tensor.tolist())
            return original_to(tensor, *args, **kwargs)

        def published(lengths):
            self.assertIs(lengths, batch.seq_lens)
            self.assertIsNone(
                getattr(self.target_result, "prefill_extend_metadata", None)
            )

        with patch.object(torch.Tensor, "to", new=move):
            result = EAGLEWorkerV2.forward_batch_generation(
                outer, batch, on_publish=published
            )
        return result, copies

    def test_only_target_uploads_metadata_and_real_cpu_mirrors_remain(self):
        batch, target, draft, outer = self.setup_chain()
        cpu_extend, cpu_prefix = batch.extend_lens, batch.prefix_lens
        gpu_lengths, cpu_lengths = batch.seq_lens, batch.seq_lens_cpu
        result, copies = self.run_chain(batch, outer)
        self.assertEqual(
            copies,
            [[3, 4], [0, 10]],
            "Draft prefill repeated the target metadata upload",
        )
        draft_fb = self.frames[-1][1]
        self.assertIs(draft_fb.extend_seq_lens, self.original_metadata[0])
        self.assertIs(draft_fb.extend_prefix_lens, self.original_metadata[1])
        self.assertIs(draft_fb.extend_seq_lens_cpu, cpu_extend)
        self.assertIs(draft_fb.extend_prefix_lens_cpu, cpu_prefix)
        self.assertEqual(cpu_extend, [3, 4])
        self.assertEqual(cpu_prefix, [0, 10])
        self.assertIs(batch.seq_lens, gpu_lengths)
        self.assertIs(batch.seq_lens_cpu, cpu_lengths)
        self.assertEqual(batch.seq_lens_sum, 17)
        self.assertEqual(draft_fb.positions.tolist(), [0, 1, 2, 10, 11, 12, 13])
        self.assertEqual(draft_fb.input_ids.tolist(), [1, 2, 91, 4, 5, 6, 92])
        self.assertEqual(self.frames[0][1].capture_hidden_mode, CaptureHiddenMode.FULL)
        self.assertEqual(draft_fb.capture_hidden_mode, CaptureHiddenMode.LAST)
        self.assertIsNone(result.prefill_extend_metadata)
        self.assertIsNotNone(result.next_draft_input)
        target.set_hicache_consumer.assert_called_once_with(
            batch.hicache_consumer_index
        )

    def test_target_padding_rebind_keeps_original_unpadded_metadata(self):
        batch, _, _, outer = self.setup_chain(pad_target=True)
        _, copies = self.run_chain(batch, outer)
        target_fb, draft_fb = self.frames[0][1], self.frames[-1][1]
        self.assertEqual(target_fb.extend_seq_lens.shape, (3,))
        self.assertEqual(target_fb.extend_prefix_lens.shape, (3,))
        self.assertEqual(draft_fb.extend_seq_lens.shape, (2,))
        self.assertIs(draft_fb.extend_seq_lens, self.original_metadata[0])
        self.assertIs(draft_fb.extend_prefix_lens, self.original_metadata[1])
        self.assertEqual(copies, [[3, 4], [0, 10]])

    def test_cpu_and_overlap_modes_keep_original_construction(self):
        for mode in (
            "cpu",
            "batch_overlap",
            "worker_overlap",
            "global_overlap",
            "plan_stream",
        ):
            with self.subTest(mode=mode):
                batch, target, _, outer = self.setup_chain(tagged=mode != "cpu")
                batch.enable_overlap = mode == "batch_overlap"
                target.enable_overlap = mode == "worker_overlap"
                self.plan_stream.return_value = mode == "plan_stream"
                with patch(
                    "sglang.srt.speculative.eagle_worker_v2.get_schedule",
                    return_value=SimpleNamespace(
                        disable_overlap_schedule=mode != "global_overlap"
                    ),
                ), patch(
                    "sglang.srt.managers.tp_worker.get_schedule",
                    return_value=SimpleNamespace(
                        disable_overlap_schedule=mode != "global_overlap"
                    ),
                ):
                    result, copies = self.run_chain(batch, outer)
                self.assertEqual(copies, [[3, 4], [0, 10], [3, 4], [0, 10]])
                self.assertIsNone(result.prefill_extend_metadata)

    def test_no_draft_does_not_retain_metadata(self):
        batch, _, _, outer = self.setup_chain(draft=False)
        result, copies = self.run_chain(batch, outer)
        self.assertEqual(copies, [[3, 4], [0, 10]])
        self.assertNotIn("capture_prefill_extend_metadata", self.target_kwargs[-1])
        self.assertIsNone(result.prefill_extend_metadata)

    def test_prebuilt_forward_batch_cannot_export_borrowed_metadata(self):
        batch, target, _, _ = self.setup_chain()
        fb = ForwardBatch.init_new(
            batch,
            target.model_runner,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_hidden_states_before_norm=False,
        )
        result = TpModelWorker.forward_batch_generation(
            target, None, forward_batch=fb, capture_prefill_extend_metadata=True
        )
        self.assertIsNone(result.prefill_extend_metadata)
        target.set_hicache_consumer.assert_not_called()

    def test_padded_cpu_prefix_keeps_original_representation(self):
        batch, _, draft, _ = self.setup_chain()
        batch.prefix_lens.append(0)
        metadata = (
            torch.tensor([3, 4], dtype=torch.int32),
            torch.tensor([0, 10], dtype=torch.int32),
        )
        draft._draft_extend_for_prefill(
            batch, torch.zeros((7, 4)), torch.tensor([91, 92]), extend_metadata=metadata
        )
        draft_fb = self.frames[-1][1]
        self.assertNotIn("extend_metadata", self.frames[-1][2])
        self.assertEqual(draft_fb.extend_prefix_lens_cpu, [0, 10, 0])
        self.assertEqual(batch.prefix_lens, [0, 10, 0])


if __name__ == "__main__":
    unittest.main()
