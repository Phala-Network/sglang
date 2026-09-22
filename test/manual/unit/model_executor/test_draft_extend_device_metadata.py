"""Real draft-extend/ForwardBatch boundary, including retained CPU mirrors.

CPU tests tag CPU storage as CUDA only to exercise the selector; they do not
claim real transfers or GPU correctness. The 805 CUDA check uses real tensors.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.eagle_worker_common import prepare_for_draft_extend


class _TaggedCuda(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def make_batch(lengths, *, device="cpu", tagged=False, cpu_mirror=True):
    gpu = torch.tensor(lengths, dtype=torch.int64, device=device)
    if tagged:
        gpu = gpu.as_subclass(_TaggedCuda)
    batch = ScheduleBatch(
        reqs=[
            SimpleNamespace(rid=str(i), lora_id=None, token_type_ids=None)
            for i in range(len(lengths))
        ],
        device=device,
        model_config=SimpleNamespace(vocab_size=128),
        seq_lens=gpu,
        seq_lens_cpu=torch.tensor(lengths, dtype=torch.int64) if cpu_mirror else None,
        seq_lens_sum=sum(lengths) if cpu_mirror else None,
        req_pool_indices=torch.arange(len(lengths), dtype=torch.int64, device=device),
        input_ids=torch.zeros(len(lengths) * 4, dtype=torch.int64, device=device),
        out_cache_loc=torch.zeros(len(lengths) * 4, dtype=torch.int64, device=device),
        forward_mode=ForwardMode.DECODE,
    )
    runner = SimpleNamespace(
        device=device,
        is_draft_worker=True,
        spec_algorithm=SimpleNamespace(is_standalone=lambda: False),
        kv_index_translator=SimpleNamespace(rebind_write_loc=lambda _: None),
        ngram_embedding_manager=SimpleNamespace(enabled=False),
        model_config=SimpleNamespace(model_is_mrope=False),
        lora_manager=None,
        ps=SimpleNamespace(attn_dcp_size=1),
        prefill_attention_backend_str="fa3",
        attn_backend=SimpleNamespace(init_forward_metadata=Mock()),
    )
    return batch, runner


def cpu_positions(backend, prefix, lengths, total):
    positions = torch.cat(
        [
            torch.arange(int(p), int(p) + int(n), dtype=torch.int64)
            for p, n in zip(prefix, lengths)
        ]
    )
    starts = torch.cumsum(lengths, dim=0, dtype=torch.int32) - lengths
    return positions, starts


class DraftExtendDeviceMetadataTest(unittest.TestCase):
    def setUp(self):
        override = get_context().override_server_args(disable_overlap_schedule=True)
        override.install()
        self.addCleanup(override.restore)
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

    def prepare(self, batch, runner, *, front=0, graph=False):
        info = EagleDraftExtendInput(hidden_states=None, num_front_tokens=front)
        graph_runner = SimpleNamespace(can_run_graph=lambda _: graph)
        count = len(batch.seq_lens) * (4 + front)
        return prepare_for_draft_extend(
            info,
            batch,
            torch.zeros(len(batch.seq_lens) * 4, dtype=torch.int64),
            4,
            runner,
            graph_runner,
            return_hidden_states_before_norm=True,
            widened_out_cache_loc=(
                torch.zeros(count, dtype=torch.int64) if front else None
            ),
            widened_positions=torch.arange(count, dtype=torch.int64) if front else None,
        )

    def capture_host_transfers(self, batch, runner):
        copies = []
        original = torch.Tensor.to

        def move(tensor, *args, **kwargs):
            if (
                tensor.shape == (len(batch.seq_lens),)
                and tensor.dtype == torch.int32
                and kwargs.get("non_blocking")
                and args
                and args[0] == runner.device
            ):
                copies.append(tensor.tolist())
            return original(tensor, *args, **kwargs)

        return copies, patch.object(torch.Tensor, "to", new=move)

    def test_cpu_mirror_draft_does_not_copy_the_same_lengths_back_to_device(self):
        batch, runner = make_batch([8000, 26000], tagged=True)
        gpu, cpu = batch.seq_lens, batch.seq_lens_cpu
        copies, guard = self.capture_host_transfers(batch, runner)
        with guard:
            result = self.prepare(batch, runner)
        self.assertEqual(
            copies, [], "Draft metadata redundantly retransferred from host"
        )
        self.assertIs(batch.seq_lens, gpu)
        self.assertIs(batch.seq_lens_cpu, cpu)
        self.assertEqual(cpu.tolist(), [8000, 26000])
        self.assertEqual(result.seq_lens.tolist(), [8004, 26004])
        self.assertEqual(result.seq_lens_cpu.tolist(), [8004, 26004])
        self.assertEqual(result.seq_lens_sum, 34008)
        self.assertEqual(result.extend_seq_lens_cpu, [4, 4])
        self.assertEqual(result.extend_prefix_lens_cpu, [8000, 26000])
        self.assertEqual(result.extend_start_loc.tolist(), [0, 4])
        self.assertEqual(
            result.positions.tolist(),
            list(range(8000, 8004)) + list(range(26000, 26004)),
        )
        self.assertEqual(result.capture_hidden_mode, CaptureHiddenMode.FULL)
        self.assertTrue(result.return_hidden_states_before_norm)
        self.assertTrue(result.forward_metadata_ready)
        runner.attn_backend.init_forward_metadata.assert_called_once_with(result)

    def test_widened_prefix_clamp_and_post_write_cpu_sum_are_preserved(self):
        batch, runner = make_batch([1, 5, 262140], tagged=True)
        result = self.prepare(batch, runner, front=2)
        self.assertEqual(result.extend_prefix_lens.tolist(), [0, 3, 262138])
        self.assertEqual(result.extend_prefix_lens_cpu, [0, 3, 262138])
        self.assertEqual(result.extend_seq_lens.tolist(), [6, 6, 6])
        self.assertEqual(result.extend_seq_lens_cpu, [6, 6, 6])
        self.assertEqual(result.seq_lens_cpu.tolist(), [5, 9, 262144])
        self.assertEqual(result.seq_lens_sum, 262158)
        self.assertEqual(result.positions.tolist(), list(range(18)))

    def test_per_forward_tensors_do_not_rewrite_scheduler_lengths(self):
        batch, runner = make_batch([17, 8000], tagged=True)
        result = self.prepare(batch, runner)
        result.extend_prefix_lens.add_(100)
        result.seq_lens.add_(100)
        result.seq_lens_cpu.add_(100)
        self.assertEqual(batch.seq_lens.tolist(), [17, 8000])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [17, 8000])
        self.assertEqual(batch.prefix_lens, [17, 8000])
        self.assertEqual(batch.extend_lens, [4, 4])

    def test_overlap_and_plan_stream_keep_the_original_metadata_path(self):
        for kind in ("batch_overlap", "global_overlap", "plan_stream"):
            with self.subTest(kind=kind):
                batch, runner = make_batch([8000], tagged=True)
                override = get_context().override_server_args(
                    disable_overlap_schedule=kind != "global_overlap"
                )
                override.install()
                try:
                    batch.enable_overlap = kind == "batch_overlap"
                    self.plan_stream.return_value = kind == "plan_stream"
                    copies, guard = self.capture_host_transfers(batch, runner)
                    with guard:
                        result = self.prepare(batch, runner)
                    self.assertEqual(copies, [[4], [8000]])
                    self.assertEqual(result.extend_prefix_lens_cpu, [8000])
                finally:
                    override.restore()
                    self.plan_stream.return_value = False

    def test_native_cpu_device_keeps_existing_host_metadata(self):
        batch, runner = make_batch([41])
        copies, guard = self.capture_host_transfers(batch, runner)
        with guard:
            result = self.prepare(batch, runner)
        self.assertEqual(copies, [[4], [41]])
        self.assertEqual(result.seq_lens_cpu.tolist(), [45])

    def test_gpu_only_path_still_has_no_fabricated_cpu_seq_lengths(self):
        batch, runner = make_batch([41], tagged=True, cpu_mirror=False)
        result = self.prepare(batch, runner)
        self.assertIsNone(result.seq_lens_cpu)
        self.assertEqual(result.extend_seq_lens_cpu, [4])
        self.assertIsNone(result.extend_prefix_lens_cpu)
        self.assertEqual(result.seq_lens.tolist(), [45])

    def test_graph_path_retains_mirrors_without_eager_metadata_planning(self):
        batch, runner = make_batch([15, 200], tagged=True)
        result = self.prepare(batch, runner, graph=True)
        self.assertEqual(result.extend_prefix_lens_cpu, [15, 200])
        self.assertEqual(result.seq_lens_sum, 223)
        runner.attn_backend.init_forward_metadata.assert_not_called()
        self.assertFalse(result.forward_metadata_ready)

    def test_empty_idle_batch_has_no_metadata_transfer_or_eager_plan(self):
        batch, runner = make_batch([], tagged=True)
        batch.forward_mode = ForwardMode.IDLE
        result = self.prepare(batch, runner)
        self.assertEqual(result.positions.numel(), 0)
        self.assertEqual(result.seq_lens_sum, 0)
        runner.attn_backend.init_forward_metadata.assert_not_called()

    def test_invalid_device_override_fails_before_position_use(self):
        for override in (
            (torch.zeros(1, dtype=torch.int32),),
            (torch.zeros(2), torch.zeros(2)),
            (torch.zeros(3, dtype=torch.int32), torch.zeros(3, dtype=torch.int32)),
            (None, None),
            (
                torch.zeros(2, dtype=torch.int32, device="meta"),
                torch.zeros(2, dtype=torch.int32, device="meta"),
            ),
        ):
            with self.subTest(override=override):
                batch, runner = make_batch([11, 20])
                batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
                batch.extend_lens = [4, 4]
                batch.prefix_lens = [11, 20]
                batch.extend_num_tokens = 8
                with self.assertRaises(ValueError):
                    ForwardBatch.init_new(
                        batch,
                        runner,
                        capture_hidden_mode=CaptureHiddenMode.FULL,
                        return_hidden_states_before_norm=False,
                        extend_metadata=override,
                    )

    def test_override_is_not_silently_ignored_in_non_extend_modes(self):
        for mode in (ForwardMode.DECODE, ForwardMode.IDLE, ForwardMode.TARGET_VERIFY):
            batch, runner = make_batch([11])
            batch.forward_mode = mode
            with self.assertRaises(ValueError):
                ForwardBatch.init_new(
                    batch,
                    runner,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    return_hidden_states_before_norm=False,
                    extend_metadata=(
                        torch.ones(1, dtype=torch.int32),
                        torch.ones(1, dtype=torch.int32),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
