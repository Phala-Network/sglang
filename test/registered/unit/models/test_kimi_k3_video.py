"""CPU checks for bounded Kimi-K3 video staging and owner-side encoding."""

import asyncio
import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import av
import numpy as np
import pytest
import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
from sglang.srt.models.kimi_k3_vl import Learnable2DInterpPosEmbDividedFixed
from sglang.srt.multimodal.kimi_k3_image_processing import (
    DEFERRED_PREPROCESSING_KEY,
    KimiK3DeferredPreprocessing,
)
from sglang.srt.multimodal.media_artifacts.kimi_k3 import KimiK3PreprocessConfig
from sglang.srt.multimodal.mm_utils import (
    iter_frame_bounded_image_batches,
    run_dp_sharded_mrope_vision_model,
)
from sglang.srt.multimodal.processors.kimi_k3 import (
    KimiK3ImageProcessor,
    _plan_k3_video,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.utils import VideoData
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Video:
    def __init__(self, frames=8, fps=8.0, shape=(8, 12), value=17):
        self.frame_count = frames
        self.avg_fps = fps
        self.frame_shape = shape
        self.value = value
        self.calls = []
        self.closed = threading.Event()

    def __len__(self):
        return self.frame_count

    def get_frames_at(self, indices):
        self.calls.append(tuple(indices))
        return np.full((len(indices), *self.frame_shape, 3), self.value, np.uint8)

    def get_frames_as_tensor(self, _indices):
        raise AssertionError("video staging must not start parallel decoders")

    def close(self):
        self.closed.set()


class _Tokenizer:
    def encode(self, text, **_kwargs):
        return [14] if text == "<|media_end|>" else [10, 11]


def _processor(*videos):
    processor = object.__new__(KimiK3ImageProcessor)
    processor.mm_tokens = SimpleNamespace(image_token_id=99)
    processor.media_proc_cfg = {
        "patch_size": 2,
        "merge_kernel_size": 2,
        "patch_limit_on_one_side": 512,
        "fixed_output_tokens": None,
        "in_patch_limit": 100,
        "in_patch_limit_each_frame": 100,
        "in_patch_limit_video": 800,
        "sample_fps": 8.0,
        "temporal_merge_kernel_size": 4,
        "timestamp_mode": "hh:mm:ss.fff",
    }
    processor._processor = SimpleNamespace(
        preprocess_config=KimiK3PreprocessConfig(
            patch_size=2,
            merge_kernel_size=2,
            in_patch_limit=100,
            patch_limit_on_one_side=512,
            fixed_output_tokens=None,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
            transparent_bg_config=None,
        )
    )
    processor._tokenizer = _Tokenizer()
    processor.processor_fingerprint = "k3-video-cpu-test"
    processor.server_args = SimpleNamespace(context_length=100000)
    processor.hf_config = SimpleNamespace()
    processor._precompute_hashes_before_cpu_transfer = Mock()
    if videos:
        processor._load_single_item = Mock(side_effect=videos)
    return processor


def _request(count=1, **kwargs):
    return SimpleNamespace(
        video_data=kwargs.pop("video_data", [f"video-{i}" for i in range(count)]),
        modalities=["video"] * count,
        **kwargs,
    )


def _process(processor, request=None, cancelled=None):
    request = request or _request()
    return processor._process_video_data_sync(
        [], [1, *([99] * len(request.video_data)), 2], request, cancelled
    )


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_video_rejects_bad_metadata_before_decode(fps):
    video = _Video(fps=fps)
    with pytest.raises(ValueError, match="positive frame rate"):
        _process(_processor(video))
    assert not video.calls
    assert video.closed.is_set()


def test_video_metadata_budget_rejects_before_indices_or_later_sources(monkeypatch):
    video = _Video(frames=10**9, shape=(2160, 3840))
    processor = _processor(video, _Video())
    monkeypatch.setenv("SGLANG_K3_VIDEO_MAX_DECODE_BYTES", "100000")
    with pytest.raises(ValueError, match="decode byte budget"):
        _process(processor, _request(2))
    processor._load_single_item.assert_called_once()
    assert not video.calls
    assert video.closed.is_set()


def test_video_cumulative_metadata_budget_is_checked_each_source(monkeypatch):
    videos = [_Video(), _Video(), _Video()]
    processor = _processor(*videos)
    plan = _plan_k3_video(videos[0], processor.media_proc_cfg, 0)
    monkeypatch.setenv(
        "SGLANG_K3_VIDEO_MAX_DECODE_BYTES",
        str(3 * plan["retained_bytes"] + plan["scratch_bytes"]),
    )
    with pytest.raises(ValueError, match="decode byte budget"):
        _process(processor, _request(3))
    assert processor._load_single_item.call_count == 2
    assert all(not video.calls for video in videos)
    assert all(video.closed.is_set() for video in videos[:2])
    assert not videos[2].closed.is_set()


def test_video_nonpositive_budget_rejects_before_loading(monkeypatch):
    processor = _processor(_Video())
    monkeypatch.setenv("SGLANG_K3_VIDEO_MAX_DECODE_BYTES", "0")
    with pytest.raises(ValueError, match="budget must be positive"):
        _process(processor)
    processor._load_single_item.assert_not_called()


@pytest.mark.parametrize("options", [{"fps": 0}, {"max_frames": -1}, {"foo": 1}])
def test_video_sampling_options_fail_before_loading(options):
    processor = _processor(_Video())
    with pytest.raises(ValueError):
        _process(processor, _request(video_data=[VideoData("video", options)]))
    processor._load_single_item.assert_not_called()


def test_video_context_lower_bound_rejects_before_decode():
    video = _Video()
    processor = _processor(video)
    processor.server_args.context_length = 1
    with pytest.raises(ValueError, match="model context budget"):
        _process(processor)
    assert not video.calls
    assert video.closed.is_set()


def test_video_context_planner_does_not_guess_twenty_header_tokens():
    video = _Video(frames=1, shape=(4, 4))
    processor = _processor(video)
    plan = _plan_k3_video(video, processor.media_proc_cfg, 0)
    # Actual stub prefix+suffix is 3 tokens. The standard final context check
    # owns exact expansion; an invented 20-token header would reject this.
    processor.server_args.context_length = plan["visual_tokens"] + 5
    output = _process(processor)
    assert len(output.input_ids) == processor.server_args.context_length


def test_video_sampling_stays_chunked_and_reuses_original_resize_plan():
    video = _Video(frames=10**9, shape=(2, 256))
    processor = _processor(video)
    request = _request(video_data=[VideoData("video", {"max_frames": 9})])
    with patch(
        "sglang.srt.multimodal.processors.kimi_k3.navit_resize_config",
        wraps=__import__(
            "sglang.srt.multimodal.processors.kimi_k3", fromlist=["navit_resize_config"]
        ).navit_resize_config,
    ) as resize:
        output = _process(processor, request)
    resize.assert_called_once()
    assert [len(call) for call in video.calls] == [4, 4, 1]
    assert video.calls[0][0] == 0 and video.calls[-1][-1] == 10**9 - 1
    assert [item.image_grid_thw[0, 0].item() for item in output.mm_items] == [4, 4, 1]
    resize_configs = [
        item.model_specific_data[DEFERRED_PREPROCESSING_KEY].resize_config
        for item in output.mm_items
    ]
    assert resize_configs[0] == resize_configs[1] == resize_configs[2]
    assert all(item.feature.dtype == torch.uint8 for item in output.mm_items)


def test_video_sample_count_avoids_finite_fps_multiplication_overflow():
    video = _Video(frames=10, fps=1e308)
    config = dict(_processor().media_proc_cfg, sample_fps=1e308)
    assert _plan_k3_video(video, config, 0)["sampled_nframes"] == 10


def test_video_identity_covers_later_frames_and_sampling_contract():
    first = _process(_processor(_Video()))
    changed = _Video()
    real_decode = changed.get_frames_at

    def decode(indices):
        frames = real_decode(indices)
        if 3 in indices:
            frames[indices.index(3), 0, 0, 0] += 1
        return frames

    changed.get_frames_at = decode
    second = _process(_processor(changed))
    third = _process(_processor(_Video(fps=16)))
    assert first.mm_items[0].identity != second.mm_items[0].identity
    assert first.mm_items[0].identity != third.mm_items[0].identity
    assert len(first.mm_items[0].identity) == 32
    assert first.video_token_id == first.im_token_id == 99


def test_video_bad_decode_closes_owner_and_returns_client_error():
    video = _Video()
    video.get_frames_at = Mock(side_effect=RuntimeError("corrupt synthetic payload"))
    with pytest.raises(ValueError, match="Could not decode Kimi-K3 video"):
        _process(_processor(video))
    assert video.closed.is_set()


def test_video_internal_resize_failure_is_not_a_client_error():
    video = _Video(shape=(256, 256))
    with patch(
        "sglang.srt.multimodal.processors.kimi_k3._resize_bicubic_if_needed",
        side_effect=RuntimeError("internal resize failure"),
    ):
        with pytest.raises(RuntimeError, match="internal resize failure"):
            _process(_processor(video))
    assert video.closed.is_set()


def test_video_cancel_mid_decode_closes_owner():
    video = _Video()
    cancelled = threading.Event()
    real_decode = video.get_frames_at

    def decode(indices):
        frames = real_decode(indices)
        cancelled.set()
        return frames

    video.get_frames_at = decode
    with pytest.raises(ValueError, match="cancelled"):
        _process(_processor(video), cancelled=cancelled)
    assert len(video.calls) == 1
    assert video.closed.is_set()


def test_video_async_cancel_keeps_cleanup_in_io_worker():
    started, release = threading.Event(), threading.Event()
    video = _Video()
    real_decode = video.get_frames_at

    def decode(indices):
        started.set()
        assert release.wait(5), "test failed to release decoder"
        return real_decode(indices)

    video.get_frames_at = decode
    processor = _processor(video)

    async def run():
        task = asyncio.create_task(
            processor._process_video_data_async([], [1, 99, 2], _request())
        )
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert await asyncio.to_thread(video.closed.wait, 5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        processor.io_executor = executor
        try:
            asyncio.run(run())
        finally:
            release.set()


def _video_item(value, frames=4):
    resize = {
        "new_width": 4,
        "new_height": 4,
        "pad_width": 0,
        "pad_height": 0,
        "num_tokens": 1,
    }
    return MultimodalDataItem(
        modality=Modality.VIDEO,
        offsets=[(0, 0)],
        feature=torch.full((frames, 3, 4, 4), value, dtype=torch.uint8),
        model_specific_data={
            "image_grid_thw": torch.tensor([[frames, 2, 2]]),
            DEFERRED_PREPROCESSING_KEY: KimiK3DeferredPreprocessing(
                backend="gpu_video",
                image_mean=[0.0] * 3,
                image_std=[1.0] * 3,
                transparent_bg_config=None,
                resize_config=resize,
            ),
        },
    )


class _StaticTower:
    device = torch.device("cpu")
    merge_kernel_size = (2, 2)
    patch_size = 2

    def __init__(self, list_output):
        self.config = SimpleNamespace(hidden_size=2)
        self.patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=torch.empty(1)))
        self.buffer = torch.empty(1, 4, 2)
        self.calls = []
        self.list_output = list_output

    def __call__(self, pixels, *, grid_hw, grid_thw_list, max_seqlen):
        self.calls.append((pixels.shape[0], grid_thw_list))
        self.buffer.fill_(pixels[0, 0, 0, 0])
        return [self.buffer] if self.list_output else self.buffer


def _model(list_output=True, data_parallel=False):
    model = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.use_data_parallel = data_parallel
    model.vision_tower = _StaticTower(list_output)
    model.mm_projector = lambda value: value
    return model


@pytest.mark.parametrize("list_output", [False, True])
def test_video_owner_batches_own_graph_replay_outputs(list_output):
    model = _model(list_output)
    with patch(
        "sglang.srt.multimodal.processors.kimi_k3._k3_to_cuda_chw",
        side_effect=lambda value: value,
    ):
        output = model.get_image_feature([_video_item(51), _video_item(102)])
    assert model.vision_tower.calls == [(16, [[4, 2, 2]]), (16, [[4, 2, 2]])]
    torch.testing.assert_close(output[:, 0, 0], torch.tensor([0.2, 0.4]))
    model.vision_tower.buffer.fill_(-1)
    torch.testing.assert_close(output[:, 0, 0], torch.tensor([0.2, 0.4]))


@pytest.mark.parametrize("rank", [0, 1])
def test_video_all_ranks_validate_shared_temporal_grid_before_collectives(rank):
    model = _model(data_parallel=True)
    with (
        get_context().override_server_args(tp_size=2),
        get_parallel().override(tp_size=2, attn_tp_size=2, attn_tp_rank=rank),
        patch("sglang.srt.multimodal.mm_utils.run_dp_sharded_mrope_vision_model") as dp,
    ):
        with pytest.raises(ValueError, match="vision frame budget"):
            model.get_image_feature([_video_item(1), _video_item(2, frames=5)])
    dp.assert_not_called()
    assert not model.vision_tower.calls


def test_video_frame_batches_preserve_complete_temporal_items():
    grids = [[4, 2, 2], [1, 2, 2], [3, 2, 2], [4, 2, 2]]
    assert list(iter_frame_bounded_image_batches([3, 1, 2, 0], grids, 4)) == [
        [3],
        [1, 2],
        [0],
    ]


@pytest.mark.parametrize("rank", [0, 1])
def test_video_dp_local_encoder_handles_owner_and_empty_rank(rank):
    embedding = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)

    class Group:
        def broadcast(self, output, src):
            assert src == 0
            assert output.shape == (1, 4, 2)
            output.copy_(embedding)

        def all_gather(self, *_args, **_kwargs):
            raise AssertionError("one complete temporal item uses broadcast")

    parallel = SimpleNamespace(attn_tp_size=2, attn_tp_rank=rank, attn_tp_group=Group())
    encode = Mock(return_value=embedding.clone())
    with patch("sglang.srt.multimodal.mm_utils.get_parallel", return_value=parallel):
        output = run_dp_sharded_mrope_vision_model(
            _StaticTower(list_output=True),
            None,
            [[4, 2, 2]],
            rope_type="rope_2d",
            pool_temporal_dimension=True,
            encode_local_items=encode,
            pixel_values_device=torch.device("cpu"),
            pixel_values_dtype=torch.float32,
        )
    if rank == 0:
        encode.assert_called_once_with([0])
    else:
        encode.assert_not_called()
    torch.testing.assert_close(output, embedding)


def test_video_temporal_position_embeddings_preserve_activation_dtype():
    module = Learnable2DInterpPosEmbDividedFixed(2, 2, 4, 8).to(torch.bfloat16)
    module.weight.data.zero_()
    # Checkpoint temporal constants can still be float32 after loading.
    module.time_weight = module.time_weight.float()
    output = module.position_embeddings(torch.tensor([[4, 2, 2]]))
    assert output.dtype == torch.bfloat16
    assert output.shape == (16, 8)


def test_video_real_cpu_decoder_accepts_synthetic_mp4_data_url():
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            array = np.full((16, 16, 3), index * 25, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    uri = "data:video/mp4;base64," + base64.b64encode(buffer.getvalue()).decode()
    processor = _processor()
    output = _process(processor, _request(video_data=[VideoData(uri)]))
    assert [item.modality for item in output.mm_items] == [Modality.VIDEO] * 2
    assert [item.feature.shape[:2] for item in output.mm_items] == [(4, 3), (4, 3)]
    assert all(item.feature.device.type == "cpu" for item in output.mm_items)
    assert output.input_ids.count(99) == sum(
        int(item.image_grid_thw[0, 1:].prod().item()) // 4 for item in output.mm_items
    )
    assert (
        output.mm_items[0].feature[0].float().mean()
        < output.mm_items[1].feature[-1].float().mean()
    )
    assert not torch.cuda.is_initialized()
