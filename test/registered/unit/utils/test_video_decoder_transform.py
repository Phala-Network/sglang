"""``VideoDecoderWrapper.get_frames_as_tensor(transform=...)`` decodes in steps and
returns what transforming every frame at once would."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import importlib.util
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.utils import video_decoder
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.test_utils import CustomTestCase

WIDTH, HEIGHT, FRAMES = 64, 48, 40


def _write_video(path):
    import av

    with av.open(path, "w") as out:
        stream = out.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for i in range(FRAMES):
            img = np.full((HEIGHT, WIDTH, 3), i * 6, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, "rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def _halve(frames):
    return frames.permute(0, 3, 1, 2)[:, :, ::2, ::2].contiguous()


@unittest.skipUnless(
    video_decoder._BACKEND == "torchcodec" and importlib.util.find_spec("av"),
    "needs torchcodec and PyAV",
)
@patch.object(video_decoder, "_is_cpu_engine", return_value=True)
class TestTransformInSteps(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "video.mp4")
        _write_video(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_matches_transforming_all_frames_at_once(self, _):
        indices = list(range(0, FRAMES, 3))
        for threads in (1, 4):
            vr = VideoDecoderWrapper(self.path, num_decode_threads=threads)
            expected = _halve(vr.get_frames_as_tensor(indices))
            steps = []

            def transform(frames):
                steps.append(len(frames))
                return _halve(frames)

            with patch.object(
                video_decoder, "_TRANSFORM_STEP_BYTES", 4 * WIDTH * HEIGHT * 3
            ):
                got = vr.get_frames_as_tensor(indices, transform=transform)
            self.assertTrue(torch.equal(got, expected))
            self.assertEqual(sum(steps), len(indices))
            self.assertEqual(max(steps), 4)


if __name__ == "__main__":
    unittest.main()
