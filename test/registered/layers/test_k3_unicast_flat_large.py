"""The flat large all-reduce: prefill-sized messages above the push window only."""
import types
import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.layers.communication import k3_ar_fusion as f

WINDOW = 512 * 1024
SHARED_BYTES_PER_TOKEN = 7168 * 2


class FlatLargeTest(unittest.TestCase):
    def setUp(self):
        for field, value in (
            (envs.SGLANG_K3_UNICAST_FLAT_LARGE, True),
            (envs.SGLANG_K3_UNICAST_FLAT_LARGE_MIN_TOKENS, 512),
        ):
            field.set(value)
            self.addCleanup(field.clear)
        self.state = types.SimpleNamespace(multicast=False)
        for name, value in (("_get_state", self.state), ("_push_max_bytes", WINDOW)):
            p = mock.patch.object(f, name, return_value=value)
            p.start()
            self.addCleanup(p.stop)

    def flat(self, tokens):
        return f.unicast_flat_large(tokens * SHARED_BYTES_PER_TOKEN, tokens)

    def test_decode_batches_stay_on_push(self):
        for tokens in (1, 36, 37, 72, 270, 511):
            self.assertFalse(self.flat(tokens), tokens)

    def test_prefill_batches_go_flat(self):
        for tokens in (512, 1024, 4096, 8192):
            self.assertTrue(self.flat(tokens), tokens)

    def test_small_message_stays_on_push_past_the_token_floor(self):
        self.assertFalse(f.unicast_flat_large(WINDOW, 8192))

    def test_switch_off(self):
        envs.SGLANG_K3_UNICAST_FLAT_LARGE.set(False)
        self.assertFalse(self.flat(8192))

    def test_multicast_build_never(self):
        self.state.multicast = True
        self.assertFalse(self.flat(8192))

    def test_floor_is_configurable(self):
        envs.SGLANG_K3_UNICAST_FLAT_LARGE_MIN_TOKENS.set(37)
        self.assertTrue(self.flat(37))
        self.assertFalse(self.flat(36))


if __name__ == "__main__":
    unittest.main()
