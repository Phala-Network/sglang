import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.mode_sampling_defaults import apply_mode_sampling_defaults
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest


class ModeSamplingDefaults(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(sampling_defaults="model", hf_config=SimpleNamespace(
            sglang_sampling_defaults_by_mode={"thinking": {"temperature": 1.0, "presence_penalty": 0.0}, "non_thinking": {"temperature": 0.7, "presence_penalty": 1.5, "top_p": 0.8}}
        ))

    def params(self, **overrides):
        request = ChatCompletionRequest(model="qualified-model", messages=[{"role": "user", "content": "Unchanged request."}], max_tokens=2000, **overrides)
        return request, request.to_sampling_params(stop=[], model_generation_config={})

    def test_omitted_mode_defaults_preserve_explicit_temperature_and_budget(self):
        request, original = self.params(temperature=0)
        result = apply_mode_sampling_defaults(request, original, self.config, False)
        self.assertEqual(result["temperature"], 0)
        self.assertEqual(result["presence_penalty"], 1.5)
        self.assertEqual(result["top_p"], 0.8)
        self.assertEqual(result["max_new_tokens"], 2000)
        self.assertEqual(original["presence_penalty"], 0)

    def test_explicit_zero_penalty_is_never_overridden(self):
        request, original = self.params(presence_penalty=0, top_p=1.0)
        result = apply_mode_sampling_defaults(request, original, self.config, False)
        self.assertEqual(result["presence_penalty"], 0)
        self.assertEqual(result["top_p"], 1.0)

    def test_thinking_and_openai_opt_out(self):
        request, original = self.params()
        result = apply_mode_sampling_defaults(request, original, self.config, True)
        self.assertEqual(result["presence_penalty"], 0)
        self.assertEqual(result["temperature"], 1.0)
        self.config.sampling_defaults = "openai"
        self.assertEqual(apply_mode_sampling_defaults(request, original, self.config, False), original)

    def test_models_without_profiles_and_unknown_mode_are_unchanged(self):
        request, original = self.params()
        other = SimpleNamespace(sampling_defaults="model", hf_config=SimpleNamespace())
        self.assertEqual(apply_mode_sampling_defaults(request, original, other, False), original)
        self.assertEqual(apply_mode_sampling_defaults(request, original, self.config, None), original)


if __name__ == "__main__":
    unittest.main()
