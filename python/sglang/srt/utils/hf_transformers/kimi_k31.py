"""Select the image-baked K3.1 metadata implementation before HF auto_map.

The upstream checkpoint carries Python tokenizer/processor modules alongside
its weights. Their fixed, reviewed implementation lives in SGLang's package;
checkpoint directories supply JSON and BPE data, never executable modules.
"""

from transformers import PretrainedConfig


def is_kimi_k31_checkpoint(model, revision=None, **kwargs):
    try:
        # PretrainedConfig only reads metadata, including for an uncached Hub
        # revision; unlike AutoConfig it never dispatches through auto_map.
        config, _ = PretrainedConfig.get_config_dict(model, revision=revision, **kwargs)
        return config.get("model_type") == "kimi_k31"
    except (OSError, ValueError):
        return False


def load_kimi_k31_config(model, revision=None, **kwargs):
    if not is_kimi_k31_checkpoint(model, revision, **kwargs):
        return None
    from sglang.srt.configs.kimi_k3 import KimiK31Config

    return KimiK31Config.from_pretrained(model, revision=revision, **kwargs)


def load_kimi_k31_tokenizer(model, *args, revision=None, **kwargs):
    from sglang.srt.multimodal.kimi_k31_hf.tokenization_kimi import TikTokenTokenizer

    # A concrete class loads JSON/BPE data without consulting AutoTokenizer's
    # auto_map or executing a checkpoint's Python, even if its caller trusts it.
    kwargs.pop("trust_remote_code", None)
    kwargs.pop("tokenizer_revision", None)
    return TikTokenTokenizer.from_pretrained(model, *args, revision=revision, **kwargs)


def load_kimi_k31_processor(model, *args, revision=None, **kwargs):
    from sglang.srt.multimodal.kimi_k31_hf.kimi_k3_processor import KimiK3Processor
    from sglang.srt.multimodal.kimi_k31_hf.kimi_k3_vision_processing import (
        KimiK3VisionProcessor,
    )

    # Construct concrete components: ProcessorMixin.from_pretrained would call
    # the checkpoint's AutoTokenizer/AutoImageProcessor maps again.
    tokenizer = load_kimi_k31_tokenizer(model, *args, revision=revision, **kwargs)
    image_kwargs = dict(kwargs)
    image_kwargs.pop("trust_remote_code", None)
    image_kwargs.pop("tokenizer_revision", None)
    image_kwargs.pop("use_fast", None)
    image_processor = KimiK3VisionProcessor.from_pretrained(
        model, revision=revision, **image_kwargs
    )
    return KimiK3Processor(image_processor=image_processor, tokenizer=tokenizer)
