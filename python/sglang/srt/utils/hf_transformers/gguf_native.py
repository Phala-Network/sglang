# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reading config and tokenizer from a GGUF whose architecture transformers lacks.

``load_gguf_checkpoint`` refuses any architecture outside its own
``GGUF_SUPPORTED_ARCHITECTURES``, and it does so before touching a single field,
so both the config and the tokenizer are unreachable for such a checkpoint --
even though the tokenizer half of that reader is entirely architecture-agnostic
(it dispatches on ``tokenizer.ggml.model``, not on the model architecture).

This module carries SGLang's own path for those checkpoints:

* ``GGUF_NATIVE_CONFIG_BUILDERS`` maps a GGUF ``general.architecture`` to a
  builder returning a fully populated config.
* ``build_gguf_tokenizer`` reuses transformers' own converters, which work fine
  once they are reached directly instead of through the gated loader.

Reaching for these is a last resort: a config.json next to the .gguf still wins,
because the checkpoint author's own config outranks anything reconstructed.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from transformers import PretrainedConfig

from sglang.srt.configs.muse_glimmer import MuseGlimmerConfig

GGUF_NATIVE_CONFIG_BUILDERS: Dict[str, Callable[[str], PretrainedConfig]] = {
    "muse-glimmer": MuseGlimmerConfig.from_gguf,
    "qwen35": lambda path: build_qwen3_5_gguf_config(path),
}


def read_gguf_architecture(gguf_path: str) -> Optional[str]:
    """The ``general.architecture`` string, or None if it cannot be read."""
    try:
        from gguf import GGUFReader

        reader = GGUFReader(gguf_path)
        field = reader.fields.get("general.architecture")
        if field is None:
            return None
        value = field.contents()
        return value if isinstance(value, str) else None
    except Exception:
        return None


def has_native_gguf_support(gguf_path: str) -> bool:
    return read_gguf_architecture(gguf_path) in GGUF_NATIVE_CONFIG_BUILDERS


def build_gguf_config(gguf_path: str) -> PretrainedConfig:
    arch = read_gguf_architecture(gguf_path)
    return GGUF_NATIVE_CONFIG_BUILDERS[arch](gguf_path)


def build_qwen3_5_gguf_config(gguf_path: str) -> PretrainedConfig:
    """Reconstruct the Qwen3.5 text/vision config from llama.cpp GGUF metadata."""
    from gguf import GGUFReader

    from sglang.srt.configs.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

    reader = GGUFReader(gguf_path)
    meta = {key: field.contents() for key, field in reader.fields.items()}
    if meta.get("general.architecture") != "qwen35":
        raise ValueError("Qwen3.5 GGUF requires general.architecture=qwen35")
    tensors = {tensor.name: tensor for tensor in reader.tensors}

    def required(key):
        if key not in meta:
            raise ValueError(f"Qwen3.5 GGUF is missing {key}")
        return meta[key]

    def integer(key):
        value = int(required(f"qwen35.{key}"))
        if value <= 0:
            raise ValueError(f"Qwen3.5 GGUF requires positive qwen35.{key}")
        return value

    if "token_embd.weight" not in tensors:
        raise ValueError("Qwen3.5 GGUF is missing token_embd.weight")
    vocab_size = int(tensors["token_embd.weight"].shape[1])
    total_layers = integer("block_count")
    mtp_layers = int(meta.get("qwen35.nextn_predict_layers", 0))
    if mtp_layers < 0 or mtp_layers >= total_layers:
        raise ValueError("Qwen3.5 GGUF has invalid nextn_predict_layers")
    layers = total_layers - mtp_layers
    interval = integer("full_attention_interval")
    key_dim = integer("ssm.state_size")
    inner_size = integer("ssm.inner_size")
    if inner_size % key_dim:
        raise ValueError("Qwen3.5 GGUF ssm.inner_size must divide by ssm.state_size")
    rope_dim = integer("rope.dimension_count")
    head_dim = integer("attention.key_length")
    if rope_dim > head_dim:
        raise ValueError(
            "Qwen3.5 GGUF rope.dimension_count exceeds attention.key_length"
        )
    rope_parameters = {
        "rope_theta": float(required("qwen35.rope.freq_base")),
        "partial_rotary_factor": rope_dim / head_dim,
        "rope_type": "default",
    }
    sections = meta.get("qwen35.rope.dimension_sections")
    if sections is not None:
        sections = [int(section) for section in sections]
        while sections and sections[-1] == 0:
            sections.pop()
        if len(sections) != 3 or sum(sections) * 2 != rope_dim:
            raise ValueError("Qwen3.5 GGUF has invalid rope.dimension_sections")
        rope_parameters.update(
            mrope_section=sections,
            mrope_interleaved=True,
        )

    text_kwargs = dict(
        vocab_size=vocab_size,
        hidden_size=integer("embedding_length"),
        intermediate_size=integer("feed_forward_length"),
        num_hidden_layers=layers,
        num_attention_heads=integer("attention.head_count"),
        num_key_value_heads=integer("attention.head_count_kv"),
        head_dim=head_dim,
        max_position_embeddings=integer("context_length"),
        rms_norm_eps=float(required("qwen35.attention.layer_norm_rms_epsilon")),
        rope_theta=rope_parameters["rope_theta"],
        rope_parameters=rope_parameters,
        partial_rotary_factor=rope_dim / head_dim,
        linear_conv_kernel_dim=integer("ssm.conv_kernel"),
        linear_key_head_dim=key_dim,
        linear_value_head_dim=key_dim,
        linear_num_key_heads=integer("ssm.group_count"),
        linear_num_value_heads=inner_size // key_dim,
        full_attention_interval=interval,
        layer_types=[
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(layers)
        ],
        bos_token_id=meta.get("tokenizer.ggml.bos_token_id"),
        eos_token_id=meta.get("tokenizer.ggml.eos_token_id"),
        dtype="bfloat16",
    )
    projector_files = sorted(Path(gguf_path).parent.glob("mmproj-*.gguf"))
    if not projector_files:
        return Qwen3_5TextConfig(**text_kwargs, architectures=["Qwen3_5ForCausalLM"])
    if len(projector_files) != 1:
        raise ValueError("Qwen3.5 GGUF requires exactly one mmproj-*.gguf")

    projector = GGUFReader(str(projector_files[0]))
    vision_meta = {key: field.contents() for key, field in projector.fields.items()}
    vision_tensors = {tensor.name: tensor for tensor in projector.tensors}
    if (
        vision_meta.get("general.architecture"),
        vision_meta.get("clip.projector_type"),
    ) != (
        "clip",
        "qwen3vl_merger",
    ):
        raise ValueError("Qwen3.5 GGUF has an unsupported vision projector")
    try:
        patch_shape = vision_tensors["v.patch_embd.weight"].shape
        position_shape = vision_tensors["v.position_embd.weight"].shape
        vision_kwargs = dict(
            depth=int(vision_meta["clip.vision.block_count"]),
            hidden_size=int(vision_meta["clip.vision.embedding_length"]),
            intermediate_size=int(vision_meta["clip.vision.feed_forward_length"]),
            num_heads=int(vision_meta["clip.vision.attention.head_count"]),
            patch_size=int(vision_meta["clip.vision.patch_size"]),
            spatial_merge_size=int(vision_meta["clip.vision.spatial_merge_size"]),
            out_hidden_size=int(vision_meta["clip.vision.projection_dim"]),
            in_channels=int(patch_shape[2]),
            num_position_embeddings=int(position_shape[1]),
            temporal_patch_size=2,
            deepstack_visual_indexes=[],
        )
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Qwen3.5 GGUF vision metadata is incomplete: {exc}") from exc
    return Qwen3_5Config(
        text_config=text_kwargs,
        vision_config=vision_kwargs,
        architectures=["Qwen3_5ForConditionalGeneration"],
        rope_scaling=rope_parameters,
    )


_GPT4O_SPLIT_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

_PRE_TOKENIZER_REGEX = {
    # LLAMA_VOCAB_PRE_TYPE_LLAMA3
    "llama-bpe": (
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
        r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
        r"\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    "gpt-4o": _GPT4O_SPLIT_REGEX,
    "llama4": _GPT4O_SPLIT_REGEX,
}

_GGML_TOKEN_TYPE_CONTROL = 3


def build_gguf_generation_config(gguf_path: str):
    """GenerationConfig from GGUF metadata, or None if there is nothing to say.

    llama.cpp records the end-of-generation ids explicitly, and for a
    Harmony-style model the distinction matters: ``eos_token_id`` ends the
    sequence and ``eot_token_id`` ends a turn, so both must stop generation while
    an end-of-*message* id must not -- stopping on that truncates the model
    mid-reasoning, before it answers.
    """
    from gguf import GGUFReader
    from transformers import GenerationConfig

    reader = GGUFReader(gguf_path)
    meta = {key: field.contents() for key, field in reader.fields.items()}

    stop_ids = []
    for key in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"):
        if key in meta:
            value = int(meta[key])
            if value not in stop_ids:
                stop_ids.append(value)
    if not stop_ids:
        return None

    fields: Dict[str, Any] = {
        "eos_token_id": stop_ids if len(stop_ids) > 1 else stop_ids[0]
    }
    if "tokenizer.ggml.bos_token_id" in meta:
        fields["bos_token_id"] = int(meta["tokenizer.ggml.bos_token_id"])
    if "tokenizer.ggml.padding_token_id" in meta:
        fields["pad_token_id"] = int(meta["tokenizer.ggml.padding_token_id"])
    return GenerationConfig(**fields)


def build_gguf_tokenizer(gguf_path: str, **kwargs: Any):
    """Build a fast tokenizer from GGUF metadata alone.

    transformers' own GGUF tokenizer path is unreachable for an architecture its
    checkpoint loader rejects, and its converters key on ``tokenizer.ggml.model``
    (here "gpt2") which loses both the special-token block and the pre-tokenizer
    regex. So the tokenizers spec is assembled directly instead: a byte-level BPE
    over the NORMAL tokens, the CONTROL tokens registered as added specials, and
    the split regex named by ``tokenizer.ggml.pre``.
    """
    import json

    from gguf import GGUFReader
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    reader = GGUFReader(gguf_path)
    meta = {key: field.contents() for key, field in reader.fields.items()}

    tokens = list(meta["tokenizer.ggml.tokens"])
    token_types = [int(t) for t in meta["tokenizer.ggml.token_type"]]
    merges = [tuple(m.split(" ", 1)) for m in meta["tokenizer.ggml.merges"]]

    pre_name = meta.get("tokenizer.ggml.pre")
    if pre_name not in _PRE_TOKENIZER_REGEX:
        raise ValueError(
            f"No pre-tokenizer regex known for tokenizer.ggml.pre={pre_name!r}; "
            f"known: {sorted(_PRE_TOKENIZER_REGEX)}"
        )

    control_ids = [
        i for i, t in enumerate(token_types) if t == _GGML_TOKEN_TYPE_CONTROL
    ]
    control = set(control_ids)
    vocab = {tok: i for i, tok in enumerate(tokens) if i not in control}

    def token_of(key):
        idx = meta.get(f"tokenizer.ggml.{key}")
        return None if idx is None else tokens[int(idx)]

    bos = token_of("bos_token_id")

    spec = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": i,
                "content": tokens[i],
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for i in control_ids
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": _PRE_TOKENIZER_REGEX[pre_name]},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": vocab,
            "merges": [list(m) for m in merges],
        },
    }

    if meta.get("tokenizer.ggml.add_bos_token") and bos is not None:
        bos_id = int(meta["tokenizer.ggml.bos_token_id"])
        spec["post_processor"] = {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": bos, "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": bos, "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 0}},
            ],
            "special_tokens": {
                bos: {"id": bos, "ids": [bos_id], "tokens": [bos]},
            },
        }

    backend = Tokenizer.from_str(json.dumps(spec))

    named = {
        bos,
        token_of("eos_token_id"),
        token_of("padding_token_id"),
        token_of("unknown_token_id"),
    }
    additional = [tokens[i] for i in control_ids if tokens[i] not in named]

    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=bos,
        eos_token=token_of("eos_token_id"),
        unk_token=token_of("unknown_token_id"),
        pad_token=token_of("padding_token_id"),
        additional_special_tokens=additional,
        chat_template=meta.get("tokenizer.chat_template"),
        **kwargs,
    )
