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
"""Per-architecture GGUF -> HF tensor name maps.

``GGUFModelLoader`` normally derives this map from ``gguf.get_tensor_name_map``,
which only covers architectures upstream gguf-py knows, and from a meta-device
``AutoModelForCausalLM.from_config`` to enumerate the HF parameter names. Neither
works for an architecture that lives outside transformers, so those are supplied
here instead.

A builder returns the complete ``{gguf_tensor_name: hf_param_name}`` map. Any
GGUF tensor left out of the map is skipped by ``gguf_quant_weights_iterator``,
which is how dummy tensors are dropped.
"""

from typing import Callable, Dict

from transformers import PretrainedConfig

# Sandwich naming: ffn_norm is the pre-FFN norm.
_MUSE_GLIMMER_LAYER_TENSORS = {
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attn_norm",
    "ffn_norm": "post_attention_layernorm",
    "post_ffw_norm": "post_ffn_norm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_gate": "self_attn.output_gate_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}

_MUSE_GLIMMER_GLOBAL_TENSORS = {
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output": "lm_head",
}

# attn_q_norm/attn_k_norm omitted: Muse Glimmer's QK-norm is non-parametric.


def build_muse_glimmer_name_map(config: PretrainedConfig) -> Dict[str, str]:
    name_map = {
        f"{gguf}.weight": f"{hf}.weight"
        for gguf, hf in _MUSE_GLIMMER_GLOBAL_TENSORS.items()
    }
    for layer in range(config.num_hidden_layers):
        for gguf, hf in _MUSE_GLIMMER_LAYER_TENSORS.items():
            name_map[f"blk.{layer}.{gguf}.weight"] = f"model.layers.{layer}.{hf}.weight"
    return name_map


# Keyed by HF ``config.model_type`` (loader.py looks it up with that), which is
# not the GGUF ``general.architecture`` that GGUF_NATIVE_CONFIG_BUILDERS uses:
# llama.cpp spells the arch "muse-glimmer" while the HF config says "muse_glimmer".
GGUF_HF_NAME_MAP_BUILDERS: Dict[str, Callable[[PretrainedConfig], Dict[str, str]]] = {
    "muse_glimmer": build_muse_glimmer_name_map,
}

# Ported from the accepted Qwen GGUF source 5404f76007558048d16a458743fcc48afce37cc9
# and shared-identity audit d67e821bd5ea95286a0f3d7896b6d46f41032df6.
# Qwen3-Next and ordinary safetensors paths intentionally do not use this adapter.
_QWEN3_5_MODEL_TYPES = ("qwen3_5", "qwen3_5_text")


def build_qwen3_5_name_map(config: PretrainedConfig) -> Dict[str, str]:
    import gguf
    import torch
    from transformers import AutoModelForCausalLM

    text_config = getattr(config, "text_config", None) or config
    num_layers = text_config.num_hidden_layers
    if getattr(text_config, "layer_types", None) is None:
        interval = getattr(text_config, "full_attention_interval", 4)
        if interval <= 0:
            raise ValueError("Qwen GGUF full_attention_interval must be positive")
        text_config.layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(num_layers)
        ]
    arch = next(
        (key for key, value in gguf.MODEL_ARCH_NAMES.items() if value == "qwen35"),
        None,
    )
    if arch is None:
        raise RuntimeError("GGUF package must support the qwen35 architecture")
    name_map = gguf.get_tensor_name_map(arch, num_layers)
    with torch.device("meta"):
        state_dict = AutoModelForCausalLM.from_config(text_config).state_dict()

    result, unresolved = {}, []
    for hf_name in state_dict:
        stem, _, suffix = hf_name.rpartition(".")
        gguf_name = name_map.get_name(stem)
        if gguf_name is None:
            unresolved.append(hf_name)
        else:
            result[f"{gguf_name}.{suffix}"] = hf_name
    # These GDN parameters have no .weight suffix; the generic splitter above
    # cannot map them using a module stem.
    for layer in range(num_layers):
        for gguf_suffix, hf_suffix in {
            "ssm_a": "linear_attn.A_log",
            "ssm_dt.bias": "linear_attn.dt_bias",
        }.items():
            hf_name = f"model.layers.{layer}.{hf_suffix}"
            if hf_name in state_dict:
                result[f"blk.{layer}.{gguf_suffix}"] = hf_name
                if hf_name in unresolved:
                    unresolved.remove(hf_name)
    if unresolved:
        raise RuntimeError(f"Qwen GGUF parameters have no tensor name: {unresolved}")
    return result


def _permute_head_blocks(tensor, index, block, offset=0):
    """Exact row/byte permutation; leave Q/K rows before offset unchanged."""
    import torch

    body = tensor[offset:]
    count = len(index)
    if body.shape[0] != count * block:
        raise ValueError(
            f"cannot split {body.shape[0]} rows into {count} groups of {block}"
        )
    rest = tuple(body.shape[1:])
    body = body.reshape(count, block, *rest)[list(index)].reshape(count * block, *rest)
    return body if offset == 0 else torch.cat([tensor[:offset], body], dim=0)


class Qwen3_5GGUFWeightTransform:
    """Undo llama.cpp's qwen35 head order, decay and folded-norm conventions.

    llama.cpp stores value head p as HF head (p % nk) * (nv / nk) + p // nk,
    and stores ssm_a as -exp(A_log). SGLang expects HF head order and A_log.
    Quantized head reordering is an exact packed-byte move, not requantization.
    """

    def __init__(self, config):
        text = getattr(config, "text_config", None) or config
        self.num_k_heads = int(text.linear_num_key_heads)
        self.num_v_heads = int(text.linear_num_value_heads)
        self.head_k_dim = int(text.linear_key_head_dim)
        self.head_v_dim = int(text.linear_value_head_dim)
        if (
            min(self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim)
            <= 0
        ):
            raise ValueError("Qwen linear attention geometry must be positive")
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("Qwen value heads must be a multiple of key heads")
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        group = self.num_v_heads // self.num_k_heads
        self.gguf_index_of_hf_head = [
            (head % group) * self.num_k_heads + head // group
            for head in range(self.num_v_heads)
        ]

    def _permute_columns(self, name, tensor, weight_type):
        import gguf
        import torch

        index = self.gguf_index_of_hf_head
        if tensor.dtype != torch.uint8:
            if tensor.shape[1] != self.value_dim:
                raise ValueError(f"{name}: expected {self.value_dim} columns")
            return _permute_head_blocks(
                tensor.T.contiguous(), index, self.head_v_dim
            ).T.contiguous()
        quant = gguf.GGMLQuantizationType(int(weight_type))
        block_size, type_size = gguf.GGML_QUANT_SIZES[quant]
        if self.head_v_dim % block_size:
            raise NotImplementedError(
                f"{name}: head dim {self.head_v_dim} does not divide into "
                f"{quant.name} blocks of {block_size}; requantization is unsupported"
            )
        group_bytes = self.head_v_dim // block_size * type_size
        if tensor.shape[1] != self.num_v_heads * group_bytes:
            raise ValueError(f"{name}: invalid packed column geometry for {quant.name}")
        return (
            tensor.reshape(tensor.shape[0], self.num_v_heads, group_bytes)[
                :, list(index), :
            ]
            .reshape(tensor.shape[0], -1)
            .contiguous()
        )

    def __call__(self, name, tensor, weight_type):
        import gguf
        import torch

        index = self.gguf_index_of_hf_head
        if name.endswith(".qweight") and weight_type == gguf.GGMLQuantizationType.BF16:
            if tensor.dtype != torch.uint8 or tensor.shape[-1] % 2:
                raise ValueError(f"{name}: invalid packed GGUF BF16 payload")
            tensor = tensor.contiguous().view(torch.bfloat16)

        if name == "model.norm.weight" or name.endswith(
            (
                ".input_layernorm.weight",
                ".post_attention_layernorm.weight",
                ".q_norm.weight",
                ".k_norm.weight",
            )
        ):
            # The linear_attn.norm gated norm is raw, not zero-centred.
            if tensor.dtype == torch.uint8:
                raise NotImplementedError(f"{name}: folded RMSNorm must be unquantized")
            return tensor - 1.0
        if ".linear_attn." not in name:
            return tensor
        if name.endswith(".linear_attn.A_log"):
            tensor = _permute_head_blocks(tensor, index, 1)
            if not bool((torch.isfinite(tensor) & (tensor < 0)).all()):
                raise ValueError(f"{name}: ssm_a must be finite and negative")
            return torch.log(-tensor.to(torch.float32)).to(tensor.dtype)
        if name.endswith(".linear_attn.dt_bias"):
            return _permute_head_blocks(tensor, index, 1)
        if ".linear_attn.conv1d." in name:
            tensor = _permute_head_blocks(
                tensor, index, self.head_v_dim, offset=2 * self.key_dim
            )
            return tensor.unsqueeze(1) if tensor.dim() == 2 else tensor
        if ".linear_attn.in_proj_qkv." in name:
            return _permute_head_blocks(
                tensor, index, self.head_v_dim, offset=2 * self.key_dim
            )
        if ".linear_attn.in_proj_z." in name:
            return _permute_head_blocks(tensor, index, self.head_v_dim)
        if ".linear_attn.in_proj_b." in name or ".linear_attn.in_proj_a." in name:
            return _permute_head_blocks(tensor, index, 1)
        if ".linear_attn.out_proj." in name:
            return self._permute_columns(name, tensor, weight_type)
        return tensor


GGUF_HF_NAME_MAP_BUILDERS.update(
    {model_type: build_qwen3_5_name_map for model_type in _QWEN3_5_MODEL_TYPES}
)


def get_gguf_weight_transform(config):
    if config.model_type not in _QWEN3_5_MODEL_TYPES:
        return None
    return Qwen3_5GGUFWeightTransform(config)


def get_missing_gguf_parameters(model, loaded_names):
    """Compare parameter identity, including aliases; equal values are not enough."""
    by_name = dict(model.named_parameters(remove_duplicate=False))
    loaded_ids = {id(by_name[name]) for name in loaded_names if name in by_name}
    return sorted(
        name
        for name, parameter in model.named_parameters()
        if id(parameter) not in loaded_ids
    )


def apply_gguf_weight_transform(weights_iterator, transform):
    # weight_utils yields metadata before packed tensors.
    weight_types = {}
    for name, tensor in weights_iterator:
        if name.endswith(".qweight_type"):
            if tensor.numel() != 1:
                raise ValueError(f"{name}: GGUF weight type must be scalar")
            weight_types[name.removesuffix(".qweight_type")] = int(
                tensor.reshape(-1)[0]
            )
            yield name, tensor
        else:
            stem, _, _ = name.rpartition(".")
            if name.endswith(".qweight") and stem not in weight_types:
                raise ValueError(f"{name}: missing GGUF weight type metadata")
            yield name, transform(name, tensor, weight_types.get(stem))
