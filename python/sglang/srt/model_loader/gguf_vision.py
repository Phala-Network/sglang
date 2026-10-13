# SPDX-License-Identifier: Apache-2.0
"""Load the separately supplied unquantized qwen35 GGUF vision projector.

Ported from 2b2584575aa53702ce331c60a70f96b038610665. llama.cpp's Qwen3-VL
converter splits Conv3d along temporal axis 2; stack the two exact slices.
Unsupported geometry, metadata, tensor names, shapes and types fail closed.
"""

import logging

logger = logging.getLogger(__name__)


def qwen35_vision_tensor_map(config):
    vision = config.vision_config
    if vision.temporal_patch_size != 2 or vision.deepstack_visual_indexes:
        raise ValueError("GGUF vision requires temporal_patch_size=2 and no deepstack")
    hidden = vision.hidden_size
    intermediate = vision.intermediate_size
    merged = hidden * vision.spatial_merge_size**2
    patch_shape = (hidden, vision.in_channels, vision.patch_size, vision.patch_size)
    mapping = {
        "v.patch_embd.weight": ("visual.patch_embed.proj.weight", patch_shape),
        "v.patch_embd.weight.1": ("visual.patch_embed.proj.weight", patch_shape),
        "v.patch_embd.bias": ("visual.patch_embed.proj.bias", (hidden,)),
        "v.position_embd.weight": (
            "visual.pos_embed.weight",
            (vision.num_position_embeddings, hidden),
        ),
        "v.post_ln.weight": ("visual.merger.norm.weight", (hidden,)),
        "v.post_ln.bias": ("visual.merger.norm.bias", (hidden,)),
        "mm.0.weight": ("visual.merger.linear_fc1.weight", (merged, merged)),
        "mm.0.bias": ("visual.merger.linear_fc1.bias", (merged,)),
        "mm.2.weight": (
            "visual.merger.linear_fc2.weight",
            (vision.out_hidden_size, merged),
        ),
        "mm.2.bias": ("visual.merger.linear_fc2.bias", (vision.out_hidden_size,)),
    }
    layers = {
        "attn_qkv": ("attn.qkv_proj", (3 * hidden, hidden)),
        "attn_out": ("attn.proj", (hidden, hidden)),
        "ffn_up": ("mlp.linear_fc1", (intermediate, hidden)),
        "ffn_down": ("mlp.linear_fc2", (hidden, intermediate)),
        "ln1": ("norm1", (hidden,)),
        "ln2": ("norm2", (hidden,)),
    }
    for layer in range(vision.depth):
        for gguf_name, (hf_name, shape) in layers.items():
            for suffix, tensor_shape in (("weight", shape), ("bias", (shape[0],))):
                mapping[f"v.blk.{layer}.{gguf_name}.{suffix}"] = (
                    f"visual.blocks.{layer}.{hf_name}.{suffix}",
                    tensor_shape,
                )
    return mapping


def _read_unquantized_tensor(tensor, expected_shape):
    import gguf
    import torch

    if tuple(reversed(tuple(map(int, tensor.shape)))) != expected_shape:
        raise ValueError(f"{tensor.name}: GGUF shape does not match {expected_shape}")
    value = torch.from_numpy(tensor.data.copy())
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        if value.dtype != torch.uint8 or value.shape[-1] % 2:
            raise ValueError(f"{tensor.name}: invalid BF16 byte payload")
        value = value.view(torch.bfloat16)
    elif tensor.tensor_type not in (
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
    ):
        raise ValueError(
            f"{tensor.name}: only F32/F16/BF16 vision tensors are supported"
        )
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"{tensor.name}: decoded payload shape does not match {expected_shape}"
        )
    return value


def qwen35_vision_weights_iterator(path, config):
    import gguf
    import torch

    reader = gguf.GGUFReader(path)
    vision = config.vision_config
    expected_metadata = {
        "general.architecture": "clip",
        "clip.projector_type": "qwen3vl_merger",
        "clip.vision.block_count": vision.depth,
        "clip.vision.embedding_length": vision.hidden_size,
        "clip.vision.attention.head_count": vision.num_heads,
        "clip.vision.feed_forward_length": vision.intermediate_size,
        "clip.vision.patch_size": vision.patch_size,
        "clip.vision.spatial_merge_size": vision.spatial_merge_size,
        "clip.vision.projection_dim": vision.out_hidden_size,
    }
    for key, expected in expected_metadata.items():
        field = reader.get_field(key)
        if field is None or field.contents() != expected:
            raise ValueError(
                f"Qwen GGUF vision metadata mismatch: {key}, expected {expected}"
            )
    mapping = qwen35_vision_tensor_map(config)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    missing = sorted(mapping.keys() - tensors.keys())
    extra = sorted(tensors.keys() - mapping.keys())
    if missing or extra or len(tensors) != len(reader.tensors):
        raise ValueError(
            f"Incomplete or unexpected Qwen GGUF vision tensors: missing={missing}, extra={extra}"
        )
    # Check all source geometry before yielding any weights.
    for name, (_, shape) in mapping.items():
        if tuple(reversed(tuple(map(int, tensors[name].shape)))) != shape:
            raise ValueError(f"{name}: GGUF shape does not match {shape}")
    patch_names = ("v.patch_embd.weight", "v.patch_embd.weight.1")
    patches = [
        _read_unquantized_tensor(tensors[name], mapping[name][1])
        for name in patch_names
    ]
    yield "visual.patch_embed.proj.weight", torch.stack(patches, dim=2)
    for name, (target, shape) in mapping.items():
        if name not in patch_names:
            yield target, _read_unquantized_tensor(tensors[name], shape)
    logger.info(
        "Qwen GGUF vision audit: source_tensors=%d loaded_parameters=%d missing=[]",
        len(tensors),
        len(mapping) - 1,
    )
