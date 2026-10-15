"""Startup-only Xing EAGLE object metadata; never read tensor values."""

import json
import logging

logger = logging.getLogger(__name__)


def _class_name(value):
    return type(value).__name__ if value is not None else None


def _tensor_metadata(value):
    if value is None:
        return None
    return {"dtype": str(value.dtype), "shape": list(value.shape)}


def _pool_metadata(pool):
    buffers = {}
    # Read owner tensors directly. Pool getters can synchronize or dequantize.
    for name in ("kv_buffer", "k_buffer", "v_buffer"):
        values = getattr(pool, name, None)
        if isinstance(values, (list, tuple)):
            buffers[name] = {
                "count": len(values),
                "tensors": [_tensor_metadata(value) for value in values],
            }
    return {
        "class": _class_name(pool),
        "logical_dtype": str(getattr(pool, "dtype", None)),
        "storage_dtype": str(getattr(pool, "store_dtype", None)),
        "size": getattr(pool, "size", None),
        "page_size": getattr(pool, "page_size", None),
        "layer_num": getattr(pool, "layer_num", None),
        "start_layer": getattr(pool, "start_layer", None),
        "end_layer": getattr(pool, "end_layer", None),
        "kv_cache_dim": getattr(pool, "kv_cache_dim", None),
        "post_capture_active": getattr(pool, "post_capture_active", False),
        "buffers": buffers,
    }


def _layer_metadata(layer):
    return {
        "class": _class_name(layer),
        "quant_method_class": _class_name(getattr(layer, "quant_method", None)),
        "weight": _tensor_metadata(getattr(layer, "weight", None)),
    }


def _runner_metadata(runner):
    model = runner.model
    inner = getattr(model, "model", None)
    return {
        "model_class": _class_name(model),
        "model_quantization": runner.model_config.quantization,
        "quant_config_class": _class_name(getattr(model, "quant_config", None)),
        "inner_quant_config_class": _class_name(getattr(inner, "quant_config", None)),
        "runner_kv_cache_dtype": str(runner.kv_cache_dtype),
        "max_total_num_tokens": runner.max_total_num_tokens,
        "pool": _pool_metadata(runner.token_to_kv_pool),
        "head": _layer_metadata(getattr(model, "lm_head", None)),
        "embedding": _layer_metadata(getattr(inner, "embed_tokens", None)),
    }


def emit_xing_spec_startup_identity(scheduler):
    """Observe final pools and head sharing after graphs/resize/weight loading.

    This diagnostic has no control role. A metadata failure cannot change
    numerical execution or make an otherwise usable model fail startup.
    """
    target = scheduler.tp_worker.model_runner
    architectures = getattr(target.model_config.hf_config, "architectures", ())
    algorithm = getattr(scheduler.spec_algorithm, "name", None)
    draft_worker = scheduler.draft_worker
    if (
        architectures != ["Xing4_0ForCausalLM"]
        or algorithm not in {"EAGLE", "EAGLE3"}
        or draft_worker is None
    ):
        return None
    try:
        draft = draft_worker.draft_worker.draft_runner
        target_inner = getattr(target.model, "model", None)
        draft_inner = getattr(draft.model, "model", None)
        target_head = getattr(getattr(target.model, "lm_head", None), "weight", None)
        draft_head = getattr(getattr(draft.model, "lm_head", None), "weight", None)
        target_embed = getattr(
            getattr(target_inner, "embed_tokens", None), "weight", None
        )
        draft_embed = getattr(
            getattr(draft_inner, "embed_tokens", None), "weight", None
        )
        record = {
            "schema": "phala.xing4.spec-startup-identity.v1",
            "stage": "after_graphs_resize_and_weight_finalize",
            "algorithm": algorithm,
            "target": _runner_metadata(target),
            "draft": _runner_metadata(draft),
            "sharing": {
                "kv_pool_same_object": target.token_to_kv_pool
                is draft.token_to_kv_pool,
                "req_to_token_pool_same_object": target.req_to_token_pool
                is draft.req_to_token_pool,
                "token_allocator_same_object": target.token_to_kv_pool_allocator
                is draft.token_to_kv_pool_allocator,
                "head_weight_same_object": target_head is not None
                and target_head is draft_head,
                "embedding_weight_same_object": target_embed is not None
                and target_embed is draft_embed,
            },
        }
        logger.info("XING_SPEC_STARTUP_IDENTITY %s", json.dumps(record, sort_keys=True))
        return record
    except Exception as exc:
        # No exception message/repr: it can contain paths or tensor contents.
        logger.warning(
            "XING_SPEC_STARTUP_IDENTITY unavailable error_type=%s", type(exc).__name__
        )
        return None
