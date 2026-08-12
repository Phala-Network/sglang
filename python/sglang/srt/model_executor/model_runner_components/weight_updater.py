from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.platforms import current_platform
from sglang.srt.utils import (
    MultiprocessingSerializer,
    dynamic_import,
    get_available_gpu_memory,
    init_custom_process_group,
)
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _unsupported_derived_weight_cache_error() -> Optional[str]:
    """Reject online weight updates that derived-weight caches cannot survive.

    The HPC-Ops bf16xfp32 GEMM caches the fp32 weight split; in-place loader
    writes are invisible to it, so an update would silently keep serving the
    old weights. The check is startup-determined and rank-uniform, so an
    update never proceeds on some workers while rejected on others.
    """
    from sglang.kernels.ops.attention.dsv4.gemm import hpc_bf16xfp32_gemm_enabled

    if hpc_bf16xfp32_gemm_enabled():
        return (
            "Online weight updates are not supported while the HPC-Ops "
            "bf16xfp32 GEMM optimization is enabled: the cached weight "
            "split would keep serving the old weights."
        )
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class WeightUpdater:
    tp_rank: int
    device: str
    gpu_id: int
    model_config: ModelConfig
    custom_weight_loaders: dict
    get_model: Callable[[], Any]
    update_model_fields: Callable[..., None]
    recapture_cuda_graph: Callable[[], None]
    get_model_runner: Callable[[], ModelRunner]
    _model_update_group: dict = field(default_factory=dict)

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            na = NetworkAddress(master_address, master_port)
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def destroy_weights_update_group(self, group_name):
        try:
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            logger.error(message)
            return False, message

    def _assert_weight_cache_inactive(self: WeightUpdater, op: str) -> None:
        """Reject weight mutations while the CUDA IPC weight cache is active:
        param.data is the daemon's master copy shared with every co-attached
        engine, so an in-place update would silently corrupt them all.
        """
        mode = self.get_model_runner().server_args.weight_cache_mode
        if mode != "off":
            raise RuntimeError(
                f"[weight_cache] {op} is not supported while the weight cache is "
                f"active (--weight-cache-mode {mode}): model weights are shared "
                f"with the daemon via CUDA IPC, so mutating them in place would "
                f"corrupt the daemon's master copy and every co-attached engine. "
                f"Restart with --weight-cache-mode off to use this operation."
            )

    def update_weights_from_disk(
        self: WeightUpdater,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        self._assert_weight_cache_inactive("update_weights_from_disk")
        error = _unsupported_derived_weight_cache_error()
        if error is not None:
            return False, error

        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id, empty_cache=False):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.get_model())
            )
            if weight_name_filter is not None:
                iter = (
                    (name, weight) for name, weight in iter if weight_name_filter(name)
                )

            return iter

        def model_load_weights(model, iter):
            loader.load_weights_and_postprocess(model, iter, target_device)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.get_model(), iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                model_load_weights(self.get_model(), iter)
                return False, message

        self.update_model_fields(
            model,
            model_path=model_path,
            load_format=load_format,
            load_config=load_config,
        )

        if recapture_cuda_graph and (
            self.device == "cuda"
            or self.device == "musa"
            or (
                current_platform.is_out_of_tree()
                and current_platform.support_cuda_graph()
            )
        ):
            self.recapture_cuda_graph()

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def update_weights_from_distributed(
        self: WeightUpdater,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
    ):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """
        self._assert_weight_cache_inactive("update_weights_from_distributed")
        error = _unsupported_derived_weight_cache_error()
        if error is not None:
            return False, error

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        if load_format == "flattened_bucket":
            return self._update_bucketed_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.get_model().load_weights(weights)
            return True, "Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def _update_bucketed_weights_from_distributed(
        self: WeightUpdater, names, dtypes, shapes, group_name
    ):
        try:
            named_tensors = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                named_tensors.append(
                    (
                        name,
                        torch.empty(shape, dtype=target_dtype, device=self.device),
                    )
                )
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor()
            torch.distributed.broadcast(
                flattened_tensor,
                src=0,
                group=self._model_update_group[group_name],
            )
            reconstructed_tensors = bucket.reconstruct_tensors()
            self.get_model().load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self: WeightUpdater,
        named_tensors: List[Tuple[str, Union[torch.Tensor, LocalSerializedTensor]]],
        load_format: Optional[str] = None,
    ):
        error = _unsupported_derived_weight_cache_error()
        if error is not None:
            return False, error

        monkey_patch_torch_reductions()
        self._assert_weight_cache_inactive("update_weights_from_tensor")
        validation_error = _validate_update_weights_from_tensor_payload(
            named_tensors, load_format, tp_rank=self.tp_rank
        )
        if validation_error is not None:
            logger.error(validation_error)
            return False, validation_error

        if load_format not in (None, "direct", "flattened_bucket"):
            if not isinstance(load_format, str):
                message = (
                    "Invalid update_weights_from_tensor payload: "
                    "load_format must be a string or null"
                )
                logger.error(message)
                return False, message
            if load_format not in self.custom_weight_loaders:
                message = (
                    "Invalid update_weights_from_tensor payload: "
                    f"unknown load_format={load_format}"
                )
                logger.error(message)
                return False, message

        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        device_module = torch.get_device_module(self.device)
        infered_device = device_module.current_device()

        try:
            named_tensors = [
                (
                    name,
                    _unwrap_tensor(
                        tensor, tp_rank=self.tp_rank, device=infered_device
                    ),
                )
                for name, tensor in named_tensors
            ]
        except Exception as exc:
            message = (
                "Invalid update_weights_from_tensor payload: failed to unwrap "
                f"tensor: {type(exc).__name__}: {exc}"
            )
            logger.error(message)
            return False, message
        if load_format == "direct":
            _model_load_weights_direct(self.get_model(), named_tensors)
        elif load_format in self.custom_weight_loaders:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.get_model(), named_tensors)
        elif load_format is None:
            self.get_model().load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def _update_weights_from_flattened_bucket(
        self: WeightUpdater,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates"""
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        self.get_model().load_weights(reconstructed_tensors)

        return True, "Success"

    def update_weights_from_ipc(self: WeightUpdater, recv_req):
        """Update weights from IPC for checkpoint-engine integration."""
        self._assert_weight_cache_inactive("update_weights_from_ipc")
        error = _unsupported_derived_weight_cache_error()
        if error is not None:
            return False, error

        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self.get_model_runner())
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device):
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])


def _validate_update_weights_from_tensor_payload(
    named_tensors, load_format, tp_rank=None
):
    """Reject malformed payloads before any model parameter can be modified."""
    error_prefix = "Invalid update_weights_from_tensor payload"
    if load_format == "flattened_bucket":
        if not isinstance(named_tensors, dict):
            return f"{error_prefix}: flattened_bucket payload must be a mapping"
        if not {"flattened_tensor", "metadata"}.issubset(named_tensors):
            return (
                f"{error_prefix}: flattened_bucket payload requires "
                "flattened_tensor and metadata"
            )
        if not isinstance(named_tensors["flattened_tensor"], torch.Tensor):
            return f"{error_prefix}: flattened_tensor must be a torch.Tensor"
        if not isinstance(named_tensors["metadata"], (list, tuple)):
            return f"{error_prefix}: metadata must be a list or tuple"
        flattened_numel = named_tensors["flattened_tensor"].numel()
        for index, metadata in enumerate(named_tensors["metadata"]):
            if not isinstance(metadata, FlattenedTensorMetadata):
                return (
                    f"{error_prefix}: metadata entry {index} must be a "
                    "FlattenedTensorMetadata"
                )
            if not isinstance(metadata.name, str) or not metadata.name:
                return f"{error_prefix}: metadata entry {index} has an invalid name"
            if not isinstance(metadata.shape, (list, tuple, torch.Size)) or any(
                not isinstance(dimension, int) or dimension < 0
                for dimension in metadata.shape
            ):
                return f"{error_prefix}: metadata entry {index} has an invalid shape"
            if not isinstance(metadata.dtype, torch.dtype):
                return f"{error_prefix}: metadata entry {index} has an invalid dtype"
            offsets = (metadata.start_idx, metadata.end_idx, metadata.numel)
            if any(not isinstance(value, int) or value < 0 for value in offsets):
                return (
                    f"{error_prefix}: metadata entry {index} has invalid offsets"
                )
            if (
                metadata.end_idx < metadata.start_idx
                or metadata.end_idx > flattened_numel
                or metadata.end_idx - metadata.start_idx != metadata.numel
            ):
                return (
                    f"{error_prefix}: metadata entry {index} is outside the "
                    "flattened tensor"
                )
        return None

    if not isinstance(named_tensors, (list, tuple)):
        return f"{error_prefix}: named_tensors must be a list or tuple"

    for index, item in enumerate(named_tensors):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return f"{error_prefix}: entry {index} must be a (name, tensor) pair"
        name, tensor = item
        if not isinstance(name, str) or not name:
            return f"{error_prefix}: entry {index} has an invalid tensor name"
        if not isinstance(tensor, (torch.Tensor, LocalSerializedTensor)):
            return f"{error_prefix}: entry {index} has an invalid tensor value"
        if isinstance(tensor, LocalSerializedTensor):
            if not isinstance(tensor.values, (list, tuple)) or not tensor.values:
                return (
                    f"{error_prefix}: entry {index} has invalid serialized "
                    "tensor values"
                )
            if any(
                not isinstance(value, (bytes, bytearray, memoryview))
                for value in tensor.values
            ):
                return (
                    f"{error_prefix}: entry {index} has a non-bytes serialized "
                    "tensor value"
                )
            if tp_rank is not None and (
                not isinstance(tp_rank, int)
                or tp_rank < 0
                or tp_rank >= len(tensor.values)
            ):
                return (
                    f"{error_prefix}: entry {index} has no serialized tensor "
                    f"for TP rank {tp_rank}"
                )
    return None
