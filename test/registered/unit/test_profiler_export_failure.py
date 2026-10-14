"""Focused legacy profiler lifecycle checks; no model server or GPU is contacted."""

import errno
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.managers.io_struct import ProfileReqType
from sglang.srt.managers.scheduler_components import profiler_manager as profiler_module
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _StoppedProfiler:
    def __init__(self, *, stop_error=None, export_error=None):
        self.stop_calls = 0
        self.export_calls = 0
        self.stop_error = stop_error
        self.export_error = export_error

    def stop(self):
        self.stop_calls += 1
        if self.stop_error:
            raise self.stop_error

    def export_chrome_trace(self, path):
        self.export_calls += 1
        if self.export_error:
            raise self.export_error


def _manager(**values):
    manager = object.__new__(SchedulerProfilerManager)
    defaults = {
        "profile_in_progress": False,
        "profile_by_stage": False,
        "merge_profiles": False,
        "torch_profiler": None,
        "torch_profiler_output_dir": Path("unused-profile-output"),
        "profiler_activities": ["CPU", "GPU"],
        "profile_prefix": "F1",
        "profile_id": "unit-profile",
        "dp_tp_cpu_group": object(),
        "rpd_profiler": None,
        "detailed_annotations": False,
        "profiler_start_forward_ct": 17,
        "profiler_target_forward_ct": 18,
        "get_forward_ct": lambda: 18,
    }
    defaults.update(values)
    for key, value in defaults.items():
        setattr(manager, key, value)
    return manager


@pytest.fixture(autouse=True)
def isolated_legacy_runtime():
    parallel = SimpleNamespace(
        tp_rank=0,
        dp_size=1,
        dp_rank=0,
        pp_size=1,
        pp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
    )
    with patch.object(
        profiler_module.envs.SGLANG_PROFILE_V2, "get", return_value=False
    ), patch.object(
        profiler_module, "get_parallel", return_value=parallel
    ), patch.object(
        profiler_module,
        "get_device",
        return_value=SimpleNamespace(gpu_id=0, base_gpu_id=0),
    ), patch.object(
        profiler_module, "set_detailed_annotations_enabled"
    ), patch.object(
        profiler_module.torch.distributed, "all_reduce"
    ) as collective, patch.object(
        profiler_module.torch.distributed, "barrier"
    ), patch.object(
        profiler_module, "_is_npu", False
    ):
        yield collective


def test_already_running_request_does_not_start_again():
    manager = _manager(profile_in_progress=True)
    request = SimpleNamespace(
        req_type=ProfileReqType.START_PROFILE,
        output_dir="unused-profile-output",
        start_step=None,
        num_steps=6,
        activities=["CPU", "GPU"],
        with_stack=False,
        record_shapes=True,
        profile_by_stage=False,
        profile_id="unit-profile",
        merge_profiles=False,
        profile_prefix="F1",
        detailed_annotations=True,
    )
    with patch.object(manager, "_start_profile") as start:
        result = manager._profile(request)
    assert not result.success
    start.assert_not_called()


def test_v2_configuration_retains_its_original_no_preflight_contract():
    manager = _manager(_profile_manager=Mock())
    manager._profile_manager.configure.return_value = SimpleNamespace(success=True)
    with patch.object(
        profiler_module.envs.SGLANG_PROFILE_V2, "get", return_value=True
    ), patch(
        "pathlib.Path.mkdir", side_effect=PermissionError("should not be consulted")
    ) as mkdir:
        result = manager._init_profile(
            None, None, 6, ["CPU"], False, False, True, "unit-profile"
        )
    assert result.success
    mkdir.assert_not_called()
    assert manager._profile_manager.configure.call_args.kwargs["output_dir"] is None


def test_auto_stop_mkdir_failure_stops_before_io_and_resets_state(
    tmp_path, caplog, isolated_legacy_runtime
):
    profiler = _StoppedProfiler()
    manager = _manager(
        profile_in_progress=True,
        torch_profiler=profiler,
        torch_profiler_output_dir=tmp_path,
    )

    def denied(*args, **kwargs):
        assert profiler.stop_calls == 1
        raise PermissionError(errno.EACCES, "private fixture path")

    with patch("pathlib.Path.mkdir", side_effect=denied), patch.object(
        manager, "_merge_profile_traces"
    ) as merge:
        manager._profile_batch_predicate(
            SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        )
    assert profiler.stop_calls == 1
    assert profiler.export_calls == 0
    assert manager.torch_profiler is None
    assert not manager.profile_in_progress
    assert manager.profiler_start_forward_ct is None
    merge.assert_not_called()
    assert "output mkdir" in caplog.text
    assert "PermissionError" in caplog.text
    assert "errno=13" in caplog.text
    assert "private fixture path" not in caplog.text
    assert manager._init_profile(
        str(tmp_path), None, 6, ["CPU"], False, False, False, "next-profile"
    ).success


def test_trace_export_failure_returns_management_error(
    tmp_path, isolated_legacy_runtime
):
    profiler = _StoppedProfiler(export_error=OSError(errno.ENOSPC, "fixture disk full"))
    manager = _manager(
        profile_in_progress=True,
        torch_profiler=profiler,
        torch_profiler_output_dir=tmp_path,
    )
    with patch.object(manager, "_merge_profile_traces") as merge:
        result = manager._stop_profile()
    assert not result.success
    assert "OSError" in result.message
    assert profiler.stop_calls == 1
    assert profiler.export_calls == 1
    merge.assert_not_called()


def test_success_exports_and_merges_once(tmp_path, isolated_legacy_runtime):
    profiler = _StoppedProfiler()
    manager = _manager(
        profile_in_progress=True,
        torch_profiler=profiler,
        torch_profiler_output_dir=tmp_path,
    )
    with patch.object(
        manager, "_merge_profile_traces", return_value=" merged"
    ) as merge:
        result = manager._stop_profile()
    assert result.success
    assert result.message == "Succeeded. merged"
    assert profiler.stop_calls == 1
    assert profiler.export_calls == 1
    merge.assert_called_once()


@pytest.mark.parametrize(
    "snapshot_error", [None, PermissionError(errno.EACCES, "fixture")]
)
def test_memory_and_cuda_cleanup_run_once(tmp_path, snapshot_error):
    manager = _manager(
        profile_in_progress=True,
        profiler_activities=["MEM", "CUDA_PROFILER"],
        torch_profiler_output_dir=tmp_path,
    )
    cudart = SimpleNamespace(cudaProfilerStop=Mock())
    with patch.object(
        profiler_module.torch.cuda, "cudart", return_value=cudart
    ), patch.object(
        profiler_module.torch.cuda.memory, "_dump_snapshot", side_effect=snapshot_error
    ) as dump, patch.object(
        profiler_module.torch.cuda.memory, "_record_memory_history"
    ) as history:
        result = manager._stop_profile()
    assert result.success == (snapshot_error is None)
    dump.assert_called_once()
    history.assert_called_once_with(enabled=None)
    cudart.cudaProfilerStop.assert_called_once()
    assert not manager.profile_in_progress


def test_cuda_profiler_only_does_not_touch_output_mount():
    manager = _manager(profile_in_progress=True, profiler_activities=["CUDA_PROFILER"])
    cudart = SimpleNamespace(cudaProfilerStop=Mock())
    with patch.object(profiler_module.torch.cuda, "cudart", return_value=cudart), patch(
        "pathlib.Path.mkdir", side_effect=PermissionError("unneeded output mount")
    ) as mkdir:
        result = manager._stop_profile()
    assert result.success
    mkdir.assert_not_called()
    cudart.cudaProfilerStop.assert_called_once()


def test_local_export_failure_does_not_strand_other_rank(tmp_path):
    rendezvous = Barrier(2, timeout=2)
    lock = Lock()
    flags = []
    calls = []

    def all_reduce(tensor, *, op, group):
        with lock:
            calls.append(group)
            flags.append(tensor.item())
        rendezvous.wait()
        tensor.fill_(max(flags))

    managers = [
        _manager(
            profile_in_progress=True,
            torch_profiler=_StoppedProfiler(
                export_error=(
                    PermissionError(errno.EACCES, "one rank's revoked mount")
                    if rank == 0
                    else None
                )
            ),
            torch_profiler_output_dir=tmp_path / str(rank),
        )
        for rank in range(2)
    ]
    with patch.object(
        profiler_module.torch.distributed, "all_reduce", side_effect=all_reduce
    ), patch.object(
        profiler_module.torch.distributed,
        "barrier",
        side_effect=lambda *args, **kwargs: rendezvous.wait(),
    ), patch.object(
        SchedulerProfilerManager, "_merge_profile_traces"
    ) as merge:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(manager._stop_profile) for manager in managers]
            results = [future.result(timeout=4) for future in futures]
    assert len(calls) == 2
    assert all(not result.success for result in results)
    assert all(not manager.profile_in_progress for manager in managers)
    assert "failure on another rank" in results[1].message
    merge.assert_not_called()


def test_cuda_runtime_failure_is_not_disguised_as_export_failure(tmp_path):
    fatal = RuntimeError("CUDA error: device-side assert triggered")
    manager = _manager(
        profile_in_progress=True,
        torch_profiler=_StoppedProfiler(stop_error=fatal),
        torch_profiler_output_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="device-side assert") as error:
        manager._stop_profile()
    assert error.value is fatal
    assert not manager.profile_in_progress


@pytest.mark.parametrize(
    "conversion_error", [None, PermissionError(errno.EACCES, "fixture")]
)
def test_rpd_cleanup_and_conversion_are_not_repeated(tmp_path, conversion_error):
    rpd = Mock()
    rpd_utils = ModuleType("sglang.srt.utils.rpd_utils")
    rpd_utils.rpd_to_chrome_trace = Mock(side_effect=conversion_error)
    manager = _manager(
        profile_in_progress=True,
        profiler_activities=["RPD"],
        rpd_profiler=rpd,
        rpd_profile_path="fixture.trace.json.gz",
        torch_profiler_output_dir=tmp_path,
    )
    with patch.dict(sys.modules, {rpd_utils.__name__: rpd_utils}):
        result = manager._stop_profile()
    assert result.success == (conversion_error is None)
    rpd.rangePop.assert_called_once()
    rpd.stop.assert_called_once()
    rpd.flush.assert_called_once()
    rpd_utils.rpd_to_chrome_trace.assert_called_once_with(
        "trace.rpd", "fixture.trace.json.gz"
    )
    assert manager.rpd_profiler is None
    assert manager.rpd_profile_path is None


def test_collective_failure_is_not_swallowed(tmp_path, isolated_legacy_runtime):
    fatal = RuntimeError("Gloo transport disconnected")
    isolated_legacy_runtime.side_effect = fatal
    manager = _manager(
        profile_in_progress=True,
        torch_profiler=_StoppedProfiler(),
        torch_profiler_output_dir=tmp_path,
    )
    with patch.object(
        profiler_module.torch.distributed, "barrier", side_effect=fatal
    ), pytest.raises(RuntimeError, match="Gloo") as error:
        manager._stop_profile()
    assert error.value is fatal
    assert not manager.profile_in_progress
