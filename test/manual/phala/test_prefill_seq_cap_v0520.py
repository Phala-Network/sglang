"""Installed SGLang v0.5.20 CPU gate; no AST extraction or synthetic module imports.

Run as a script in a disposable runc/network-none/no-GPU container. An optional
five-file hash-guarded overlay is installed BEFORE any SGLang import. This is an
installed-image-dependency/source-overlay gate, not a new immutable image gate.
"""

import argparse
import contextlib
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path("/sgl-workspace/sglang")
FILES = (
    "python/sglang/srt/arg_groups/fields/exec_.py",
    "python/sglang/srt/arg_groups/cuda_graph_hook.py",
    "python/sglang/srt/model_executor/cuda_graph_config.py",
    "python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py",
    "python/sglang/srt/managers/scheduler_components/dp_attn.py",
)
PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--suite", choices=("green", "red"), required=True)
PARSER.add_argument("--manifest", type=Path, required=True)
PARSER.add_argument("--install-guarded-overlay", type=Path)
PARSER.add_argument("--summary", type=Path, required=True)
OPTS = PARSER.parse_args()
RECEIPT = {
    "suite": OPTS.suite,
    "scope": "v0.5.20 installed imports plus real msgspec CLI/graph resolution/namespace and CPU replay eligibility/local DP vote; no model/GPU init or collectives",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_guards():
    if sys.platform != "linux" or not ROOT.is_dir():
        raise RuntimeError(
            "This gate must run inside the disposable Linux v0.5.20 container"
        )
    if any(Path("/dev").glob("nvidia[0-9]*")) or Path("/dev/nvidiactl").exists():
        raise RuntimeError("GPU device nodes are present; use runc without --gpus")
    if OPTS.suite == "red" and OPTS.install_guarded_overlay:
        raise RuntimeError("The red suite must use the unmodified frozen r10 source")
    manifest = json.loads(OPTS.manifest.read_text())
    entries = {entry["path"]: entry for entry in manifest["files"]}
    if set(entries) != set(FILES):
        raise RuntimeError("Manifest must describe exactly the reviewed five files")
    prepared = []
    for rel in FILES:
        destination = ROOT / rel
        if not destination.resolve().is_relative_to(ROOT.resolve()):
            raise RuntimeError("Destination escaped installed source root")
        current = destination.read_bytes()
        expected = entries[rel]
        if OPTS.install_guarded_overlay or OPTS.suite == "red":
            # Windows raw hashes may include CRLF; the reviewed LF digest is
            # also an exact accepted input identity for Linux installed files.
            permitted = {expected["before_raw_sha256"], expected["before_lf_sha256"]}
        else:
            permitted = {expected["after_sha256"]}
        if digest(current) not in permitted:
            raise RuntimeError(f"Installed source precondition mismatch: {rel}")
        if OPTS.install_guarded_overlay:
            payload = (OPTS.install_guarded_overlay / rel).read_bytes()
            if digest(payload) != expected["after_sha256"]:
                raise RuntimeError(f"Overlay payload mismatch: {rel}")
            prepared.append((destination, payload))
    # Validate all five before modifying any installed file. Writes exist only
    # in this disposable container's writable layer; no source bind mount.
    for destination, payload in prepared:
        destination.write_bytes(payload)
    installed = {rel: digest((ROOT / rel).read_bytes()) for rel in FILES}
    for rel, expected in entries.items():
        if OPTS.suite == "green" and installed[rel] != expected["after_sha256"]:
            raise RuntimeError(f"Post-overlay hash mismatch: {rel}")
    RECEIPT.update(
        source_head_before_overlay=manifest["source_head"],
        installed_hashes=installed,
        guarded_overlay=bool(prepared),
    )


def import_installed():
    # Real imports, with only explicit context publication through the public
    # runtime API. No sys.path edits, sys.modules insertion, AST/exec or mocks.
    global torch, msgspec, ServerArgs, prepare_server_args, ExecGraph
    global get_context, get_exec, Backend, CaptureHiddenMode, ForwardMode
    global ForwardBatch, ScheduleBatch, PrefillCudaGraphRunner, dp_attn
    global SpeculativeAlgorithm, parse_cuda_graph_config
    import msgspec
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before gate imports")
    from sglang.srt.runtime_context import get_context, get_exec
    from sglang.srt.server_args import ServerArgs, prepare_server_args

    # Seed actual NS(...) config bags before importing modules that read them.
    bootstrap = ServerArgs(model_path="dummy", device="cpu")
    bootstrap.resolve_once()
    get_context().set_server_args(bootstrap)
    import inspect

    from sglang.srt.arg_groups.cuda_graph_hook import parse_cuda_graph_config
    from sglang.srt.arg_groups.fields.exec_ import ExecGraph
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler_components import dp_attn
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    paths = {
        name: Path(inspect.getfile(obj)).resolve()
        for name, obj in {
            "ServerArgs": ServerArgs,
            "PrefillCudaGraphRunner": PrefillCudaGraphRunner,
            "ScheduleBatch": ScheduleBatch,
            "ForwardBatch": ForwardBatch,
            "dp_vote": dp_attn._local_prefill_cuda_graph_vote,
            "exec_graph_fields": ExecGraph,
        }.items()
    }
    for name, path in paths.items():
        if not path.is_relative_to((ROOT / "python/sglang").resolve()):
            raise RuntimeError(
                f"Imported {name} outside expected installed root: {path}"
            )
    if torch.cuda.is_initialized():
        raise RuntimeError("Installed imports initialized CUDA")
    RECEIPT["import_origins"] = {name: str(path) for name, path in paths.items()}
    RECEIPT["torch_version"] = torch.__version__


def resolved(cli=()):
    args = prepare_server_args(
        [
            "--model-path",
            "dummy",
            "--device",
            "cpu",
            "--context-length",
            "1048576",
            "--cuda-graph-backend-prefill",
            "breakable",
            "--cuda-graph-max-bs-prefill",
            "16384",
            *cli,
        ]
    )
    # The real pipeline returns early for model_path=dummy, before graph
    # hooks (pipeline.py/server_args.py). Invoke the real hook exactly once
    # for this CPU fixture, then publish its real resolution declarations.
    args.resolve_once()
    parse_cuda_graph_config(args)
    get_context().set_server_args(args)
    return args, get_exec().graph.cuda_graph_config.prefill


def make_runner(cap):
    # __init__ allocates/captures GPU buffers and is deliberately out of CPU
    # scope. The actual eligibility methods/helpers are left unmodified.
    runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
    runner.max_seq_len_prefill = cap
    runner._is_full_backend = False
    runner._capture_req_slots = 16
    runner.prefill_backend_name = Backend.BREAKABLE
    runner.has_mha_companion_layers = False
    runner.enable_lora = False
    runner._capture_chunked_prefix = False
    runner.capture_hidden_mode = CaptureHiddenMode.FULL
    runner.capture_num_tokens = [4, 8, 16, 1024, 2048, 4096, 8192, 16384]
    runner.max_num_tokens = 16384
    return runner


def make_batches(lengths, *, tokens=1024, prefixes=None, mode=None):
    mode = mode or ForwardMode.EXTEND
    bs = len(lengths)
    cpu_lengths = torch.tensor(lengths, dtype=torch.int64, device="cpu")
    prefixes = prefixes if prefixes is not None else [0] * bs
    forward = ForwardBatch(
        forward_mode=mode,
        batch_size=bs,
        input_ids=torch.zeros(tokens, dtype=torch.int64, device="cpu"),
        req_pool_indices=torch.arange(bs, dtype=torch.int64, device="cpu"),
        seq_lens=cpu_lengths.clone(),
        out_cache_loc=torch.zeros(tokens, dtype=torch.int64, device="cpu"),
        seq_lens_sum=sum(lengths),
        seq_lens_cpu=cpu_lengths,
        extend_prefix_lens_cpu=prefixes,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    # The real ScheduleBatch dataclass and batch_size method, without allocating
    # scheduler KV pools. Requests only supply row cardinality in EXTEND/MIXED.
    schedule = ScheduleBatch.__new__(ScheduleBatch)
    schedule.reqs = [object() for _ in lengths]
    schedule.forward_mode = mode
    schedule.extend_num_tokens = tokens
    schedule.input_embeds = None
    schedule.replace_embeds = None
    schedule.prefix_lens = prefixes
    schedule.return_logprob = False
    schedule.seq_lens_cpu = cpu_lengths
    return schedule, forward


def vote(runner, schedule):
    return dp_attn._local_prefill_cuda_graph_vote(
        local_batch=schedule,
        prefill_graph_runner=runner,
        coordinated_prefill=True,
        breakable_prefill=True,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        model_config=None,
    )


class BoundCase(unittest.TestCase):
    def setUp(self):
        # Standard installed runtime override, restored after every test.
        override = get_context().override_server_args(device="cpu")
        override.install()
        self.addCleanup(override.restore)

    def verdicts(self, expected, lengths, cap=16384, **kwargs):
        runner = make_runner(cap)
        schedule, forward = make_batches(lengths, **kwargs)
        self.assertEqual(vote(runner, schedule), expected, "actual DP local vote")
        self.assertEqual(runner.can_run_graph(forward), expected, "actual forward gate")
        self.assertFalse(torch.cuda.is_initialized())


class TestEligibilityCausal(BoundCase):
    """All six fail by AssertionError on unmodified v0.5.20; no new CLI/kwargs used."""

    def test_cold_cap_plus_one(self):
        self.verdicts(False, [8193], cap=8192, tokens=8193)

    def test_cached_prefix_plus_new(self):
        self.verdicts(False, [16385], tokens=4, prefixes=[16381])

    def test_mixed_one_long_member(self):
        self.verdicts(False, [16385, 100], prefixes=[15362, 99], mode=ForwardMode.MIXED)

    def test_near_1m(self):
        self.verdicts(False, [1000015], tokens=16384, prefixes=[983631])

    def test_missing_host_mirror(self):
        runner = make_runner(16384)
        schedule, forward = make_batches([20000])
        schedule.seq_lens_cpu = forward.seq_lens_cpu = None
        self.assertFalse(vote(runner, schedule))
        self.assertFalse(runner.can_run_graph(forward))

    def test_dp_group_long_rank_veto(self):
        runner = make_runner(16384)
        s1, f1 = make_batches([100])
        s2, f2 = make_batches([20000])
        # Same min reduction as MLPSyncBatchInfo, performed on CPU metadata.
        group = bool(min(vote(runner, s1), vote(runner, s2)))
        self.assertFalse(group)
        for forward in (f1, f2):
            forward.global_num_tokens_cpu = [1024, 1024]
            forward.can_run_dp_prefill_cuda_graph = group
            self.assertFalse(runner.can_run_graph(forward))


class TestEligibilityControls(BoundCase):
    def test_default_none_preserves_long_replay(self):
        self.verdicts(True, [1000015], cap=None)

    def test_exact_cap(self):
        self.verdicts(True, [16384], prefixes=[15360])

    def test_below_cap(self):
        self.verdicts(True, [16383])

    def test_per_request_not_aggregate(self):
        self.verdicts(True, [12000, 12000])

    def test_token_bucket_limit_remains(self):
        self.verdicts(False, [16000, 16000], tokens=17000)

    def test_padding_waste_limit_remains(self):
        self.verdicts(False, [17], tokens=17)

    def test_missing_host_unlimited_unchanged(self):
        runner = make_runner(None)
        schedule, forward = make_batches([1000015])
        schedule.seq_lens_cpu = forward.seq_lens_cpu = None
        self.assertTrue(vote(runner, schedule))
        self.assertTrue(runner.can_run_graph(forward))

    def test_idle_vote_unchanged(self):
        self.assertTrue(vote(make_runner(16384), None))


class TestInstalledConfig(BoundCase):
    def test_default_none_namespace_and_readback(self):
        args, phase = resolved()
        self.assertTrue(issubclass(type(args), msgspec.Struct))
        self.assertIn(
            "cuda_graph_max_seq_len_prefill",
            {field.name for field in msgspec.structs.fields(ExecGraph)},
        )
        self.assertIsNone(args.cuda_graph_max_seq_len_prefill)
        self.assertIsNone(phase.max_seq_len)
        self.assertIsNone(
            args.resolved_dict()["cuda_graph_config"]["prefill"]["max_seq_len"]
        )

    def test_cli_to_namespace_and_readback(self):
        args, phase = resolved(["--cuda-graph-max-seq-len-prefill", "16384"])
        self.assertEqual(phase.max_seq_len, 16384)
        self.assertEqual(
            args.resolved_dict()["cuda_graph_config"]["prefill"]["max_seq_len"], 16384
        )
        self.assertEqual(args.resolved_dict()["context_length"], 1048576)
        self.assertEqual(phase.max_bs, 16384)
        self.verdicts(False, [16385], cap=phase.max_seq_len)

    def test_json_wins_over_cli(self):
        _, phase = resolved(
            [
                "--cuda-graph-max-seq-len-prefill",
                "16384",
                "--cuda-graph-config",
                '{"prefill":{"max_seq_len":8192}}',
            ]
        )
        self.assertEqual(phase.max_seq_len, 8192)
        self.verdicts(False, [8193], cap=phase.max_seq_len)

    def test_json_null_disables_cap(self):
        _, phase = resolved(
            [
                "--cuda-graph-max-seq-len-prefill",
                "16384",
                "--cuda-graph-config",
                '{"prefill":{"max_seq_len":null}}',
            ]
        )
        self.assertIsNone(phase.max_seq_len)
        self.verdicts(True, [1000015], cap=phase.max_seq_len)

    def test_json_only(self):
        _, phase = resolved(["--cuda-graph-config", '{"prefill":{"max_seq_len":4096}}'])
        self.assertEqual(phase.max_seq_len, 4096)

    def test_reject_non_integer_cli(self):
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as exc,
        ):
            resolved(["--cuda-graph-max-seq-len-prefill", "1.5"])
        self.assertEqual(exc.exception.code, 2)

    def test_reject_zero_negative(self):
        for cap in ("0", "-1"):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                resolved(["--cuda-graph-max-seq-len-prefill", cap])

    def test_reject_bool_json(self):
        with self.assertRaises(ValueError):
            resolved(["--cuda-graph-config", '{"prefill":{"max_seq_len":true}}'])

    def test_decode_json_does_not_accept_prefill_cap(self):
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as exc,
        ):
            resolved(["--cuda-graph-config", '{"decode":{"max_seq_len":8192}}'])
        self.assertEqual(exc.exception.code, 2)


def main():
    source_guards()
    import_installed()
    classes = [TestEligibilityCausal, TestEligibilityControls]
    if OPTS.suite == "green":
        classes.append(TestInstalledConfig)
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in classes
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failures = [test.id() for test, _ in result.failures]
    errors = [test.id() for test, _ in result.errors]
    expected = {
        f"__main__.TestEligibilityCausal.{name}"
        for name in unittest.defaultTestLoader.getTestCaseNames(TestEligibilityCausal)
    }
    RECEIPT.update(
        tests_run=result.testsRun,
        failures=failures,
        errors=errors,
        skipped=[str(test) for test, _ in result.skipped],
        cuda_initialized=torch.cuda.is_initialized(),
    )
    if OPTS.suite == "red":
        qualified = (
            set(failures) == expected
            and not errors
            and not result.skipped
            and result.testsRun == 14
            and not torch.cuda.is_initialized()
        )
        RECEIPT["expected_red_failures"] = sorted(expected)
        RECEIPT["red_qualified"] = qualified
        return 1 if qualified else 2
    return (
        0
        if result.wasSuccessful()
        and result.testsRun == 23
        and not result.skipped
        and not torch.cuda.is_initialized()
        else 2
    )


if __name__ == "__main__":
    exit_code = 2
    try:
        exit_code = main()
    except BaseException as exc:
        RECEIPT["bootstrap_error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        RECEIPT["exit_code"] = exit_code
        OPTS.summary.parent.mkdir(parents=True, exist_ok=True)
        OPTS.summary.write_text(json.dumps(RECEIPT, indent=2), encoding="utf8")
    sys.exit(exit_code)
