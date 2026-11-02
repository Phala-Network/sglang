"""Dependency-free regression of extracted production methods, not GPU replay.

Run directly with Python. Real tensor/CUDA integration remains a separate gate.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
RUNNER = ROOT / "python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py"
HOOK = ROOT / "python/sglang/srt/arg_groups/cuda_graph_hook.py"


def extract(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class PrefillGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host_max = staticmethod(extract(RUNNER, "_batch_max_seq_len_cpu", {}))
        cls.eligible = staticmethod(
            extract(
                RUNNER,
                "can_replay_locally",
                {
                    "Backend": NS(BREAKABLE="breakable"),
                    "is_cuda": lambda: True,
                    "_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR": 4,
                },
            )
        )

    def vote(self, cap, maximum, metadata_cap=None):
        runner = NS(
            _is_full_backend=False,
            prefill_backend_name="breakable",
            has_mha_companion_layers=False,
            capture_hidden_mode=0,
            _has_uncapturable_chunked_prefix=lambda _: False,
            max_seq_len_prefill=cap,
            max_context_size=metadata_cap,
            _captured_attn_metadata_max_bs=None,
            _qwen_bcg_hc_sidechannel=False,
            _qwen_bcg_mtp_draft=False,
            _fa4_prefill=False,
        )
        return self.eligible(
            runner,
            batch_size=8,
            num_tokens=None,
            input_embeds=None,
            replace_embeds=None,
            prefix_lens=None,
            is_target_verify=False,
            capture_hidden_mode=None,
            return_logprob=False,
            batch_max_context_len=maximum,
        )

    def test_c8_per_request_boundaries(self):
        for length, expected in [(65535, True), (65536, True), (65537, False)]:
            with self.subTest(length=length):
                self.assertEqual(
                    self.vote(65536, self.host_max([length] * 8, 8)), expected
                )

    def test_missing_short_non_cpu_fail_closed(self):
        class GpuLengths:
            device = NS(type="cuda")

            def __len__(self):
                raise AssertionError("GPU data must not be inspected")

        for mirror in [None, [1] * 7, GpuLengths()]:
            self.assertFalse(self.vote(65536, self.host_max(mirror, 8)))
        self.assertEqual(self.host_max([], 0), 0)
        self.assertEqual(self.host_max([10] * 8 + [999999], 8), 10)

    def test_host_tensor_protocol(self):
        class HostLengths(list):
            device = NS(type="cpu")

            def __getitem__(self, item):
                value = super().__getitem__(item)
                return HostLengths(value) if isinstance(item, slice) else value

            def tolist(self):
                return list(self)

        self.assertEqual(self.host_max(HostLengths([1, 65536]), 2), 65536)

    def test_independent_metadata_cap_and_unset(self):
        self.assertTrue(self.vote(None, None))
        self.assertTrue(self.vote(None, 999999))
        self.assertFalse(self.vote(65536, 60000, metadata_cap=32768))
        self.assertTrue(self.vote(65536, 60000, metadata_cap=None))

    def test_dcp1_planner_is_noop(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("DCP1 must not prepare metadata or access pools")

        method = extract(
            RUNNER,
            "_plan_dcp_metadata",
            {
                "get_parallel": lambda: NS(attn_dcp_size=1),
                "get_req_to_token_pool": forbidden,
                "get_token_to_kv_pool": forbidden,
                "forward_context": forbidden,
            },
        )
        runner = NS(
            model_runner=NS(
                model=NS(prepare_context_parallel_metadata_for_dcp=forbidden)
            )
        )
        self.assertIsNone(method(runner, NS(batch_size=8)))

    def test_live_batch_assignment_preserved(self):
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        matches = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Name)
            and n.value.id == "forward_batch"
            and any(
                isinstance(t, ast.Attribute) and t.attr == "live_forward_batch"
                for t in n.targets
            )
        ]
        self.assertTrue(matches, "PR42265 replay binding must remain present")

    def test_config_precedence_and_validation(self):
        class Config:
            def __init__(self):
                self.prefill = NS(max_seq_len=None)
                self.decode = NS()

        def declare(args, source, **kwargs):
            args.__dict__.update(kwargs)

        parse = extract(
            HOOK,
            "parse_cuda_graph_config",
            {
                "resolving_view": lambda args: args,
                "CudaGraphConfig": Config,
                "default_cuda_graph_config": Config,
                "declare_resolution": declare,
                "Phase": NS(DECODE="decode", PREFILL="prefill"),
                "Backend": NS(DISABLED="disabled"),
            },
        )

        def args(value, explicit=None):
            fields = dict.fromkeys(
                [
                    "disable_cuda_graph",
                    "disable_prefill_cuda_graph",
                    "disable_decode_cuda_graph",
                    "cuda_graph_backend_decode",
                    "cuda_graph_backend_prefill",
                    "cuda_graph_max_bs_decode",
                    "cuda_graph_max_bs_prefill",
                    "cuda_graph_bs_decode",
                    "cuda_graph_bs_prefill",
                    "cuda_graph_tc_compiler",
                    "cuda_graph_prefill_max_context",
                ],
                None,
            )
            return NS(
                **fields,
                cuda_graph_config=explicit,
                cuda_graph_max_seq_len_prefill=value,
            )

        for value in [None, 1, 65536]:
            config = args(value)
            parse(config)
            self.assertEqual(config.cuda_graph_config.prefill.max_seq_len, value)
        for value in [0, -1, True, 1.5, "65536"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(args(value))
            with self.subTest(json_value=value), self.assertRaises(ValueError):
                parse(args(65536, {"prefill": {"max_seq_len": value}}))
        config = args(0, {"prefill": {"max_seq_len": 65536}})
        parse(config)
        self.assertEqual(config.cuda_graph_config.prefill.max_seq_len, 65536)


if __name__ == "__main__":
    unittest.main()
