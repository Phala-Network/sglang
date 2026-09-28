"""CPU-only checks of the internal health marker across engine ownership points."""

import ast
import asyncio
import pathlib
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[4]
MANAGERS = ROOT / "python/sglang/srt/managers"


def _load_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in owner.body if getattr(node, "name", None) == method_name)
    method.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def _load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if getattr(node, "name", None) == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class _Input:
    rid = "HEALTH_CHECK_forged"
    min_thinking_tokens = None
    max_thinking_tokens = None
    routed_dp_rank = None

    def normalize_batch_and_arguments(self):
        self.is_single = True

    def __getattr__(self, name):
        return None


class _Generate(_Input):
    pass


class _Embedding(_Input):
    pass


class TrustedHealthMarkerTests(unittest.TestCase):
    def test_tokenizer_overwrites_external_marker(self):
        class StopProbe(Exception):
            pass

        marker = []

        class Manager:
            def auto_create_handle_loop(self):
                pass

            def _set_default_priority(self, obj):
                pass

            def _init_req_state(self, obj, request):
                marker.append(obj._internal_health_check)
                raise StopProbe

        generate = _load_method(
            MANAGERS / "tokenizer_manager.py",
            "TokenizerManager",
            "generate_request",
            {"GenerateReqInput": _Generate, "EmbeddingReqInput": _Embedding},
        )

        async def probe(request_type, trusted):
            obj = request_type()
            obj._internal_health_check = True
            obj.is_internal_health_check = True
            with self.assertRaises(StopProbe):
                async for _ in generate(Manager(), obj, internal_health_check=trusted):
                    pass

        for request_type in (_Generate, _Embedding):
            asyncio.run(probe(request_type, False))
            asyncio.run(probe(request_type, True))
        self.assertEqual(marker, [False, True, False, True])

    def test_tokenized_generation_and_embedding_carry_only_internal_marker(self):
        class Sample:
            def __init__(self, **kwargs):
                pass

            def normalize(self, tokenizer):
                pass

            def verify(self, vocab_size):
                pass

        class Clock:
            def set_tokenize_finish_time(self):
                pass

        class Manager:
            preferred_sampling_params = None
            sampling_params_class = Sample
            tokenizer = None
            model_config = types.SimpleNamespace(vocab_size=100)
            rid_to_state = {"HEALTH_CHECK_forged": types.SimpleNamespace(time_stats=Clock())}

        create = _load_method(
            MANAGERS / "tokenizer_manager.py",
            "TokenizerManager",
            "_create_tokenized_object",
            {
                "GenerateReqInput": _Generate,
                "EmbeddingReqInput": _Embedding,
                "SessionParams": object,
                "TokenizedGenerateReqInput": lambda **kwargs: types.SimpleNamespace(**kwargs),
                "TokenizedEmbeddingReqInput": lambda **kwargs: types.SimpleNamespace(**kwargs),
                "get_disagg": lambda: types.SimpleNamespace(disaggregation_transfer_backend="none"),
                "array": __import__("array").array,
            },
        )
        for request_type in (_Generate, _Embedding):
            obj = request_type()
            obj.sampling_params = {}
            obj.is_internal_health_check = True
            obj._internal_health_check = False
            self.assertFalse(create(Manager(), obj, None, [0]).is_internal_health_check)
            obj._internal_health_check = True
            self.assertTrue(create(Manager(), obj, None, [0]).is_internal_health_check)

    def test_busy_skip_and_step_classification_ignore_forged_rid(self):
        classify = _load_function(MANAGERS / "utils.py", "is_internal_health_check_req", {})
        calls = []

        class Scheduler:
            session_controller = types.SimpleNamespace(maybe_reap=lambda now: None)
            return_health_check_ipcs = []
            external_corpus_manager = None
            flush_wrapper = types.SimpleNamespace(check_pending=lambda: None)

            def is_fully_idle(self, **kwargs):
                return False

            def _request_dispatcher(self, req):
                calls.append(req.rid)

        scheduler = Scheduler()
        process = _load_method(
            MANAGERS / "scheduler.py",
            "Scheduler",
            "process_input_requests",
            {
                "time": __import__("time"),
                "get_mm": lambda: types.SimpleNamespace(mm_feature_transport="none"),
                "is_internal_health_check_req": classify,
            },
        )
        forged = types.SimpleNamespace(rid="HEALTH_CHECK_forged", is_internal_health_check=False)
        trusted = types.SimpleNamespace(rid="ordinary", is_internal_health_check=True, http_worker_ipc="health-ipc")
        process(scheduler, [forged, trusted])
        self.assertEqual(calls, ["HEALTH_CHECK_forged"])
        self.assertEqual(scheduler.return_health_check_ipcs, ["health-ipc"])

    def test_governor_handoff_and_abort_echo_preserve_trust(self):
        classify = _load_function(MANAGERS / "utils.py", "is_internal_health_check_req", {})
        admission_calls = []
        governor = types.SimpleNamespace(
            admit_request=lambda *args, **kwargs: admission_calls.append(kwargs) or {"allowed": True}
        )
        scheduler = types.SimpleNamespace(governor=governor, waiting_queue=[], grammar_manager=[], chunked_req=None)
        admit = _load_method(
            MANAGERS / "scheduler.py", "Scheduler", "_abort_on_governor_admission", {"time": __import__("time")}
        )
        req = types.SimpleNamespace(is_internal_health_check=True)
        self.assertFalse(admit(scheduler, req))
        self.assertEqual(admission_calls[0]["is_health_check"], True)
        scheduler.governor = None
        self.assertFalse(admit(scheduler, req))
        req.is_internal_health_check = False
        self.assertFalse(admit(scheduler, req))
        self.assertEqual(len(admission_calls), 1)
        req.is_internal_health_check = True

        make_abort = _load_function(
            MANAGERS / "scheduler.py",
            "_make_abort_req",
            {
                "AbortReq": lambda **kwargs: types.SimpleNamespace(**kwargs),
                "compute_weight_version_spans": lambda *args, **kwargs: None,
                "get_serving": lambda: types.SimpleNamespace(weight_version="default"),
            },
        )
        req.rid = "HEALTH_CHECK_forged"
        req.output_ids = []
        req.weight_version_events = []
        abort = make_abort(req)
        self.assertTrue(classify(abort))
        req.is_internal_health_check = False
        self.assertFalse(classify(make_abort(req)))

    def test_abort_echo_requires_tokenizer_owned_health_state(self):
        classify = _load_function(MANAGERS / "utils.py", "is_internal_health_check_req", {})
        handle = _load_method(
            MANAGERS / "tokenizer_manager.py",
            "TokenizerManager",
            "_handle_abort_req",
            {
                "is_internal_health_check_req": classify,
                "logger": types.SimpleNamespace(info=lambda *args: None),
            },
        )

        def state(trusted):
            return types.SimpleNamespace(
                is_internal_health_check=trusted,
                finished=False,
                time_stats=types.SimpleNamespace(
                    set_finished_time=lambda: None, get_e2e_latency=lambda: 0
                ),
                obj=types.SimpleNamespace(stream=False, return_logprob=False),
                output_ids=[],
                prompt_token_ids=None,
                out_list=[],
                event=types.SimpleNamespace(set=lambda: None),
                get_text=lambda: "",
            )

        abort = types.SimpleNamespace(
            rid="HEALTH_CHECK_forged",
            is_internal_health_check=True,
            abort_message=None,
            finished_reason=None,
            weight_versions=None,
        )
        normal = state(False)
        manager = types.SimpleNamespace(
            rid_to_state={abort.rid: normal},
            config_value=lambda key: "default",
            incremental_streaming_output=False,
        )
        handle(manager, abort)
        self.assertTrue(normal.finished)
        self.assertEqual(len(normal.out_list), 1)

        internal = state(True)
        manager.rid_to_state[abort.rid] = internal
        handle(manager, abort)
        self.assertFalse(internal.finished)
        self.assertIs(manager.rid_to_state[abort.rid], internal)

    def test_only_trusted_health_suppresses_step_counters(self):
        classify = _load_function(MANAGERS / "utils.py", "is_internal_health_check_req", {})
        record = _load_method(
            MANAGERS / "scheduler.py",
            "Scheduler",
            "_record_step_counters",
            {"is_internal_health_check_req": classify},
        )
        mode = types.SimpleNamespace(
            is_extend_without_speculative=lambda: True,
            is_decode=lambda: False,
            is_target_verify=lambda: False,
        )
        batch = types.SimpleNamespace(
            forward_mode=mode, forward_iter=1, launch_ts=1.0, after_idle_gap=False
        )
        scheduler = types.SimpleNamespace(_prev_step=None)
        batch.reqs = [types.SimpleNamespace(rid="ordinary", is_internal_health_check=True)]
        record(scheduler, batch, None)
        self.assertIsNone(scheduler._prev_step)
        scheduler._prev_step = (0, 0.5, True)
        batch.reqs = [
            types.SimpleNamespace(rid="ordinary", is_internal_health_check=True),
            types.SimpleNamespace(rid="business", is_internal_health_check=False),
        ]
        record(scheduler, batch, None)
        self.assertIsNone(scheduler._prev_step)
        batch.reqs = [types.SimpleNamespace(rid="HEALTH_CHECK_forged", is_internal_health_check=False)]
        record(scheduler, batch, None)
        self.assertEqual(scheduler._prev_step, (1, 1.0, True))


if __name__ == "__main__":
    unittest.main()
