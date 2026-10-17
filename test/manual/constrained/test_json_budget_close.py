"""CPU qualification of JSON-budget policy; no model or CUDA execution.

The CUDA verifier's exact scalar body is compiled as a CPU host shim. Scheduler
methods and the EAGLE sampling function are extracted from the candidate source,
so these checks exercise production code while retaining an explicit CPU limit.
"""

import ast
import ctypes
import hashlib
import json
import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import xgrammar as xgr

from sglang.srt.constrained import json_budget_close as policy
from sglang.srt.constrained.base_grammar_backend import GrammarMask
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path("/work/implementation/evidence") / os.environ.get(
    "JSON_BUDGET_TEST_RUN", "cpu-v2"
)
AUDIT = Path("/audit")
EVIDENCE.mkdir(parents=True, exist_ok=False)
DETAILS = {}


def extracted(path, names, globals_=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert {n.name for n in nodes} == set(names)
    namespace = dict(globals_ or {})
    compile_tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(compile_tree), str(ROOT / path), "exec"),
        namespace,
    )
    return namespace


def host_verifier():
    path = ROOT / "python/sglang/kernels/jit/include/sgl_kernel/speculative/eagle.cuh"
    text = path.read_text()
    start = text.rfind("template <", 0, text.index("__global__ void VerifyTreeGreedy("))
    body = text.index("{", start)
    depth = 1
    end = body + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    exact = text[start:end]
    shim = (
        "#include <cstdint>\nstruct { uint32_t x; } blockIdx;\n#define __global__\n"
        + exact
        + """
extern "C" void host_verify(int32_t* p,int32_t* a,int32_t* n,int64_t* c,int64_t* i,int64_t* t,int64_t* s,int64_t* q,uint32_t bs,uint32_t ns,uint32_t nd) {
 for(uint32_t b=0;b<bs;++b) { blockIdx.x=b; VerifyTreeGreedy<int32_t,int64_t>(p,a,n,c,i,t,s,q,bs,ns,nd); }
}
"""
    )
    source = EVIDENCE / "verify_tree_greedy_host.cpp"
    library = EVIDENCE / "verify_tree_greedy_host.so"
    source.write_text(shim)
    subprocess.run(
        ["g++", "-shared", "-fPIC", "-O2", str(source), "-o", str(library)],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(library))
    lib.host_verify.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_uint32] * 3

    def verify(
        *,
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index=None,
        retrieve_index=None,
        retrive_next_token=None,
        retrieve_next_token=None,
        retrive_next_sibling=None,
        retrieve_next_sibling=None,
        target_predict,
        topk=1,
    ):
        args = [
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrive_index if retrive_index is not None else retrieve_index,
            (
                retrive_next_token
                if retrive_next_token is not None
                else retrieve_next_token
            ),
            (
                retrive_next_sibling
                if retrive_next_sibling is not None
                else retrieve_next_sibling
            ),
            target_predict,
        ]
        assert all(x.device.type == "cpu" and x.is_contiguous() for x in args)
        lib.host_verify(
            *[x.data_ptr() for x in args],
            candidates.shape[0],
            accept_index.shape[1],
            candidates.shape[1],
        )
        return predicts, accept_index, accept_token_num

    DETAILS["host_verifier"] = {
        "kind": "exact CUDA scalar acceptance body in CPU-only host shim; no CUDA execution",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "extracted_body_sha256": hashlib.sha256(exact.encode()).hexdigest(),
        "shim_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    return verify


class BudgetCloseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized() and not list(
            Path("/dev").glob("nvidia*")
        )
        meta = AUDIT / "tokenizer-meta"
        cfg = json.loads((meta / "config.json").read_text())
        cls.generation = json.loads((meta / "generation_config.json").read_text())
        settings = json.loads(
            (
                AUDIT
                / "gpu0-combined-candidate-v1/results-representatives/server-info.json"
            ).read_text()
        )
        cls.tokenizer = get_tokenizer(
            str(meta),
            tokenizer_mode=settings["tokenizer_mode"],
            tokenizer_backend=settings["tokenizer_backend"],
            trust_remote_code=False,
            local_files_only=True,
        )
        cls.backend = XGrammarGrammarBackend(
            cls.tokenizer,
            cfg["vocab_size"],
            model_eos_token_ids=[2],
            any_whitespace=True,
            max_whitespace_cnt=1,
        )
        cls.info = (
            cls.tokenizer.init_xgrammar()[0]
            if hasattr(cls.tokenizer, "init_xgrammar")
            else xgr.TokenizerInfo.from_huggingface(
                cls.tokenizer, vocab_size=cfg["vocab_size"], stop_token_ids=[2]
            )
        )
        cls.vocab = cls.info.decoded_vocab
        cls.vocab_size = cfg["vocab_size"]
        excluded = set(cls.info.special_token_ids) | {2}
        cls.byte_index = {}
        for token, raw in enumerate(cls.vocab):
            if raw and token not in excluded:
                cls.byte_index.setdefault(raw[0], []).append((token, raw))
        cls.verify = staticmethod(host_verifier())
        cls.traverse = staticmethod(
            extracted(
                "python/sglang/srt/speculative/spec_utils.py",
                ["traverse_tree"],
                {"torch": torch},
            )["traverse_tree"]
        )
        cls.eagle_sample = staticmethod(
            extracted(
                "python/sglang/srt/speculative/eagle_utils.py",
                ["_verify_uses_greedy", "eagle_sample"],
                {
                    "torch": torch,
                    "envs": envs,
                    "_is_cpu": True,
                    "_is_hip": False,
                    "_is_xpu": False,
                    "_is_cuda": False,
                    "_is_npu": False,
                    "get_spec": lambda: NS(speculative_use_rejection_sampling=False),
                    "verify_tree_greedy_func": cls.verify,
                    "logger": policy.logger,
                },
            )["eagle_sample"]
        )
        result_path = (
            "python/sglang/srt/managers/scheduler_components/batch_result_processor.py"
        )
        tree = ast.parse((ROOT / result_path).read_text())
        methods = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef)
            and any(
                isinstance(x, ast.FunctionDef) and x.name == "advance_grammar_fsm"
                for x in n.body
            )
        )
        selected = [
            n
            for n in methods.body
            if isinstance(n, ast.FunctionDef)
            and n.name
            in (
                "_accept_grammar_tokens",
                "advance_grammar_fsm",
                "_resolve_spec_v2_tokens",
            )
        ]
        helper = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_get_speculative_output_stride"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                helper,
                ast.ClassDef(
                    name="Processor",
                    bases=[],
                    keywords=[],
                    body=selected,
                    decorator_list=[],
                ),
            ],
            type_ignores=[],
        )
        namespace = {}
        exec(
            compile(ast.fix_missing_locations(module), str(ROOT / result_path), "exec"),
            namespace,
        )
        cls.processor = namespace["Processor"]()
        cls.processor.model_worker = NS(
            on_verify_complete_cpu=lambda *args, **kwargs: None
        )
        DETAILS["resolved_default_repetition_penalty"] = cls.generation.get(
            "repetition_penalty"
        )

    def setUp(self):
        self.enable = envs.SGLANG_ENABLE_JSON_BUDGET_CLOSE.override(True)
        self.enable.__enter__()

    def tearDown(self):
        self.enable.__exit__(None, None, None)

    @classmethod
    def encode(cls, raw):
        if isinstance(raw, str):
            raw = raw.encode()
        best = {0: ()}
        for offset in range(len(raw)):
            if offset not in best:
                continue
            for token, piece in cls.byte_index.get(raw[offset], ()):
                if raw.startswith(piece, offset):
                    end, ids = offset + len(piece), (*best[offset], token)
                    if end not in best or (len(ids), ids) < (len(best[end]), best[end]):
                        best[end] = ids
        assert len(raw) in best
        return list(best[len(raw)])

    def request(self, schema, prefix, *, hard_cap=None, wrapped=True):
        key = json.dumps(schema, separators=(",", ":"))
        ids = self.encode(prefix) if isinstance(prefix, (bytes, str)) else prefix
        p = SamplingParams(
            max_new_tokens=hard_cap or len(ids) + 16,
            temperature=0,
            json_schema=key,
            repetition_penalty=self.generation.get("repetition_penalty", 1.05),
        )
        p.normalize(self.tokenizer)
        req = Req("cpu-fixture", "", [], p, eos_token_ids={2}, require_reasoning=False)
        grammar = self.backend.dispatch_json(key)
        if wrapped:
            grammar = ReasonerGrammarObject(grammar, [3])
            grammar.maybe_init_reasoning(False)
        for token in ids:
            grammar.accept_token(token)
        req.grammar = grammar
        return req

    def tree(self, reqs, children=None):
        width = 4
        rows = []
        for row, req in enumerate(reqs):
            grammar = (
                req.grammar.grammar
                if isinstance(req.grammar, ReasonerGrammarObject)
                else req.grammar
            )
            root = (
                grammar.accepted_tokens[-1]
                if grammar is not None and grammar.accepted_tokens
                else 174
            )
            rows.append(
                [root, *(children[row] if children is not None else [838, 174, 2])]
            )
        return NS(
            draft_token_num=width,
            tree_topk=1,
            max_tree_depth=width,
            draft_token=torch.tensor(rows, dtype=torch.int64).reshape(-1),
            retrieve_index=torch.arange(len(rows) * width, dtype=torch.int64).reshape(
                -1, width
            ),
            retrieve_next_token=torch.tensor(
                [[1, 2, 3, -1]] * len(rows), dtype=torch.int64
            ),
            retrieve_next_sibling=torch.full((len(rows), width), -1, dtype=torch.int64),
        )

    def batch(self, reqs):
        return NS(
            reqs=reqs,
            has_grammar=True,
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            sampling_info=NS(is_all_greedy=True, has_custom_logit_processor=False),
            return_logprob=False,
            return_hidden_states=False,
        )

    def select(self, reqs, predictions=None, children=None, **kwargs):
        tree = self.tree(reqs, children)
        target = torch.tensor(
            predictions or [[838, 838, 838, 838]] * len(reqs), dtype=torch.int64
        )
        result = policy.maybe_apply_json_budget_close(
            tree,
            self.batch(reqs),
            target,
            tp_size=kwargs.get("tp_size", 1),
            pp_size=kwargs.get("pp_size", 1),
        )
        return tree, target, result

    def test_01_real_frozen_cohorts_and_barrier_ahead(self):
        outcomes = []
        for relative, case, number, position, expected in [
            (
                "gpu0-mhc-reference-singlepoint-v1/results-representatives",
                "35",
                "03",
                1991,
                1993,
            ),
            (
                "gpu0-combined-candidate-v1/results-representatives",
                "17",
                "02",
                1988,
                1990,
            ),
            (
                "gpu0-combined-candidate-v1/results-representatives",
                "35",
                "03",
                1987,
                1989,
            ),
        ]:
            directory = AUDIT / relative
            request = json.loads(
                (
                    directory / "requests" / f"{number}-reasoning-off-{case}.json"
                ).read_text()
            )
            source = next(
                json.loads(line)
                for line in (directory / "GENERATED-TOKEN-IDS.jsonl")
                .read_text()
                .splitlines()
                if json.loads(line)["case_id"] == "reasoning-off-" + case
            )
            ids = source["generated_token_ids"]
            req = self.request(
                request["response_format"]["json_schema"]["schema"],
                ids[: position - 1],
                hard_cap=request["max_tokens"],
            )
            self.assertEqual(req.sampling_params.top_k, 1)
            self.assertEqual(req.sampling_params.repetition_penalty, 1.05)
            self.assertEqual(len(req.output_ids), 0)  # deliberately behind the barrier
            self.assertIsNotNone(policy._request_grammar(req))
            original_history = list(req.grammar.grammar.accepted_tokens)
            for original, replacement in [
                (838, 174),
                (self.encode(",")[0], 32678),
                (2, 2),
            ]:
                inner = req.grammar.grammar
                prefix = b"".join(self.vocab[t] for t in inner.accepted_tokens)
                self.assertEqual(
                    policy._select(
                        inner,
                        inner.matcher,
                        prefix,
                        len(inner.accepted_tokens),
                        request["max_tokens"],
                        original,
                    ),
                    replacement,
                )
                _, target, selected = self.select([req], predictions=[[original] * 4])
                self.assertEqual(int(selected[0, 0]), replacement)
                req.grammar.accept_token(replacement)
            self.assertTrue(req.grammar.is_terminated())
            final_ids = req.grammar.grammar.accepted_tokens
            self.assertEqual(len(final_ids), expected)
            self.assertEqual(final_ids[: position - 1], original_history)
            final = b"".join(self.vocab[t] for t in final_ids if t != 2)
            import jsonschema

            jsonschema.Draft202012Validator(
                request["response_format"]["json_schema"]["schema"]
            ).validate(json.loads(final))
            outcomes.append(
                {
                    "source": relative,
                    "case": case,
                    "proposal_position": position,
                    "proposal_id": ids[position - 1],
                    "resolved_top_k": req.sampling_params.top_k,
                    "resolved_repetition_penalty": req.sampling_params.repetition_penalty,
                    "fsm_history_used_while_output_ids_empty": True,
                    "final_tokens_including_eos": len(final_ids),
                    "strictly_below_H": len(final_ids) < request["max_tokens"],
                }
            )
        DETAILS["frozen_replays"] = outcomes

    def test_02_default_off_identity(self):
        req = self.request(
            {"type": "array", "items": {"type": "string"}}, "[" + '"x",' * 20 + '"last'
        )
        with envs.SGLANG_ENABLE_JSON_BUDGET_CLOSE.override(False):
            _, target, result = self.select([req])
        self.assertIs(result, target)
        req.sampling_params.max_new_tokens = 4096
        with patch.object(
            policy, "_vocab", side_effect=AssertionError("far-from-cap lookup")
        ):
            _, target, result = self.select([req])
        self.assertIs(result, target)

    def test_03_match_recomputes_child_and_mismatch_returns_bonus(self):
        req = self.request(
            {
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["values"],
                "additionalProperties": False,
            },
            '{"values":["x',
        )
        original = list(req.grammar.grammar.accepted_tokens)
        tree, target, selected = self.select(
            [req],
            predictions=[[838, self.encode(",")[0], 2, 2]],
            children=[[174, 32678, 2]],
        )
        self.assertEqual(selected[0].tolist()[:3], [174, 32678, 2])
        predicts = torch.zeros(4, dtype=torch.int32)
        indices = torch.full((1, 4), -1, dtype=torch.int32)
        count = torch.empty(1, dtype=torch.int32)
        self.verify(
            predicts=predicts,
            accept_index=indices,
            accept_token_num=count,
            candidates=tree.draft_token.reshape(1, 4),
            retrieve_index=tree.retrieve_index,
            retrieve_next_token=tree.retrieve_next_token,
            retrieve_next_sibling=tree.retrieve_next_sibling,
            target_predict=selected,
        )
        self.assertEqual(indices[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(int(count[0]), 3)
        self.assertEqual(predicts[:3].tolist(), [174, 32678, 2])
        self.assertEqual(req.grammar.grammar.accepted_tokens, original)
        # Mismatch after a matching quote returns the newly derived ]} bonus.
        tree, _, selected = self.select(
            [req],
            predictions=[[838, self.encode(",")[0], 838, 838]],
            children=[[174, 838, 2]],
        )
        predicts.zero_()
        indices.fill_(-1)
        self.verify(
            predicts=predicts,
            accept_index=indices,
            accept_token_num=count,
            candidates=tree.draft_token.reshape(1, 4),
            retrieve_index=tree.retrieve_index,
            retrieve_next_token=tree.retrieve_next_token,
            retrieve_next_sibling=tree.retrieve_next_sibling,
            target_predict=selected,
        )
        self.assertEqual(int(count[0]), 1)
        self.assertEqual(indices[0].tolist(), [0, 1, -1, -1])
        self.assertEqual(predicts[:2].tolist(), [174, 32678])
        self.assertEqual(req.grammar.grammar.accepted_tokens, original)

    def test_04_actual_tree_dfs_rollback_copy_and_idempotent_commit(self):
        req = self.request({"type": "array", "items": {"type": "string"}}, '["value')
        before = list(req.grammar.grammar.accepted_tokens)
        mask_before = xgr.allocate_token_bitmask(1, self.vocab_size)
        req.grammar.fill_vocab_mask(mask_before, 0)
        bracket = self.encode("]")[0]
        tree = self.tree([req], children=[[174, bracket, 2]])
        mask = xgr.allocate_token_bitmask(4, self.vocab_size)
        self.traverse(
            tree.retrieve_next_token[0],
            tree.retrieve_next_sibling[0],
            tree.draft_token,
            req.grammar,
            mask,
            self.vocab_size,
        )
        after = xgr.allocate_token_bitmask(1, self.vocab_size)
        req.grammar.fill_vocab_mask(after, 0)
        self.assertTrue(torch.equal(mask_before, after))
        self.assertEqual(req.grammar.grammar.accepted_tokens, before)
        copied = req.grammar.grammar.copy()
        self.assertEqual(copied.accepted_tokens, [])
        self.assertFalse(copied.matcher.is_completed())
        self.assertEqual(req.grammar.grammar.accepted_tokens, before)
        # Source methods: barrier advances once; result processor reuses retained run.
        result = NS(
            next_token_ids=torch.tensor([174, bracket, 2, 838]),
            accept_lens=torch.tensor([4]),
            copy_done=None,
            grammar_advanced=False,
            num_non_draft_tokens_per_req=1,
            speculative_num_draft_tokens=4,
            block_accept_lens=None,
            cap_lens=None,
            speculative_output_stride=4,
        )
        batch = self.batch([req])
        batch.forward_mode = NS(is_decode=lambda: True, is_extend=lambda: False)
        req.kv.kv_committed_len = len(before)
        self.processor.advance_grammar_fsm(result, batch)
        self.processor.advance_grammar_fsm(result, batch)
        self.assertEqual(result.grammar_retained_tokens, [[174, bracket, 2]])
        self.assertEqual(
            req.grammar.grammar.accepted_tokens, [*before, 174, bracket, 2]
        )
        accepted = self.processor._resolve_spec_v2_tokens(result, batch)
        self.assertEqual(accepted, [[174, bracket, 2]])
        self.assertEqual(req.kv.kv_committed_len, len(before) + 3)
        self.assertEqual(
            req.grammar.grammar.accepted_tokens, [*before, 174, bracket, 2]
        )

    def test_05_schema_required_and_minitems(self):
        schemas = [
            (
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
                '{"a":"value"',
            ),
            (
                {"type": "array", "items": {"type": "string"}, "minItems": 3},
                '["one","two"',
            ),
        ]
        for schema, prefix in schemas:
            req = self.request(schema, prefix)
            _, target, result = self.select(
                [req], predictions=[[self.encode(",")[0]] * 4]
            )
            self.assertTrue(torch.equal(result, target))

    def test_06_escaped_partial_numeric_and_business_tail(self):
        schema = {"type": "array", "items": {"type": "string"}}
        req = self.request(schema, b'["a\\"b')
        self.assertEqual(self.vocab[int(self.select([req])[2][0, 0])], b'"]')
        # No proposal ends the string: preserve ongoing business text.
        token = self.encode("more")[0]
        _, target, result = self.select([req], predictions=[[token] * 4])
        self.assertTrue(torch.equal(result, target))
        req = self.request(schema, b'["dangling\\')
        before = list(req.grammar.grammar.accepted_tokens)
        _, target, result = self.select([req])
        self.assertEqual(
            int(result[0, 0]), int(target[0, 0])
        )  # escaped quote stays content
        self.assertEqual(
            self.vocab[int(result[0, 1])], b'"]'
        )  # later genuine quote closes
        self.assertEqual(req.grammar.grammar.accepted_tokens, before)
        req = self.request({"type": "array", "items": {"type": "number"}}, "[12")
        _, target, result = self.select([req], predictions=[[self.encode("3")[0]] * 4])
        self.assertTrue(torch.equal(result, target))

    def test_07_mixed_requests_and_unsupported(self):
        schema = {"type": "array", "items": {"type": "string"}}
        eligible = self.request(schema, '["value')
        unsupported = self.request(schema, '["value')
        unsupported.return_logprob = True
        _, target, result = self.select([eligible, unsupported])
        self.assertEqual(self.vocab[int(result[0, 0])], b'"]')
        self.assertTrue(torch.equal(result[1], target[1]))
        for attribute, value in [
            ("require_reasoning", True),
            ("is_retracted", True),
            ("to_finish", object()),
            ("return_sampling_mask", True),
            ("custom_logit_processor", "opaque"),
        ]:
            req = self.request(schema, '["value')
            setattr(req, attribute, value)
            _, target, result = self.select([req])
            self.assertIs(result, target)
        for kwargs in [{"tp_size": 2}, {"pp_size": 2}]:
            _, target, result = self.select([eligible], **kwargs)
            self.assertIs(result, target)
        for attr, value in [
            ("ignore_eos", True),
            ("n", 2),
            ("min_new_tokens", 3),
            ("stop_strs", ["stop"]),
            ("top_k", 2),
        ]:
            req = self.request(schema, '["value')
            setattr(req.sampling_params, attr, value)
            _, target, result = self.select([req])
            self.assertIs(result, target)

        for attribute, value in [("tree_topk", 2), ("max_tree_depth", 3)]:
            tree = self.tree([eligible])
            setattr(tree, attribute, value)
            target = torch.full((1, 4), 838, dtype=torch.int64)
            result = policy.maybe_apply_json_budget_close(
                tree, self.batch([eligible]), target, tp_size=1, pp_size=1
            )
            self.assertIs(result, target)
        tree = self.tree([eligible])
        tree.retrieve_next_sibling[0, 0] = 1
        target = torch.full((1, 4), 838, dtype=torch.int64)
        result = policy.maybe_apply_json_budget_close(
            tree, self.batch([eligible]), target, tp_size=1, pp_size=1
        )
        self.assertIs(result, target)

    def test_08_strict_cap_and_complete_boundary(self):
        schema = {"type": "array", "items": {"type": "string"}}
        req = self.request(schema, "[" + '"x",' * 20 + '"last"')
        count = len(req.grammar.grammar.accepted_tokens)
        req.sampling_params.max_new_tokens = count + 2
        _, target, result = self.select([req], predictions=[[self.encode(",")[0]] * 4])
        self.assertIs(result, target)  # ] + EOS would end exactly at H
        req.sampling_params.max_new_tokens = count + 3
        selected = self.select([req], predictions=[[self.encode(",")[0]] * 4])[2]
        self.assertEqual(self.vocab[int(selected[0, 0])], b"]")

    def test_09_actual_eagle_sampling_after_penalty_and_mask(self):
        req = self.request({"type": "array", "items": {"type": "string"}}, '["value')
        tree = self.tree([req])
        before = list(req.grammar.grammar.accepted_tokens)
        mask = xgr.allocate_token_bitmask(4, self.vocab_size)
        self.traverse(
            tree.retrieve_next_token[0],
            tree.retrieve_next_sibling[0],
            tree.draft_token,
            req.grammar,
            mask,
            self.vocab_size,
        )
        logits = torch.full((4, self.vocab_size), -1000.0)
        logits[:, 838] = 10.0
        logits[:, 174] = 1.0
        batch = self.batch([req])
        batch.seq_lens = torch.tensor([100])
        batch.device = "cpu"
        batch.forward_mode = NS(is_idle=lambda: False)
        batch.sampling_info.acc_additive_penalties = None
        batch.sampling_info.acc_scaling_penalties = torch.ones((1, self.vocab_size))
        batch.sampling_info.acc_scaling_penalties[0, 838] = 1.05
        batch.sampling_info.logit_bias = None
        with patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=NS(tp_size=1, pp_size=1),
        ):
            predict, lens, index = self.eagle_sample(
                tree,
                batch,
                NS(next_token_logits=logits),
                GrammarMask(req.grammar, mask),
            )
        self.assertEqual(int(lens[0]), 1)
        self.assertEqual(self.vocab[int(predict[0])], b'"]')
        self.assertEqual(index[0].tolist(), [0, -1, -1, -1])
        self.assertEqual(req.grammar.grammar.accepted_tokens, before)
        DETAILS["actual_eagle_sampling"] = {
            "original_penalty": 1.05,
            "after_original_mask_and_penalty": True,
            "changed_bonus_bytes_hex": "225d",
            "accepted_length": 1,
            "request_matcher_unchanged": True,
        }


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BudgetCloseTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {
        "kind": "CPU source/matcher qualification, not GPU/runtime acceptance",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "gpu_or_model_weights_loaded": False,
        "details": DETAILS,
        "failure_traces": [
            {"test": str(t), "trace": v} for t, v in [*result.failures, *result.errors]
        ],
    }
    (EVIDENCE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    raise SystemExit(not result.wasSuccessful())
