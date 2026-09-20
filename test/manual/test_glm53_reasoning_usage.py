"""Source-bound CPU regressions; GPU/serving imports intentionally excluded."""
import ast, json, logging, re, unittest, typing, enum, functools
from pathlib import Path
from types import SimpleNamespace as NS
ROOT = Path(__file__).resolve().parents[2] / 'python/sglang/srt'

def load(path, ns, names=None):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            continue
        if names is None or getattr(n, 'name', None) in names:
            body.append(n)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), 'exec'), ns)
    return ns
base_tree = ast.parse((ROOT / 'function_call/base_format_detector.py').read_text(encoding='utf-8'))
base = next((n for n in base_tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseFormatDetector'))
base.bases = []
base.body = [n for n in base.body if getattr(n, 'name', None) in ('__init__', '_get_tool_indices', '_ends_with_partial_token')]
ns = dict(vars(typing), json=json, re=re, logging=logging, Enum=enum.Enum, lru_cache=functools.lru_cache, safe_literal_eval=ast.literal_eval, ToolCallItem=lambda **kw: NS(**kw), StreamingParseResult=lambda normal_text='', calls=None: NS(normal_text=normal_text, calls=calls or []), get_model_structural_tag=None)
ns['envs'] = NS(SGLANG_FORWARD_UNKNOWN_TOOLS=NS(get=lambda: False))
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), base], type_ignores=[])), 'base', 'exec'), ns)
load(ROOT / 'function_call/glm47_moe_detector.py', ns)
Detector = ns['Glm47MoeDetector']

def tool(name, props):
    return NS(function=NS(name=name, parameters={'type': 'object', 'properties': props}))

def call(name, pairs=()):
    return '<tool_call>' + name + ''.join(('<arg_key>' + k + '</arg_key><arg_value>' + v + '</arg_value>' for k, v in pairs)) + '</tool_call>'

class CacheTests(unittest.TestCase):

    def test_reasoning_cap(self):
        tree = ast.parse((ROOT / 'managers/schedule_batch.py').read_text(encoding='utf-8'))
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Req'))
        method = next((n for n in cls.body if getattr(n, 'name', None) == '_cap_reasoning_tokens_at_finished_len'))
        scope = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'cap', 'exec'), scope)
        for finished, count, expected in [(3, 5, 3), (3, 2, 2), (None, 5, 5)]:
            req = NS(finished_len=finished, reasoning_tokens=count)
            scope[method.name](req)
            self.assertEqual(req.reasoning_tokens, expected)
if __name__ == '__main__':
    unittest.main(verbosity=2)
