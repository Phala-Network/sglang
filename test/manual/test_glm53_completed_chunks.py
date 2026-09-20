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

    def test_chunk_backup_after_rotation_guard_and_storage_unchanged(self):
        tree = ast.parse((ROOT / 'mem_cache/unified_radix_cache.py').read_text(encoding='utf-8'))
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'UnifiedRadixCache'))
        m = next((n for n in cls.body if getattr(n, 'name', None) == 'cache_unfinished_req'))
        backup = next((n for n in m.body if isinstance(n, ast.If) and 'chunked' in ast.unparse(n.test) and ('write_through' in ast.unparse(n.test))))
        rotation = next((n for n in m.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'result.rotation_tail_declined'))
        self.assertLess(m.body.index(rotation), m.body.index(backup))
        self.assertTrue(any((isinstance(n, ast.Return) for n in rotation.body)))
        block = compile(ast.fix_missing_locations(ast.Module(body=[backup], type_ignores=[])), 'backup', 'exec')
        for chunked, enabled, policy, node_id, backed, expected in [(True, True, 'write_through', 1, False, 1), (False, True, 'write_through', 1, False, 0), (True, True, 'write_through_selective', 1, False, 0), (True, True, 'write_through', 0, False, 0), (True, True, 'write_through', 1, True, 0)]:
            actions = []
            cache = NS(tree_core=NS(enable_hicache=enabled, is_root=lambda x: x == 0, node_by_id=lambda x: NS(backuped=backed), _build_backup_kv_action=lambda x: 'backup'), cache_controller=NS(write_policy=policy), _apply_cache_actions=actions.extend)
            exec(block, dict(self=cache, chunked=chunked, result=NS(last_device_node=node_id)))
            self.assertEqual(len(actions), expected)
if __name__ == '__main__':
    unittest.main(verbosity=2)
