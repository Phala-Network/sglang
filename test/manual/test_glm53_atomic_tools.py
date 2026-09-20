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

class ParserTests(unittest.TestCase):

    def test_atomic_truncation_and_chunk_sizes(self):
        ts = [tool('f', {'city': {'type': 'string'}})]
        text = call('f', [('city', 'Paris')])
        for size in (1, 7, len(text)):
            d = Detector()
            out = []
            seen = ''
            for i in range(0, len(text), size):
                chunk = text[i:i + size]
                seen += chunk
                r = d.parse_streaming_increment(chunk, ts)
                if '</tool_call>' not in seen:
                    self.assertFalse(r.calls)
                out += r.calls
            self.assertEqual(len(out), 1)
            self.assertEqual(json.loads(out[0].parameters), {'city': 'Paris'})
        self.assertFalse(Detector().parse_streaming_increment(text[:-5], ts).calls)

    def test_const_boolean_null_numeric_and_nested(self):
        vals = {'b': ({'const': True}, 'True', True), 'n': ({'const': None}, 'None', None), 'i': ({'const': 7}, '7', 7), 'a': ({'type': 'array'}, '[1,{"ok":true}]', [1, {'ok': True}]), 's': ({'const': '7'}, '7', '7')}
        ts = [tool('f', {k: v[0] for k, v in vals.items()})]
        text = call('f', [(k, v[1]) for k, v in vals.items()])
        result = Detector().parse_streaming_increment(text, ts).calls
        self.assertEqual(json.loads(result[0].parameters), {k: v[2] for k, v in vals.items()})

    def test_union_string_vs_integer(self):
        ts = [tool('f', {'value': {'oneOf': [{'const': '7'}, {'const': 8}]}})]
        for raw, expected in [('7', '7'), ('8', 8)]:
            self.assertEqual(json.loads(Detector().parse_streaming_increment(call('f', [('value', raw)]), ts).calls[0].parameters), {'value': expected})

    def test_multiple_and_truncated_tail(self):
        ts = [tool('f', {})]
        r = Detector().parse_streaming_increment(call('f') + call('f') + '<tool_call>f', ts)
        self.assertEqual([c.tool_index for c in r.calls], [0, 1])

    def test_unknown_and_normal_text(self):
        ts = [tool('f', {})]
        r = Detector().parse_streaming_increment('before ' + call('unknown') + call('f') + ' after', ts)
        self.assertEqual(r.normal_text, 'before  after')
        self.assertEqual(len(r.calls), 1)
        self.assertEqual(r.calls[0].tool_index, 0)

    def test_root_union_discriminator(self):
        t = tool('f', {})
        t.function.parameters = {'oneOf': [{'properties': {'kind': {'const': 'number'}, 'value': {'type': 'integer'}}}, {'properties': {'kind': {'const': 'text'}, 'value': {'type': 'string'}}}]}
        for kind, expected in [('number', 7), ('text', '7')]:
            r = Detector().parse_streaming_increment(call('f', [('value', '7'), ('kind', kind)]), [t])
            self.assertEqual(json.loads(r.calls[0].parameters)['value'], expected)
if __name__ == '__main__':
    unittest.main(verbosity=2)
