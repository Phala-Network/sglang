import hashlib,json,pathlib,py_compile
root=pathlib.Path('/tmp/kimi-model-cache')
expected=json.loads((root/'base-files.json').read_text())
for name,digest in expected.items():
    target=pathlib.Path('/sgl-workspace/sglang')/name
    assert hashlib.sha256(target.read_bytes()).hexdigest()==digest, 'Unexpected base file: '+name
    body=(root/'runtime'/name).read_bytes()
    compile(body,str(target),'exec')
    target.write_bytes(body)
    py_compile.compile(str(target),doraise=True)
print('Installed three guarded checkpoint-cache source files')
