from pathlib import Path
import hashlib,json,shutil,py_compile
root=Path('/sgl-workspace/sglang'); manifest=json.loads(Path('/opt/phala-control/source-manifest.json').read_text())
for rel,entry in manifest.items():
 p=root/rel; expected=entry['base_sha256']; assert (hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None)==expected, rel
for rel,entry in manifest.items():
 p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path('/opt/phala-control/patched')/rel,p);assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['final_sha256']
 if p.suffix=='.py':py_compile.compile(str(p),doraise=True)
print('PASS guarded source installation',len(manifest))

for rel,expected in json.loads(Path("/opt/phala-control/privacy-source-binding.json").read_text()).items():
 assert hashlib.sha256((root/"python/sglang/srt"/rel).read_bytes()).hexdigest()==expected,rel
print("PASS final privacy source binding")
