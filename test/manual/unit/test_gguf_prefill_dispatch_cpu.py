"""Execute actual source dispatch with sentinel kernel calls; no tensor/kernel correctness claim."""
import ast,json,pathlib
class Tensor:
 def __init__(self,shape):self.shape=shape;self.dtype="bf16";self.device="cpu"
 @property
 def T(self):return self
 def __matmul__(self,other):return "dense"
class Torch:
 @staticmethod
 def empty(*a,**kw):return "empty"
def load(path):
 tree=ast.parse(path.read_text(encoding="utf-8"));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="fused_mul_mat_gguf")
 ns={"torch":Torch,"IMATRIX_QUANT_TYPES":{16},"UNQUANTIZED_TYPES":{0,1,30},"MMVQ_QUANT_TYPES":{8,12,16},"MMQ_QUANT_TYPES":{8,12},"DEQUANT_TYPES":{8,12,16},"ggml_mul_mat_vec_a8":lambda *a:"mmvq","ggml_mul_mat_a8":lambda *a:"mmq","dequantize_gguf_weight":lambda *a:Tensor((1,1)),"WeightType":lambda x:x}
 node.returns=None
 for a in node.args.args:a.annotation=None
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {"DEQUANT_MIN_ROWS","DEQUANT_MIN_WORK"} for t in n.targets):
   exec(compile(ast.Module(body=[n],type_ignores=[]),str(path),"exec"),ns)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),"exec"),ns);return ns[node.name]
p=pathlib.Path(__file__).resolve().parents[3]/"python/sglang/srt/layers/quantization/gguf.py";c=load(p)
cases=[
 (0,17408,5120,8,"empty"),
 (1,17408,5120,8,"mmvq"),
 (2,248320,5120,8,"mmvq"),
 (47,17408,5120,8,"mmq"),
 (48,17408,5120,8,"mmq"),
 (511,1024,1024,8,"mmq"),
 (512,1024,1024,8,"mmq"),
 # Row threshold, with work already above the minimum on both sides.
 (1023,17408,5120,8,"mmq"),
 (1024,17408,5120,8,"dense"),
 # Work immediately below and equal to 2**29, with rows at the threshold.
 (1024,1023,512,8,"mmq"),
 (1024,1024,512,8,"dense"),
 (16384,17408,5120,8,"dense"),
 (16384,17408,5120,30,"dense"),
 (16,1024,1024,16,"mmvq"),
 (17,1024,1024,16,"dense"),
 (48,17408,5120,12,"mmq"),
]
results=[]
for rows,out,inner,q,expected_c in cases:
 x,w=Tensor((rows,inner)),Tensor((out,inner));actual_c=c(x,w,q)
 assert actual_c==expected_c,(rows,out,inner,q,actual_c)
 results.append({"rows":rows,"out":out,"in":inner,"qtype":q,"candidate":actual_c})
print(json.dumps({"passed":len(results),"scope":"actual-source CPU dispatch only, no numeric or performance proof","cases":results},indent=2))
