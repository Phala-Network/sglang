"""One-shot exact-GGUF slice numerical check; no timings or serving load.
Run only on an idle authorized GPU in an isolated container.
"""
import argparse,ast,json,pathlib
import gguf,numpy as np,torch
import sglang.srt.layers.quantization.gguf as installed
torch.backends.cuda.matmul.allow_tf32=False
p=argparse.ArgumentParser();p.add_argument("--gguf",required=True);p.add_argument("--source",required=True);a=p.parse_args()
source=pathlib.Path(a.source);tree=ast.parse(source.read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="fused_mul_mat_gguf" or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {"DEQUANT_MIN_ROWS","DEQUANT_MIN_WORK"} for t in n.targets)]
ns=dict(vars(installed));exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),ns);candidate=ns["fused_mul_mat_gguf"]
reader=gguf.GGUFReader(a.gguf);tensor=next(t for t in reader.tensors if t.name.startswith("blk.0.") and "ffn_gate.weight" in t.name and int(t.tensor_type)==8)
assert int(tensor.shape[0])==5120 and int(tensor.shape[1])>=512
packed=np.array(tensor.data[:512],copy=True);w=torch.from_numpy(packed).to("cuda")
assert w.ndim==2 and w.shape[0]==512
rng=torch.Generator(device="cuda");rng.manual_seed(8948)
x=torch.randn((1024,5120),generator=rng,device="cuda",dtype=torch.bfloat16)
# Dispatch uses actual installed kernels; source candidate does not modify installed module.
calls=[];orig_dq=ns["dequantize_gguf_weight"];orig_vec=ns["ggml_mul_mat_vec_a8"]
def dq(*args):calls.append("dequant");return orig_dq(*args)
def vec(*args):calls.append("mmvq");return orig_vec(*args)
ns["dequantize_gguf_weight"]=dq;ns["ggml_mul_mat_vec_a8"]=vec
with torch.inference_mode():
 y=candidate(x,w,8);assert calls==["dequant"],calls
 baseline=installed.ggml_mul_mat_a8(w,x,8,w.shape[0])
 block,byte=gguf.GGML_QUANT_SIZES[8]
 wf=installed.ggml_dequantize(w,8,w.shape[0],w.shape[1]//byte*block,torch.float32)
 reference=x.float()@wf.T
 def metrics(v):
  delta=v.float()-reference
  assert torch.isfinite(v).all()
  return {"max_abs":delta.abs().max().item(),"relative_l2":(delta.norm()/reference.norm()).item()}
 result={"dense":metrics(y),"mmq":metrics(baseline)}
 # Broad screen against gross corruption, not full-model accuracy acceptance.
 assert result["dense"]["relative_l2"]<0.02 and result["mmq"]["relative_l2"]<0.02,result
 calls.clear();one=candidate(x[:1],w,8);expected=orig_vec(w,x[:1],8,w.shape[0]);assert calls==["mmvq"] and torch.equal(one,expected)
 result.update({"tensor":tensor.name,"qtype":8,"slice_weight_shape":[512,5120],"input_shape":[1024,5120],"single_token_original_mmvq_bit_equal":True,"scope":"single real-weight slice numerical functional check; no timing or full-model qualification"})
 print(json.dumps(result,indent=2))
