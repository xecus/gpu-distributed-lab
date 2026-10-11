"""A01: record device properties and time FP32 vs BF16 square matmul with CUDA events.

TF32 is disabled so FP32 really runs in FP32. TFLOPs uses 2*n^3 FLOPs per matmul.
"""
import argparse,csv,time,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=4096);p.add_argument('--iters',type=int,default=30)
p.add_argument('--out',default='runs/A01');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
prop=torch.cuda.get_device_properties(0)
(out/'device.json').write_text(json.dumps({'name':prop.name,'SMs':prop.multi_processor_count,
 'HBM_gib':prop.total_memory/2**30,'compute_capability':[prop.major,prop.minor]},indent=2))
rows=[]
for dtype in [torch.float32,torch.bfloat16]:
 torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
 x=torch.randn(a.size,a.size,device='cuda',dtype=dtype);y=torch.randn_like(x)
 for _ in range(5):z=x@y # warmup (cuBLAS heuristics, clocks)
 # CUDA events time the GPU stream itself; host time.time() would only measure kernel launch.
 start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 start.record()
 for _ in range(a.iters):z=x@y
 end.record();torch.cuda.synchronize();ms=start.elapsed_time(end)/a.iters
 rows.append({'dtype':str(dtype),'size':a.size,'matmul_ms':ms,'TFLOPs':2*a.size**3/(ms/1000)/1e12,
   'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
 del x,y,z
with (out/'matmul.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(rows)
