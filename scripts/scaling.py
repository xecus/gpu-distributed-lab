"""Strong/weak scaling efficiency and approximate MFU from several summary.json files.

MFU uses the 6 * params * tokens/s training FLOPs estimate; pass the dense (non-sparse) BF16 peak of
the GPU you actually used, since it differs between H100, H200 and B200.
"""
import argparse,json,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--gpus',nargs='+',type=int,required=True)
p.add_argument('--kind',choices=['strong','weak'],required=True);p.add_argument('--params',type=int)
p.add_argument('--peak-bf16-tflops-per-gpu',type=float,help='non-sparse peak from the actual GPU specification')
p.add_argument('--out',default='runs/scaling.csv');a=p.parse_args()
if len(a.runs)!=len(a.gpus):raise ValueError('run/GPU count mismatch')
items=[]
for path,gpu in zip(a.runs,a.gpus):
 s=json.loads(Path(path,'summary.json').read_text());items.append((gpu,s))
basegpu,bases=items[0];rows=[]
for gpu,s in items:
 ratio=gpu/basegpu;speed=bases['median_step_s']/s['median_step_s']
 efficiency=speed/ratio if a.kind=='strong' else s['mean_tokens_s']/bases['mean_tokens_s']/ratio
 mfu=None
 if a.params and a.peak_bf16_tflops_per_gpu:mfu=6*a.params*s['mean_tokens_s']/(gpu*a.peak_bf16_tflops_per_gpu*1e12)
 rows.append({'gpus':gpu,'kind':a.kind,'step_speedup':speed,'scaling_efficiency':efficiency,
  'approx_MFU':mfu,'tokens_s':s['mean_tokens_s'],'usd_per_million_tokens':s['steady_usd_per_million_tokens']})
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(rows)
