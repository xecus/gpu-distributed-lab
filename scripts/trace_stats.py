"""CUDA NCCL kernel interval union: communication duration, not exposed stall time."""
import argparse,json,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('--out',default='runs/trace_stats.csv');a=p.parse_args()
data=json.loads(Path(a.trace).read_text())
def union_us(intervals):
 total=0;start=end=None
 for l,r in sorted(intervals):
  if start is None:start,end=l,r
  elif l<=end:end=max(end,r)
  else:total+=end-start;start,end=l,r
 return total+(0 if start is None else end-start)
kernels=[e for e in data['traceEvents'] if e.get('ph')=='X' and 'kernel' in str(e.get('cat','')).lower()]
comm=[e for e in kernels if 'nccl' in e.get('name','').lower()]
ints=lambda events:[(e['ts'],e['ts']+e['dur']) for e in events if 'dur' in e]
row={'gpu_kernel_union_ms':union_us(ints(kernels))/1000,'nccl_kernel_union_ms':union_us(ints(comm))/1000,
 'nccl_kernel_count':len(comm),'gpu_kernel_count':len(kernels),'source':a.trace,'note':'NCCL overlaps compute; do not subtract this from step time. Inspect timeline for exposed stalls.'}
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
print(row)
