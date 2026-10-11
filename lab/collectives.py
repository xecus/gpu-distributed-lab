"""A04: bandwidth/latency of AllReduce, AllGather and ReduceScatter over message sizes.

Each collective is first checked for numerical correctness, then timed. bus_GBs applies the
standard NCCL bus-bandwidth factor so results are comparable across world sizes.
Launch with torchrun (scripts/part_a.sh A04).
"""
import argparse,csv,os,time
from pathlib import Path
import torch
import torch.distributed as dist
p=argparse.ArgumentParser();p.add_argument('--out',default='runs/A04/collectives.csv');p.add_argument('--iters',type=int,default=50)
p.add_argument('--cpu',action='store_true');p.add_argument('--max-power',type=int,default=28)
a=p.parse_args();local=int(os.environ.get('LOCAL_RANK',0))
if not a.cpu:torch.cuda.set_device(local)
dev='cpu' if a.cpu else f'cuda:{local}';dist.init_process_group('gloo' if a.cpu else 'nccl')
rank=dist.get_rank();world=dist.get_world_size()
def sync():
 if not a.cpu:torch.cuda.synchronize()
rows=[]
# Message sizes 2^10 .. 2^max_power bytes (fp32 = 4 bytes), rounded to a multiple of world.
for power in range(10,a.max_power+1,2):
 n=max(world,(2**power//4)//world*world)
 for name in ['all_reduce','all_gather','reduce_scatter']:
  x=torch.full((n,),float(rank+1),device=dev)
  if name=='all_reduce':
   fn=lambda:dist.all_reduce(x);fn();torch.testing.assert_close(x,torch.full_like(x,world*(world+1)/2));size=n*4;factor=2*(world-1)/world
  elif name=='all_gather':
   y=torch.empty(n*world,device=dev);fn=lambda:dist.all_gather_into_tensor(y,x);fn()
   for r,chunk in enumerate(y.chunk(world)):torch.testing.assert_close(chunk,torch.full_like(chunk,r+1))
   size=n*4*world;factor=(world-1)/world
  else:
   if a.cpu:continue # Gloo availability varies; NCCL implements reduce_scatter_tensor.
   y=torch.empty(n//world,device=dev);fn=lambda:dist.reduce_scatter_tensor(y,x);fn()
   torch.testing.assert_close(y,torch.full_like(y,world*(world+1)/2));size=n*4;factor=(world-1)/world
  x.zero_()
  for _ in range(5):fn() # warmup: NCCL communicator setup and lazy allocation
  sync();dist.barrier();t=time.perf_counter()
  for _ in range(a.iters):fn()
  # Report the slowest rank; a collective is only done when every rank is done.
  sync();dt=torch.tensor([(time.perf_counter()-t)/a.iters],device=dev);dist.all_reduce(dt,op=dist.ReduceOp.MAX);secs=dt.item()
  rows.append({'collective':name,'bytes':size,'world':world,'latency_ms':secs*1000,'algorithm_GBs':size/secs/1e9,'bus_GBs':size/secs/1e9*factor})
if rank==0:
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(rows)
dist.destroy_process_group()
