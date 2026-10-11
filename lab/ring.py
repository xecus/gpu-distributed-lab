"""Educational P2P ring ReduceScatter + AllGather, not production optimized."""
import os,argparse,csv,time
from pathlib import Path
import torch
import torch.distributed as dist

def ring_allreduce(x):
 """In-place ring AllReduce (sum) = ReduceScatter phase + AllGather phase.

 x is split into world chunks. In world-1 ReduceScatter steps each rank sends one chunk to its
 right neighbour and adds the chunk received from its left; afterwards rank r owns the fully
 reduced chunk (r+1)%world. World-1 AllGather steps then circulate the reduced chunks.
 Each rank sends 2*(world-1)/world of the data, which is why bus_GBs uses that factor.
 """
 rank=dist.get_rank();world=dist.get_world_size()
 if x.numel()%world:raise ValueError('tensor size must divide world size')
 chunks=list(x.chunk(world));left=(rank-1)%world;right=(rank+1)%world
 def exchange(send_index,recv_index,add):
  # Separate receive buffer; paired nonblocking send/recv avoids blocking ring deadlock.
  recv=torch.empty_like(chunks[recv_index]);send=chunks[send_index].contiguous()
  works=dist.batch_isend_irecv([dist.P2POp(dist.isend,send,right),dist.P2POp(dist.irecv,recv,left)])
  for work in works:work.wait()
  if add:chunks[recv_index].add_(recv)
  else:chunks[recv_index].copy_(recv)
 for i in range(world-1):exchange((rank-i)%world,(rank-i-1)%world,True) # ReduceScatter
 for i in range(world-1):exchange((rank-i+1)%world,(rank-i)%world,False) # AllGather
 return x

def main():
 p=argparse.ArgumentParser();p.add_argument('--elements',type=int,default=1048576);p.add_argument('--iters',type=int,default=20)
 p.add_argument('--cpu',action='store_true');p.add_argument('--out',default='runs/A05/ring.csv');a=p.parse_args()
 local=int(os.environ.get('LOCAL_RANK',0))
 if not a.cpu:torch.cuda.set_device(local)
 dev='cpu' if a.cpu else f'cuda:{local}';dist.init_process_group('gloo' if a.cpu else 'nccl')
 rank=dist.get_rank();world=dist.get_world_size();n=a.elements//world*world
 # For NCCL, initialize collective group before P2P.
 dist.barrier();g=torch.Generator().manual_seed(rank+123);original=torch.randn(n,generator=g).to(dev)
 # Verify against the library AllReduce before timing anything.
 expected=original.clone();dist.all_reduce(expected);actual=ring_allreduce(original.clone())
 torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
 rows=[]
 def sync():
  if not a.cpu:torch.cuda.synchronize()
 for name,fn in [('ring',ring_allreduce),('nccl_or_gloo',dist.all_reduce)]:
  samples=[]
  for i in range(a.iters+5):
   x=original.clone();dist.barrier();sync();t=time.perf_counter();fn(x);sync()
   dt=torch.tensor([time.perf_counter()-t],device=dev);dist.all_reduce(dt,op=dist.ReduceOp.MAX)
   if i>=5:samples.append(dt.item())
  secs=sum(samples)/len(samples);size=n*4
  rows.append({'implementation':name,'world':world,'bytes':size,'latency_ms':secs*1000,
       'algorithm_GBs':size/secs/1e9,'bus_GBs':size/secs/1e9*2*(world-1)/world})
 if rank==0:
  out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
  with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 dist.destroy_process_group()
if __name__=='__main__':main()
