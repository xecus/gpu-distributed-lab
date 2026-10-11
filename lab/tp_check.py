"""Check attention+MLP TP forward, gradients and SGD update vs unsharded model."""
import os,copy
import torch
import torch.distributed as dist
from lab.model import Transformer,apply_tp
local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);dist.init_process_group('nccl')
world=dist.get_world_size();torch.manual_seed(42)
ref=Transformer(dim=64,layers=2,heads=8,seq=16).cuda();tp=apply_tp(copy.deepcopy(ref),world)
x=torch.randint(0,256,(2,16),device='cuda');y=torch.randint(0,256,(2,16),device='cuda')
loss=lambda m:torch.nn.functional.cross_entropy(m(x).reshape(-1,256),y.reshape(-1))
a=loss(ref);b=loss(tp);torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
a.backward();b.backward()
for (n,p),(name,q) in zip(ref.named_parameters(),tp.named_parameters()):
 assert n==name
 grad=q.grad.full_tensor() if hasattr(q.grad,'full_tensor') else q.grad
 torch.testing.assert_close(p.grad,grad,rtol=1e-3,atol=1e-5)
torch.optim.SGD(ref.parameters(),lr=.01).step();torch.optim.SGD(tp.parameters(),lr=.01).step()
for p,q in zip(ref.parameters(),tp.parameters()):
 full=q.full_tensor() if hasattr(q,'full_tensor') else q
 torch.testing.assert_close(p,full,rtol=1e-4,atol=1e-5)
if dist.get_rank()==0:print('TP attention+MLP parity PASS')
dist.destroy_process_group()
