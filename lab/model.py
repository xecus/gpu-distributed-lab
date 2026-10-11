"""Causal Transformer; tensor parallel attention + MLP, replicated embedding/head."""
import torch
from torch import nn
from torch.nn import functional as F
class Block(nn.Module):
 """Pre-LN block: causal self-attention + 4x GELU MLP. No biases so TP sharding stays simple."""
 def __init__(self,dim,heads):
  super().__init__();self.heads=heads
  self.norm1=nn.LayerNorm(dim);self.norm2=nn.LayerNorm(dim)
  self.q=nn.Linear(dim,dim,bias=False);self.k=nn.Linear(dim,dim,bias=False);self.v=nn.Linear(dim,dim,bias=False)
  self.proj=nn.Linear(dim,dim,bias=False);self.fc1=nn.Linear(dim,4*dim,bias=False);self.fc2=nn.Linear(4*dim,dim,bias=False)
 def forward(self,x):
  b,s,d=x.shape;z=self.norm1(x)
  q,k,v=[layer(z).view(b,s,self.heads,-1).transpose(1,2) for layer in (self.q,self.k,self.v)]
  a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
  x=x+self.proj(a.transpose(1,2).reshape(b,s,-1))
  return x+self.fc2(F.gelu(self.fc1(self.norm2(x))))
class Transformer(nn.Module):
 """Byte-level (vocab 256) decoder with learned positional embeddings."""
 def __init__(self,vocab=256,dim=256,layers=4,heads=8,seq=256):
  super().__init__();assert dim%heads==0
  self.tok=nn.Embedding(vocab,dim);self.pos=nn.Embedding(seq,dim)
  self.blocks=nn.ModuleList([Block(dim,heads) for _ in range(layers)])
  self.norm=nn.LayerNorm(dim);self.out=nn.Linear(dim,vocab,bias=False)
 def forward(self,ids):
  x=self.tok(ids)+self.pos(torch.arange(ids.shape[1],device=ids.device))
  for block in self.blocks:x=block(x)
  return self.out(self.norm(x))
def apply_tp(model,world):
 """Megatron-style TP over one device mesh of size world.

 q/k/v and fc1 are split by output column (each rank owns heads/world heads and 4*dim/world
 hidden units); proj and fc2 are split by input row, so each block needs one AllReduce in
 forward for attention and one for the MLP. Embeddings, norms and the LM head stay replicated.
 """
 from torch.distributed.device_mesh import init_device_mesh
 from torch.distributed.tensor.parallel import parallelize_module,ColwiseParallel,RowwiseParallel
 mesh=init_device_mesh('cuda',(world,))
 for wrapped in model.blocks:
  block=getattr(wrapped,'_checkpoint_wrapped_module',wrapped)
  if block.heads%world:raise ValueError('heads must divide TP size')
  parallelize_module(block,mesh,{'q':ColwiseParallel(),'k':ColwiseParallel(),'v':ColwiseParallel(),
   'proj':RowwiseParallel(),'fc1':ColwiseParallel(),'fc2':RowwiseParallel()})
  block.heads//=world # local q/k/v now hold only this rank's heads
 return model
