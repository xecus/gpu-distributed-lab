"""Evaluate a HF model or saved LoRA adapter on held-out token windows."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--revision',default='main')
p.add_argument('--adapter');p.add_argument('--tokens',required=True);p.add_argument('--seq',type=int,default=256)
p.add_argument('--windows',type=int,default=20);p.add_argument('--out',default='runs/eval.json');a=p.parse_args()
m=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,torch_dtype=torch.bfloat16).cuda()
if a.adapter:
    from peft import PeftModel
    m=PeftModel.from_pretrained(m,a.adapter)
m.eval();data=np.memmap(a.tokens,dtype=np.uint32,mode='r');losses=[]
with torch.inference_mode():
 for i in range(min(a.windows,(len(data)-1)//a.seq)):
    ids=torch.tensor(data[i*a.seq:i*a.seq+a.seq+1].astype('int64'),device='cuda')[None]
    logits=m(input_ids=ids[:,:-1]).logits
    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1))
    losses.append(loss.item())
if not losses:raise ValueError('not enough validation tokens')
value=sum(losses)/len(losses);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'loss':value,'perplexity':math.exp(min(value,80)),'windows':len(losses),'seq':a.seq},indent=2))
