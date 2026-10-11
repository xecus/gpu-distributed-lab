"""Shared measurable training loop for the toy Transformer and Qwen experiments.

One script covers every parallel mode so that only the parallelism changes between lessons:
  single  one process
  manual  replicated model, gradients averaged by an explicit all_reduce (A06)
  ddp     torch DistributedDataParallel (A07, A13, A14)
  zero1/2/3  DeepSpeed ZeRO stages (A08-A10, B01-B03)
  fsdp    PyTorch FSDP1 with per-block wrapping (A11, B03)
  tp      tensor parallel attention + MLP on the toy model (A12)

Writes per-rank metrics CSV, environment JSON and resumable checkpoints. Never calls AWS APIs.
"""
import argparse,contextlib,csv,json,os,time,platform,subprocess
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn import functional as F
from lab.model import Transformer,Block,apply_tp

def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['single','manual','ddp','zero1','zero2','zero3','fsdp','tp'],default='single')
    p.add_argument('--model',default='toy',help='toy or Hugging Face repo id')
    p.add_argument('--revision',default='main',help='pin to HF commit for reproducibility')
    p.add_argument('--lora',action='store_true')
    p.add_argument('--tokens',help='uint32 token memmap produced by lab.prepare_data')
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--accum',type=int,default=1)
    p.add_argument('--seq',type=int,default=256)
    p.add_argument('--dim',type=int,default=256)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--heads',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--out',default='runs/debug')
    p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--keep-checkpoints',type=int,default=2)
    p.add_argument('--resume',help='step_N checkpoint directory (same world size/mode)')
    p.add_argument('--activation-checkpointing',action='store_true')
    p.add_argument('--profile',action='store_true',help='one step trace, excluded from timing')
    p.add_argument('--sampler',action='store_true',help='use DistributedSampler over fixed indexed corpus windows')
    p.add_argument('--export-hf',help='FSDP resume + gather to HF directory, without training')
    p.add_argument('--cpu',action='store_true',help='CPU smoke tests; single/manual/ddp only')
    return p

def main():
    a=parser().parse_args()
    if a.steps<=0 or a.warmup<0 or min(a.batch,a.accum,a.seq)<1:raise ValueError('invalid sizes')
    # torchrun sets RANK/WORLD_SIZE/LOCAL_RANK; defaults allow plain `python -m lab.train`.
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if a.mode=='single' and world!=1:raise ValueError('single requires 1 process')
    if a.cpu and a.mode not in ['single','manual','ddp']:raise ValueError('GPU required for this mode')
    device=torch.device('cpu' if a.cpu else f'cuda:{local}')
    if not a.cpu:torch.cuda.set_device(device)
    if world>1 or a.mode.startswith('zero') or a.mode=='fsdp':
        dist.init_process_group('gloo' if a.cpu else 'nccl')
    active=dist.is_initialized()
    torch.manual_seed(a.seed)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ds_config=None;hf_ds=None
    # ZeRO: DeepSpeed owns the optimizer, BF16 and gradient accumulation.
    if a.mode.startswith('zero'):
        ds_config={'train_micro_batch_size_per_gpu':a.batch,'gradient_accumulation_steps':a.accum,
          'train_batch_size':a.batch*a.accum*world,'bf16':{'enabled':True},
          'zero_optimization':{'stage':int(a.mode[-1])},'steps_per_print':1000000,
          'optimizer':{'type':'AdamW','params':{'lr':a.lr,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0.01}}}
        if a.model!='toy' and a.mode=='zero3':
            from transformers.integrations import HfDeepSpeedConfig
            hf_ds=HfDeepSpeedConfig(ds_config) # Keep alive during from_pretrained.
    if a.model=='toy':
        model=Transformer(dim=a.dim,layers=a.layers,heads=a.heads,seq=a.seq)
    else:
        if not a.tokens:raise ValueError('real models require --tokens, synthetic ids are not a corpus')
        from transformers import AutoModelForCausalLM,AutoConfig
        # FSDP: only rank0 loads real weights; other ranks build on 'meta' and receive them
        # through sync_module_states, so host RAM holds one copy instead of eight.
        if a.mode=='fsdp' and rank!=0:
            config=AutoConfig.from_pretrained(a.model,revision=a.revision)
            with torch.device('meta'):model=AutoModelForCausalLM.from_config(config,attn_implementation='sdpa')
        else:
            model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,torch_dtype=torch.float32,
                                                       attn_implementation='sdpa')
        if a.lora:
            from peft import LoraConfig,get_peft_model
            model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,task_type='CAUSAL_LM',
                    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        if a.activation_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model.config.use_cache=False
    if a.model=='toy' and a.activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
        for i,b in enumerate(model.blocks):model.blocks[i]=checkpoint_wrapper(b)
    params=sum(p.numel() for p in model.parameters())
    # FSDP must wrap before optimizer creation. ZeRO handles optimizer itself.
    if a.mode=='fsdp':
        from functools import partial
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        wrap_class=Block
        if a.model!='toy':
            from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
            wrap_class=Qwen3DecoderLayer
        model=FSDP(model,auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls={wrap_class}),
            sync_module_states=True,param_init_fn=lambda module:module.to_empty(device=device,recurse=False),device_id=device,mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.bfloat16,buffer_dtype=torch.bfloat16))
    elif not a.mode.startswith('zero'):model.to(device)
    if a.mode=='tp':
        if world<2:raise ValueError('TP requires >=2 GPU processes')
        if a.model!='toy':raise ValueError('TP lesson supports the toy model only')
        if (4*a.dim)%world:raise ValueError('FFN dim must divide world size')
        model=apply_tp(model,world)
    if a.mode=='ddp':
        from torch.nn.parallel import DistributedDataParallel as DDP
        model=DDP(model,device_ids=None if a.cpu else [local])
    engine=None
    if a.mode.startswith('zero'):
        import deepspeed
        engine,opt,_,_=deepspeed.initialize(model=model,model_parameters=[p for p in model.parameters() if p.requires_grad],config=ds_config)
        model=engine
    else:opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr)
    if a.mode=='manual' and active:
        for p in model.parameters():dist.broadcast(p.data,0)
    tokens=None
    if a.tokens:
        import numpy as np
        tokens=np.memmap(a.tokens,dtype=np.uint32,mode='r')
        if len(tokens)<a.seq+2:raise ValueError('token corpus too small')
    data_fingerprint=None
    if a.tokens:
        import hashlib
        h=hashlib.sha256()
        with open(a.tokens,'rb') as file:
            for chunk in iter(lambda:file.read(8*1024*1024),b''):h.update(chunk)
        data_fingerprint=h.hexdigest()
    sampler_cache={}
    # Batches are a pure function of (seed, step, micro, rank), so resume reproduces the exact
    # same data stream without saving any dataloader state.
    def batch(step,micro):
        # TP workers see the SAME batch; DP workers see distinct rank streams.
        data_rank=0 if a.mode=='tp' else rank
        if a.sampler:
            from torch.utils.data.distributed import DistributedSampler
            size=65536;replicas=1 if a.mode=='tp' else world
            sampler=DistributedSampler(range(size),num_replicas=replicas,rank=data_rank,seed=a.seed,drop_last=True)
            per_rank=len(sampler);offset=(step*a.accum+micro)*a.batch;epoch=offset//per_rank
            if epoch not in sampler_cache:
                sampler.set_epoch(epoch);sampler_cache.clear();sampler_cache[epoch]=list(sampler)
            indices=[sampler_cache[epoch][(offset+i)%per_rank] for i in range(a.batch)]
            if tokens is None:
                starts=torch.tensor([((i*2654435761+a.seed)%256) for i in indices])[:,None]
                x=(starts+torch.arange(a.seq+1))%256
            else:
                import numpy as np
                positions=[(i*2654435761+a.seed)%(len(tokens)-a.seq-1) for i in indices]
                x=torch.tensor(np.stack([tokens[i:i+a.seq+1] for i in positions]).astype('int64'))
            x=x.to(device);return x[:,:-1],x[:,1:]
        g=torch.Generator().manual_seed(a.seed+step*100003+micro*997+data_rank*10000019)
        if tokens is None:
            starts=torch.randint(0,256,(a.batch,1),generator=g)
            x=(starts+torch.arange(a.seq+1))%256
        else:
            positions=torch.randint(0,len(tokens)-a.seq-1,(a.batch,),generator=g).tolist()
            import numpy as np
            x=torch.tensor(np.stack([tokens[i:i+a.seq+1] for i in positions]).astype('int64'))
        x=x.to(device);return x[:,:-1],x[:,1:]
    def state_context():
        if a.mode=='fsdp':
            from torch.distributed.fsdp import StateDictType,LocalStateDictConfig,FullyShardedDataParallel as FSDP
            return FSDP.state_dict_type(model,StateDictType.LOCAL_STATE_DICT,LocalStateDictConfig(offload_to_cpu=True))
        return contextlib.nullcontext()
    def bare():return model.module if a.mode=='ddp' else model # unwrap DDP: no 'module.' key prefix
    # Every rank writes its own shard (rank{r}.pt or DeepSpeed files) plus meta_rank{r}.json.
    def checkpoint(step):
        dest=out/f'step_{step}';dest.mkdir(exist_ok=True)
        meta={'step':step,'world':world,'mode':a.mode,'seed':a.seed,'args':vars(a),'data_sha256':data_fingerprint}
        if engine:
            engine.save_checkpoint(str(dest),tag='ds',client_state=meta)
        else:
            with state_context():sd=bare().state_dict()
            torch.save({'model':sd,'optimizer':opt.state_dict(),'meta':meta,
                 'rng':torch.get_rng_state(),'cuda_rng':None if a.cpu else torch.cuda.get_rng_state()},dest/f'rank{rank}.pt')
        (dest/f'meta_rank{rank}.json').write_text(json.dumps(meta,indent=2))
        if active:dist.barrier()
        if local==0 and a.keep_checkpoints>0:
            import shutil
            dirs=sorted([d for d in out.glob('step_*') if d.is_dir() and d.name[5:].isdigit()],key=lambda d:int(d.name[5:]))
            for old in dirs[:-a.keep_checkpoints]:shutil.rmtree(old,ignore_errors=True)
        if active:dist.barrier()
    start=0
    # Resume refuses to continue if world size, mode, data or any shape-affecting arg changed.
    if a.resume:
        meta=json.loads((Path(a.resume)/f'meta_rank{rank}.json').read_text())
        if (meta['world'],meta['mode'])!=(world,a.mode):raise ValueError('resume topology/mode mismatch')
        if meta.get('data_sha256')!=data_fingerprint:raise ValueError('token corpus checksum changed')
        old=meta['args']
        for key in ['model','revision','lora','tokens','batch','accum','seq','dim','layers','heads','seed','activation_checkpointing','sampler']:
            if old.get(key)!=vars(a).get(key):raise ValueError(f'resume setting differs: {key}')
        if engine:
            loaded,_=engine.load_checkpoint(a.resume,tag='ds')
            if loaded is None:raise RuntimeError('DeepSpeed checkpoint not found')
        else:
            sd=torch.load(Path(a.resume)/f'rank{rank}.pt',map_location='cpu',weights_only=False)
            with state_context():bare().load_state_dict(sd['model'])
            opt.load_state_dict(sd['optimizer']);torch.set_rng_state(sd['rng'])
            if not a.cpu:torch.cuda.set_rng_state(sd['cuda_rng'])
        start=meta['step']
    if a.export_hf:
        if a.mode!='fsdp' or not a.resume or a.model=='toy':raise ValueError('export-hf requires real-model FSDP resume')
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig
        with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
            state=model.state_dict()
        if rank==0:
            model.module.save_pretrained(a.export_hf,state_dict=state)
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(a.model,revision=a.revision).save_pretrained(a.export_hf)
        if active:dist.barrier();dist.destroy_process_group()
        return
    def sync():
        if not a.cpu:torch.cuda.synchronize(device)
    # One optimizer update = accum micro-batches. DDP skips gradient all-reduce (no_sync) on all
    # but the last micro-batch; manual mode averages gradients explicitly after accumulation.
    def update(step):
        if not engine:opt.zero_grad(set_to_none=True)
        losses=[]
        for micro in range(a.accum):
            x,y=batch(step,micro)
            no_sync=model.no_sync() if a.mode=='ddp' and micro<a.accum-1 else contextlib.nullcontext()
            with no_sync:
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=not a.cpu):
                    result=model(x) if a.model=='toy' else model(input_ids=x)
                    logits=result if a.model=='toy' else result.logits
                    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1))
                losses.append(loss.detach().float())
                if engine:engine.backward(loss);engine.step()
                else:(loss/a.accum).backward()
        if not engine:
            if a.mode=='manual' and active:
                for p in model.parameters():
                    if p.grad is not None:dist.all_reduce(p.grad);p.grad.div_(world)
            opt.step()
        return torch.stack(losses).mean()
    # Warmup updates optimizer; steps denote all updates including warmup.
    meta={'args':vars(a),'world':world,'rank':rank,'host':platform.node(),'torch':torch.__version__,
          'cuda':torch.version.cuda,'data_sha256':data_fingerprint,'parameter_count':params,'gpu':None if a.cpu else torch.cuda.get_device_name(device)}
    try:meta['git']=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):meta['git']=None
    (out/f'environment_rank{rank}.json').write_text(json.dumps(meta,indent=2))
    fields=['step','rank','world','mode','global_batch','loss','step_s','tokens_s','allocated_gib','reserved_gib']
    # Timing: barrier + synchronize before, synchronize after, max over ranks = step time.
    with (out/f'metrics_rank{rank}_from{start}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for step in range(start,a.steps):
            if active:dist.barrier()
            sync()
            if not a.cpu:torch.cuda.reset_peak_memory_stats(device)
            t=time.perf_counter();loss=update(step);sync();elapsed=time.perf_counter()-t
            # Synchronization and loss reduction excluded from step time; max across ranks below.
            stats=torch.tensor([elapsed],device=device)
            if active:dist.all_reduce(stats,op=dist.ReduceOp.MAX)
            if active and a.mode!='tp':dist.all_reduce(loss);loss/=world
            gb=a.batch*a.accum*(1 if a.mode=='tp' else world)
            if step>=a.warmup:
                writer.writerow(dict(step=step+1,rank=rank,world=world,mode=a.mode,global_batch=gb,loss=loss.item(),
                  step_s=stats.item(),tokens_s=gb*a.seq/stats.item(),
                  allocated_gib=0 if a.cpu else torch.cuda.max_memory_allocated(device)/2**30,
                  reserved_gib=0 if a.cpu else torch.cuda.max_memory_reserved(device)/2**30));f.flush()
            if rank==0 and (step+1)%10==0:print(f'step={step+1} loss={loss.item():.4f} seconds={stats.item():.4f}',flush=True)
            if a.save_every>0 and (step+1)%a.save_every==0:checkpoint(step+1)
        if a.steps>start and (a.save_every<=0 or a.steps%a.save_every):checkpoint(a.steps)
    # Profile one extra update after the timed loop so tracing overhead never enters step_s.
    if a.profile:
        acts=[torch.profiler.ProfilerActivity.CPU]
        if not a.cpu:acts.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=acts,record_shapes=True,profile_memory=True) as prof:update(a.steps)
        prof.export_chrome_trace(str(out/f'trace_rank{rank}.json'))
    if active:dist.destroy_process_group()
if __name__=='__main__':main()
