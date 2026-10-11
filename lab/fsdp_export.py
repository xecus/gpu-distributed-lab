"""All-rank FSDP checkpoint to Hugging Face directory; invoke via torchrun."""
import argparse,json,sys,runpy,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
rank=int(os.environ.get('RANK',0));meta=json.loads((Path(a.checkpoint)/f'meta_rank{rank}.json').read_text());old=meta['args']
cmd=['lab.train','--mode','fsdp','--resume',a.checkpoint,'--export-hf',a.out,'--out',str(Path(a.out).parent/'export-state')]
for key in ['model','revision','tokens','batch','accum','seq','dim','layers','heads','seed','lr']:
 if old.get(key) is not None:cmd+=['--'+key.replace('_','-'),str(old[key])]
for key in ['sampler','activation_checkpointing','lora']:
 if old.get(key):cmd+=['--'+key.replace('_','-')]
sys.argv=cmd;runpy.run_module('lab.train',run_name='__main__')
